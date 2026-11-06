import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import offline_edm as edm
import baseline as base
import atari_suite
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
from utils import set_seed

from paths import ROOT, CODE_ROOT, CONFIG_ROOT, SCRIPT_ROOT
EXPERIMENT = 'edm-uniform-100x1000-v1'


class UniformTrainer(edm.OfflineTrainer):
    def train_component(self, name, steps):
        assert name == 'denoiser'
        assert steps == self._cfg.denoiser.training.steps_per_epoch
        before = self.num_batch_train.denoiser
        started = time.monotonic()
        logs = edm.upstream.Trainer.train_component(self, name, steps)
        assert self.num_batch_train.denoiser - before == steps
        assert self.num_batch_train.denoiser == self.epoch * steps
        rows = [{k: float(v) if isinstance(v, (torch.Tensor, np.number)) else v
                 for k, v in row.items()} for row in logs]
        assert all(np.isfinite(v) for row in rows for v in row.values() if isinstance(v, (int, float)))
        report = dict(epoch=self.epoch, updates=self.num_batch_train.denoiser,
                      updates_this_epoch=steps, seconds=time.monotonic() - started,
                      loss=float(np.mean([r['denoiser/train/loss_denoising'] for r in rows])),
                      gradient_norm_mean=float(np.mean([r['denoiser/train/grad_norm_before_clip'] for r in rows])),
                      gradient_norm_max=float(np.max([r['denoiser/train/grad_norm_before_clip'] for r in rows])),
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        with Path('training-progress.jsonl').open('a') as f:
            f.write(json.dumps(report) + '\n')
        base.emit('training_epoch', **report)
        return logs

    def save_checkpoint(self):
        edm.upstream.Trainer.save_checkpoint(self)
        if self.epoch and (self.epoch == 1 or self.epoch % 10 == 0 or self.epoch == self._cfg.training.num_final_epochs):
            self.validate_generator()


@torch.no_grad()
def check_saved_model(model, sampler_class, source, run):
    model.eval()
    restored = type(model)(model.cfg).cuda().eval()
    saved = torch.load(run / 'checkpoint-final.pt', map_location='cuda', weights_only=True)
    restored.load_state_dict(saved['model'], strict=True)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert all(torch.isfinite(v).all() for v in restored.state_dict().values())
    records = json.loads((source / 'data-manifest.json').read_text())['episodes']
    episodes = base.load_episodes(source, records, 'validation')
    obs, act, _ = base.evaluation_windows(episodes, 1, 1, 606)
    cfg = DiffusionSamplerConfig(num_steps_denoising=1)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(911)
        before, _ = sampler_class(model, cfg).sample(obs[:1, :4], act[:1, :4])
        torch.manual_seed(911)
        after, _ = sampler_class(restored, cfg).sample(obs[:1, :4], act[:1, :4])
    assert torch.isfinite(before).all()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    base.save_json(run / 'checkpoint-check.json', dict(status='passed', identical_reloaded_prediction=True))



def verify_backbone():
    expected = json.loads((CONFIG_ROOT / "backbone-sha256.json").read_text())
    for relative, digest in expected["files"].items():
        actual = hashlib.sha256((CODE_ROOT / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Upstream backbone/config changed: {relative}")



def build_config(game, seed, native, name, smoke=False):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with initialize_config_dir(config_dir=str(CODE_ROOT / 'config'), version_base='1.3'):
        cfg = compose(config_name='trainer')
    epochs, steps = (2, 10) if smoke else (100, 1000)
    cfg.env.train.id = f'{game}NoFrameskip-v4'
    cfg.common.devices = 'all'
    cfg.common.seed = seed
    cfg.static_dataset.path = str(native)
    cfg.training.num_final_epochs = epochs
    cfg.denoiser.training.steps_first_epoch = steps
    cfg.denoiser.training.steps_per_epoch = steps
    cfg.training.compile_wm = False
    cfg.evaluation.should = False
    cfg.rew_end_model.training.start_after_epochs = 1000000000
    cfg.actor_critic.training.start_after_epochs = 1000000000
    cfg.checkpointing.save_agent_every = 10
    cfg.checkpointing.num_to_keep = 11
    cfg.wandb.mode = 'offline'
    cfg.wandb.project = EXPERIMENT
    cfg.wandb.name = f'EDM-{name}'
    OmegaConf.resolve(cfg)
    assert cfg.denoiser.training.batch_size == 32
    assert cfg.denoiser.training.steps_first_epoch == cfg.denoiser.training.steps_per_epoch
    assert cfg.training.num_final_epochs * steps == (20 if smoke else 100000)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', choices=['Breakout', 'Boxing'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1
    verify_backbone()
    name = f'{args.game}-seed{args.seed}' + ('-smoke' if args.smoke else '')
    run = ROOT / 'results' / EXPERIMENT / name
    run.mkdir(parents=True, exist_ok=False)
    native = edm.data_view(run, args.game)
    cfg = build_config(args.game, args.seed, native, name, args.smoke)
    epochs = cfg.training.num_final_epochs
    steps = cfg.denoiser.training.steps_per_epoch
    (run / '.hydra').mkdir()
    OmegaConf.save(cfg, run / '.hydra/config.yaml')
    hashes = {}
    snapshot = run / 'source-snapshot'
    snapshot.mkdir()
    for filename in ['train.py', 'paths.py', 'offline_edm.py', 'baseline.py', 'atari_suite.py']:
        source_file = SCRIPT_ROOT / filename
        shutil.copy2(source_file, snapshot / filename)
        hashes[filename] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    base.save_json(run / 'source-hashes.json', hashes)
    base.save_json(run / 'code-version.json', dict(commit=subprocess.check_output(
        ['git', '-C', str(CODE_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        dirty=bool(subprocess.check_output(['git', '-C', str(CODE_ROOT), 'status', '--porcelain'], text=True).strip())))
    shutil.copy2(CONFIG_ROOT / 'protocol.json', run / 'experiment-protocol.json')
    source = ROOT / f'datasets/{args.game}-v1'
    shutil.copy2(source / 'data-manifest.json', run / 'data-manifest.json')
    (run / 'data').symlink_to(source / 'data', target_is_directory=True)
    base.save_json(run / 'hardware.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('HOSTNAME'), seed=args.seed))
    sampler_class = DiffusionSampler
    edm.upstream.set_seed = lambda ignored: set_seed(args.seed)
    os.chdir(run)
    trainer = UniformTrainer(cfg, CODE_ROOT)
    if args.smoke:
        with torch.random.fork_rng(devices=[0]):
            edm.verify_batch(trainer)
    frozen = {name: {k: v.detach().clone() for k, v in getattr(trainer.agent, name).state_dict().items()}
              for name in ('actor_critic', 'rew_end_model')}
    base.save_json(run / 'architecture.json', dict(method='EDM',
        parameters=sum(p.numel() for p in trainer.agent.denoiser.parameters())))
    started = time.monotonic()
    trainer.run()
    assert trainer.epoch == epochs and trainer.num_batch_train.denoiser == epochs * steps
    assert trainer.num_batch_train.actor_critic == trainer.num_batch_train.rew_end_model == 0
    for name, before in frozen.items():
        assert all(torch.equal(v, before[k]) for k, v in getattr(trainer.agent, name).state_dict().items())
    training = dict(epochs=epochs, updates_per_epoch=steps, updates=epochs * steps,
                    seconds=time.monotonic() - started, actor_updates=0, reward_updates=0)
    base.save_json(run / 'training-summary.json', training)
    torch.save(dict(model=trainer.agent.denoiser.state_dict(), updates=epochs * steps), run / 'checkpoint-final.pt')
    if args.smoke:
        check_saved_model(trainer.agent.denoiser, sampler_class, source, run)
    else:
        eval_cfg = json.loads((CONFIG_ROOT / 'protocol.json').read_text())
        atari_suite.evaluate(trainer.agent.denoiser, eval_cfg, run,
                             json.loads((source / 'data-manifest.json').read_text())['episodes'])
    base.save_json(run / 'complete.json', dict(status='complete', method='EDM', game=args.game,
        seed=args.seed, training=training, smoke=args.smoke))
    base.emit('run_complete', method='EDM', game=args.game, seed=args.seed, training=training)


if __name__ == '__main__':
    main()
