import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from paths import ROOT
import trainer as upstream
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
import baseline as base
from atari_suite import predict


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)



class OfflineTrainer(upstream.Trainer):
    def train_component(self, name, steps):
        assert name == "denoiser"
        begin = time.monotonic()
        logs = super().train_component(name, steps)
        values = [{key: float(value) if isinstance(value, (torch.Tensor, np.number)) else value
                   for key, value in row.items()} for row in logs]
        for row in values:
            assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float))), row
        summary = dict(epoch=self.epoch, updates=self.num_batch_train.denoiser,
                       seconds=time.monotonic() - begin,
                       loss=float(np.mean([v["denoiser/train/loss_denoising"] for v in values])))
        with Path("training-progress.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        emit("training_epoch", **summary)
        return logs

    def save_checkpoint(self):
        super().save_checkpoint()
        if self.epoch and (self.epoch == 1 or self.epoch % 25 == 0 or self.epoch == self._cfg.training.num_final_epochs):
            self.validate_generator()

    @torch.no_grad()
    def validate_generator(self):
        model = self.agent.denoiser
        was_training = model.training
        model.eval()
        data = ROOT / f"datasets/{self._cfg.env.train.id.removesuffix('NoFrameskip-v4')}-v1"
        records = json.loads((data / "data-manifest.json").read_text())["episodes"]
        episodes = base.load_episodes(data, records, "validation")
        obs, act, ids = base.evaluation_windows(episodes, 2, 2, 543)
        padded_actions = torch.cat([act, torch.zeros(len(act), 1, device="cuda", dtype=torch.long)], dim=1)
        batch = base.batch_from(obs, padded_actions)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(995)
            loss = float(model(batch)[0])
            assert np.isfinite(loss)
            metrics = {}
            long_obs, long_act, long_ids = base.evaluation_windows(episodes, 1, 100, 606)
            long_obs, long_act, long_ids = long_obs[:8], long_act[:8], long_ids[:8]
            for steps in (1, 3):
                sampler = DiffusionSampler(model, DiffusionSamplerConfig(num_steps_denoising=steps))
                torch.manual_seed(505)
                next_frame = predict(sampler, obs[:, :4], act[:, :4])
                one = base.aggregate(base.scores(next_frame, obs[:, 4], obs[:, 3]))
                torch.manual_seed(707)
                context = long_obs[:, :4].clone()
                for t in range(100):
                    pred = predict(sampler, context, long_act[:, t:t+4])
                    context = torch.cat([context[:, 1:], pred[:, None]], dim=1)
                metrics[str(steps)] = dict(next_frame=one, frame100=base.aggregate(base.scores(pred, long_obs[:, 103], long_obs[:, 102])))
        report = dict(epoch=self.epoch, updates=self.num_batch_train.denoiser, loss=loss,
                      metrics=metrics, window_ids=ids, rollout_window_ids=long_ids)
        with Path("validation-progress.jsonl").open("a") as f:
            f.write(json.dumps(report) + "\n")
        emit("validation", epoch=self.epoch, updates=report["updates"], loss=loss, metrics=metrics)
        model.train(was_training)



def verify_batch(trainer):
    batch = next(iter(trainer._data_loader_train.denoiser)).to(torch.device("cuda"))
    assert batch.obs.shape == (32, 6, 3, 64, 64)
    assert batch.act.shape == batch.mask_padding.shape == (32, 6)
    model = trainer.agent.denoiser
    contexts, predictions, actions = [], [], []
    def before(module, args):
        contexts.append(args[2].detach().clone())
        actions.append(args[3].detach().clone())
    handle = model.inner_model.register_forward_pre_hook(before)
    original_wrap = model.wrap_model_output
    def wrap(*args):
        result = original_wrap(*args)
        predictions.append(result.detach().clone())
        return result
    model.wrap_model_output = wrap
    try:
        model.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        assert len(contexts) == len(predictions) == len(actions) == 2
        assert torch.equal(contexts[0].reshape(32, 4, 3, 64, 64), batch.obs[:, :4] / model.cfg.sigma_data)
        assert torch.equal(contexts[1].reshape(32, 4, 3, 64, 64)[:, -1], predictions[0] / model.cfg.sigma_data)
        assert torch.equal(actions[0], batch.act[:, :4]) and torch.equal(actions[1], batch.act[:, 1:5])
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        base.save_json(Path("batch-parity.json"), dict(status="passed", shape=list(batch.obs.shape),
            targets=2, generated_context_verified=True, action_alignment_verified=True, finite_gradients=True))
    finally:
        handle.remove()
        del model.wrap_model_output
        model.zero_grad()



def data_view(run, game):
    native = ROOT / f"datasets/{game}-native-v2"
    assert (native / "complete.json").exists()
    view = run / "native_data"
    for folder, source_split in (("train", "train"), ("test", "validation")):
        dest = view / folder
        dest.mkdir(parents=True)
        for source in (native / source_split).iterdir():
            if source.is_dir():
                (dest / source.name).symlink_to(source, target_is_directory=True)
            elif source.name == "info.pt":
                shutil.copy2(source, dest / source.name)
    return view
