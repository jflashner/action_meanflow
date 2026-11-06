import io
import json
from pathlib import Path
import sys

import torch
from hydra.utils import instantiate

from train import build_config, verify_backbone
from baseline import scores
from data import Batch
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig


def main():
    torch.set_num_threads(2)
    verify_backbone()
    reports = []
    for game, actions, parameters in [('Breakout', 4, 4405955), ('Boxing', 18, 4406851)]:
        for seed in (0, 1, 2):
            seeded = build_config(game, seed, Path('/tmp/unused-native-data'), f'{game}-seed{seed}')
            assert seeded.common.seed == seed
            assert seeded.training.num_final_epochs * seeded.denoiser.training.steps_per_epoch == 100000
        cfg = build_config(game, 0, Path('/tmp/unused-native-data'), game)
        assert cfg.training.num_final_epochs == 100
        assert cfg.denoiser.training.steps_first_epoch == cfg.denoiser.training.steps_per_epoch == 1000
        assert cfg.denoiser.training.batch_size == 32
        assert cfg.denoiser.training.num_autoregressive_steps == 1
        assert cfg.actor_critic.training.start_after_epochs > 100
        assert cfg.rew_end_model.training.start_after_epochs > 100
        smoke = build_config(game, 0, Path('/tmp/unused-native-data'), game, smoke=True)
        assert smoke.training.num_final_epochs == 2
        assert smoke.denoiser.training.steps_first_epoch == smoke.denoiser.training.steps_per_epoch == 10
        torch.manual_seed(0)
        model_cfg = instantiate(cfg.agent, num_actions=actions).denoiser
        model = Denoiser(model_cfg)
        assert sum(p.numel() for p in model.parameters()) == parameters
        model.setup_training(instantiate(cfg.denoiser.sigma_distribution))
        obs = torch.rand(1, 6, 3, 64, 64).mul(2).sub(1)
        act = torch.randint(actions, (1, 6))
        zeros = torch.zeros(1, 6, dtype=torch.long)
        batch = Batch(obs, act, zeros.float(), zeros, zeros, torch.ones(1, 6, dtype=torch.bool), [], [])
        loss, _ = model(batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        model.eval()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = Denoiser(model_cfg).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        for steps in (1, 3):
            torch.manual_seed(505)
            output, _ = DiffusionSampler(model, DiffusionSamplerConfig(steps)).sample(obs[:, :4], act[:, :4])
            torch.manual_seed(505)
            reloaded, _ = DiffusionSampler(restored, DiffusionSamplerConfig(steps)).sample(obs[:, :4], act[:, :4])
            assert output.shape == (1, 3, 64, 64) and torch.isfinite(output).all()
            torch.testing.assert_close(output, reloaded, rtol=0, atol=0)
        reports.append(dict(game=game, parameters=parameters, finite_loss=float(loss.detach()), checkpoint_roundtrip=True))
    values = scores(torch.ones(1, 3, 4, 4), torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4))
    assert values['mse'] == [0.25]
    assert abs(values['psnr_db'][0] - 6.020599913) < 1e-5
    assert not any('meanflow' in name.lower() for name in sys.modules)
    print(json.dumps(dict(status='passed', device='cpu', games=reports, meanflow_dependency=False)))


if __name__ == '__main__':
    main()
