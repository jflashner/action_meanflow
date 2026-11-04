"""
Image metrics and generation utilities for evaluating world models (used by the trainer and by eval_wm.py).

All frames are in [-1, 1], with shape (..., c, h, w).
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from data import Batch, collate_segments_to_batch, Dataset, make_segment, SegmentId
from models.diffusion import (
    DiffusionSamplerConfig,
    FlowSamplerConfig,
    make_sampler,
    sampler_config_with_steps,
    WorldModelDenoiser,
)


SamplerConfig = Union[DiffusionSamplerConfig, FlowSamplerConfig]


# Metrics (one value per sample)


def psnr(pred: Tensor, target: Tensor) -> Tensor:
    mse = (pred - target).div(2).pow(2).flatten(-3).mean(-1)  # images in [0, 1]
    return 10 * torch.log10(1 / mse.clamp(min=1e-10))


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(pred: Tensor, target: Tensor) -> Tensor:
    shape = pred.shape[:-3]
    c, h, w = pred.shape[-3:]
    x = pred.reshape(-1, c, h, w).add(1).div(2)
    y = target.reshape(-1, c, h, w).add(1).div(2)
    win = _gaussian_window(device=x.device).expand(c, 1, 11, 11)
    filt = lambda z: F.conv2d(z, win, groups=c)
    mu_x, mu_y = filt(x), filt(y)
    sigma_x = filt(x * x) - mu_x**2
    sigma_y = filt(y * y) - mu_y**2
    sigma_xy = filt(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return s.flatten(1).mean(1).reshape(shape)


class LPIPS:
    """Wrapper around the `lpips` package (AlexNet). `LPIPS.create()` returns None if it is not installed."""

    def __init__(self, device: torch.device) -> None:
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @classmethod
    def create(cls, device: torch.device) -> Optional["LPIPS"]:
        try:
            return cls(device)
        except Exception as e:  # missing package or weights
            print(f"LPIPS disabled ({e.__class__.__name__}: {e})")
            return None

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        shape = pred.shape[:-3]
        c, h, w = pred.shape[-3:]
        d = self.net(pred.reshape(-1, c, h, w), target.reshape(-1, c, h, w), normalize=False)
        return d.reshape(shape)


def frame_metrics(
    pred: Tensor, target: Tensor, lpips_fn: Optional[LPIPS] = None, chunk_size: int = 2048
) -> Dict[str, Tensor]:
    shape = pred.shape[:-3]
    pred, target = pred.reshape(-1, *pred.shape[-3:]), target.reshape(-1, *target.shape[-3:])
    out: Dict[str, List[Tensor]] = {}
    for i in range(0, pred.size(0), chunk_size):  # bounded memory for long rollouts
        p, t = pred[i : i + chunk_size], target[i : i + chunk_size]
        out.setdefault("psnr", []).append(psnr(p, t))
        out.setdefault("ssim", []).append(ssim(p, t))
        if lpips_fn is not None:
            out.setdefault("lpips", []).append(lpips_fn(p, t))
    return {k: torch.cat(v).reshape(shape) for k, v in out.items()}


# Test data


def sample_segments(dataset: Dataset, num: int, seq_length: int, seed: int = 0) -> Batch:
    """Fixed random segments (no padding) of `seq_length` frames, episodes weighted by length."""
    rng = np.random.RandomState(seed)
    lengths = np.array(dataset.lengths)
    valid = np.nonzero(lengths >= seq_length)[0]
    assert len(valid) > 0, f"No episode with at least {seq_length} frames in {dataset.name}"
    p = (lengths[valid] - seq_length + 1).astype(np.float64)
    episode_ids = rng.choice(valid, size=num, p=p / p.sum())
    segments = []
    for episode_id in episode_ids:
        episode = dataset.load_episode(int(episode_id))
        start = rng.randint(0, len(episode) - seq_length + 1)
        segments.append(make_segment(episode, SegmentId(int(episode_id), start, start + seq_length), should_pad=False))
    return collate_segments_to_batch(segments)


# Generation


@torch.no_grad()
def rollout(
    denoiser: WorldModelDenoiser,
    sampler_cfg: SamplerConfig,
    obs: Tensor,
    act: Tensor,
    horizon: int,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Autoregressive generation of `horizon` frames.

    obs: (b, n, c, h, w) ground-truth context. act: (b, n + horizon - 1), act[:, i] is the action taken at frame i,
    so that frame n + j is generated from frames j ... n + j - 1 and actions j ... n + j - 1.
    """
    n = denoiser.cfg.inner_model.num_steps_conditioning
    assert obs.size(1) == n and act.size(1) >= n + horizon - 1
    sampler = make_sampler(denoiser, sampler_cfg)
    generator_state = torch.random.get_rng_state(), (torch.cuda.get_rng_state() if obs.is_cuda else None)
    if seed is not None:
        torch.manual_seed(seed)
    buffer = obs.clone()
    frames = []
    for j in range(horizon):
        next_obs, _ = sampler.sample(buffer[:, -n:], act[:, j : j + n])
        frames.append(next_obs)
        buffer = torch.cat((buffer[:, 1:], next_obs.unsqueeze(1)), dim=1)
    if seed is not None:
        torch.random.set_rng_state(generator_state[0])
        if generator_state[1] is not None:
            torch.cuda.set_rng_state(generator_state[1])
    return torch.stack(frames, dim=1)


@torch.no_grad()
def next_frame_metrics(
    denoiser: WorldModelDenoiser,
    sampler_cfg: SamplerConfig,
    batch: Batch,
    nfes: Sequence[int],
    batch_size: int = 64,
    lpips_fn: Optional[LPIPS] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """One-step prediction from ground-truth context (teacher forcing), for each number of function evaluations."""
    n = denoiser.cfg.inner_model.num_steps_conditioning
    out = {}
    for nfe in nfes:
        cfg = sampler_config_with_steps(sampler_cfg, nfe)
        values: Dict[str, List[Tensor]] = {}
        for i in range(0, batch.obs.size(0), batch_size):
            obs = batch.obs[i : i + batch_size].to(denoiser.device)
            act = batch.act[i : i + batch_size].to(denoiser.device)
            pred = rollout(denoiser, cfg, obs[:, :n], act[:, :n], horizon=1, seed=seed + i)[:, 0]
            for k, v in frame_metrics(pred, obs[:, n], lpips_fn).items():
                values.setdefault(k, []).append(v.cpu())
        for k, v in values.items():
            out[f"{k}_nfe{nfe}"] = torch.cat(v).mean().item()
    return out
