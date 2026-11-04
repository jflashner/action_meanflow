from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from .base import quantize
from .flow_denoiser import FlowDenoiser


@dataclass
class FlowSamplerConfig:
    num_steps: int = 1  # number of function evaluations, uniform time grid from t = 1 (noise) to t = 0 (data)


class FlowSampler:
    def __init__(self, denoiser: FlowDenoiser, cfg: FlowSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.times = torch.linspace(1, 0, cfg.num_steps + 1).tolist()

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        prev_obs, ctx_level = self.denoiser.augment_context(prev_obs, inference=True)
        z = torch.randn(b, c, h, w, device=device)
        trajectory = [z]
        for t_, r_ in zip(self.times[:-1], self.times[1:]):
            z = self.denoiser.step(z, t_, r_, prev_obs, prev_act, ctx_level)
            trajectory.append(z)
        return quantize(z), trajectory
