from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import add_dims, extra_conds, masked_mean, quantize, WorldModelDenoiser
from .inner_model import InnerModelConfig


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    context_noise_max: float = 0.0  # GameNGen-style noise augmentation of the context frames (0: disabled)
    context_noise_prob: float = 1.0
    context_noise_inference: float = 0.0

    def __post_init__(self) -> None:
        self.inner_model.num_extra_conds = int(self.context_noise_max > 0)


class Denoiser(WorldModelDenoiser):
    """EDM denoiser of DIAMOND."""

    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__(cfg)
        self.sample_sigma_training = None

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def make_sampler(self, num_steps: int):
        from .diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

        return DiffusionSampler(self, DiffusionSamplerConfig(num_steps_denoising=num_steps))

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1, 1))))

    def compute_model_output(
        self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, cs: Conditioners, ctx_level: Optional[Tensor] = None
    ) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act, extra_conds(ctx_level))

    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        return quantize(d)

    @torch.no_grad()
    def denoise(
        self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor, ctx_level: Optional[Tensor] = None
    ) -> Tensor:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs, ctx_level)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def compute_loss(
        self,
        next_obs: Tensor,
        obs: Tensor,
        act: Tensor,
        mask: Tensor,
        ctx_level: Optional[Tensor],
        need_denoised: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Any]]:
        b = next_obs.size(0)
        sigma = self.sample_sigma_training(b, self.device)
        noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs, ctx_level)

        target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
        loss = masked_mean(F.mse_loss(model_output, target, reduction="none").flatten(1).mean(1), mask)

        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return loss, denoised, {}
