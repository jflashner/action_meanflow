"""
Flow-based next-frame generators sharing DIAMOND's U-Net: flow matching (FM), MeanFlow and shortcut models.

Convention (as in MeanFlow): z_t = (1 - t) x + t eps, with t = 0 data and t = 1 noise, and velocity v = eps - x.
The network is u(z_t, t, h): the average velocity on the interval [r, t] with h = t - r, so that z_r = z_t - h u.
  - FM:       h = 0 only, u is the instantaneous velocity (Euler sampling with a few steps).
  - MeanFlow: u is trained with the MeanFlow identity u = v - h du/dt (Geng et al., 2025), du/dt by JVP.
  - Shortcut: u(z_t, t, 2d) is trained to match two half steps of size d (Frans et al., 2024).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from .base import add_dims, extra_conds, masked_mean, quantize, WorldModelDenoiser
from .inner_model import InnerModelConfig


OBJECTIVES = ("fm", "meanflow", "shortcut")


@dataclass
class FlowDenoiserConfig:
    inner_model: InnerModelConfig
    objective: str  # fm | meanflow | shortcut
    sigma_data: float = 0.5  # only used to rescale the context frames, as in DIAMOND's EDM denoiser
    parameterization: str = "u"  # u: the network outputs u; x: it outputs a clean frame x and u = (z_t - x) / t
    loss_space: str = "u"  # u: error on u; x: error on x = z_t - t u (the u error scaled by t)
    t_eps: float = 0.05  # lower bound on t in the division of the x-parameterization
    time_loc: float = -0.4  # logit-normal time distribution, equivalent to DIAMOND's log-normal sigma distribution
    time_scale: float = 1.2
    adaptive_weight_p: float = 0.0  # adaptive loss weighting of MeanFlow, loss / (loss + c)^p (0: plain MSE)
    adaptive_weight_c: float = 1e-3
    # MeanFlow
    meanflow_ratio: float = 0.25  # fraction of samples with r != t
    jvp: str = "autograd"  # autograd (torch.func.jvp, exact) or finite_difference
    jvp_fd_eps: float = 5e-3
    # Shortcut
    shortcut_bootstrap_frac: float = 0.25  # fraction of the batch used for self-consistency targets
    shortcut_log2_min_step: int = 7  # smallest step size 2^-7 = 1/128, used for the flow matching part
    # Context noise augmentation (see base.py)
    context_noise_max: float = 0.0
    context_noise_prob: float = 1.0
    context_noise_inference: float = 0.0

    def __post_init__(self) -> None:
        assert self.objective in OBJECTIVES, f"objective must be one of {OBJECTIVES}"
        assert self.parameterization in ("u", "x") and self.loss_space in ("u", "x")
        assert self.jvp in ("autograd", "finite_difference")
        self.inner_model.num_extra_conds = 1 + int(self.context_noise_max > 0)  # interval h (+ context noise level)


class FlowDenoiser(WorldModelDenoiser):
    def __init__(self, cfg: FlowDenoiserConfig) -> None:
        super().__init__(cfg)

    def setup_training(self, *args, **kwargs) -> None:
        pass  # no sigma distribution, the time distribution is part of the config

    def make_sampler(self, num_steps: int):
        from .flow_sampler import FlowSampler, FlowSamplerConfig

        return FlowSampler(self, FlowSamplerConfig(num_steps=num_steps))

    @property
    def min_step(self) -> float:
        return 2.0 ** -self.cfg.shortcut_log2_min_step

    @property
    def flow_interval(self) -> float:
        """Interval h at which the network predicts the instantaneous velocity."""
        return self.min_step if self.cfg.objective == "shortcut" else 0.0

    # Network

    def predict_u(
        self, z: Tensor, t: Tensor, h: Tensor, obs: Tensor, act: Tensor, ctx_level: Optional[Tensor] = None
    ) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        out = self.inner_model(z, t, rescaled_obs, act, extra_conds(h, ctx_level))
        if self.cfg.parameterization == "x":
            return (z - out) / add_dims(t.clamp(min=self.cfg.t_eps), z.ndim)
        return out

    # Sampling

    @torch.no_grad()
    def step(
        self, z: Tensor, t: float, r: float, obs: Tensor, act: Tensor, ctx_level: Optional[Tensor] = None
    ) -> Tensor:
        """One sampling step from time t to time r < t."""
        b = z.size(0)
        t_ = torch.full((b,), t, device=z.device)
        h = self.flow_interval if self.cfg.objective == "fm" else t - r
        u = self.predict_u(z, t_, torch.full((b,), h, device=z.device), obs, act, ctx_level)
        return z - (t - r) * u

    # Training

    def sample_time(self, n: int) -> Tensor:
        s = torch.randn(n, device=self.device) * self.cfg.time_scale + self.cfg.time_loc
        return s.sigmoid()

    def compute_loss(
        self,
        next_obs: Tensor,
        obs: Tensor,
        act: Tensor,
        mask: Tensor,
        ctx_level: Optional[Tensor],
        need_denoised: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Any]]:
        x = next_obs
        b = x.size(0)
        eps = torch.randn_like(x)
        u_fn = lambda z_, t_, h_, idx=slice(None): self.predict_u(
            z_, t_, h_, obs[idx], act[idx], None if ctx_level is None else ctx_level[idx]
        )

        if self.cfg.objective == "fm":
            t, h, z, target, logs = self._fm_targets(x, eps)
        elif self.cfg.objective == "meanflow":
            t, h, z, target, logs = self._meanflow_targets(x, eps, u_fn)
        else:
            t, h, z, target, logs = self._shortcut_targets(x, eps, u_fn)

        u = u_fn(z, t, h)
        err = u - target
        if self.cfg.loss_space == "x":
            err = err * add_dims(t, err.ndim)
        loss = masked_mean(self._weighted_loss(err), mask)
        logs["mse"] = masked_mean(err.detach().pow(2).flatten(1).mean(1), mask)  # the weighted loss is ~1 when p = 1

        denoised = None
        if need_denoised:
            # Posterior mean estimate x = z_t - t v, used as context for the next autoregressive step (as in EDM)
            with torch.no_grad():
                if self.cfg.objective == "fm":
                    v = u
                else:
                    v = u_fn(z, t, torch.full_like(t, self.flow_interval))
                denoised = quantize(z - add_dims(t, z.ndim) * v)

        return loss, denoised, logs

    def _weighted_loss(self, err: Tensor) -> Tensor:
        if self.cfg.adaptive_weight_p > 0:
            # MeanFlow's adaptive weighting on the per-sample squared l2 norm
            loss = err.pow(2).flatten(1).sum(1)
            weight = (loss.detach() + self.cfg.adaptive_weight_c) ** self.cfg.adaptive_weight_p
            return loss / weight
        return err.pow(2).flatten(1).mean(1)

    def _interpolate(self, x: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        t_ = add_dims(t, x.ndim)
        return (1 - t_) * x + t_ * eps

    def _fm_targets(self, x: Tensor, eps: Tensor):
        t = self.sample_time(x.size(0))
        h = torch.zeros_like(t)
        return t, h, self._interpolate(x, eps, t), eps - x, {}

    def _meanflow_targets(self, x: Tensor, eps: Tensor, u_fn: Callable):
        b = x.size(0)
        t, r = self.sample_time(b), self.sample_time(b)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        r = torch.where(torch.rand(b, device=x.device) < self.cfg.meanflow_ratio, r, t)
        h = t - r
        z = self._interpolate(x, eps, t)
        v = eps - x
        target = v.clone()

        # u_tgt = v - h du/dt, with du/dt = v d_z u + d_t u (r fixed, so dh/dt = 1). Only needed where h > 0.
        idx = (h > 0).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            fn = lambda z_, t_, h_: u_fn(z_, t_, h_, idx)
            z_i, t_i, h_i, v_i = z[idx], t[idx], h[idx], v[idx]
            with torch.no_grad():
                if self.cfg.jvp == "autograd":
                    _, dudt = torch.func.jvp(fn, (z_i, t_i, h_i), (v_i, torch.ones_like(t_i), torch.ones_like(h_i)))
                else:
                    e = self.cfg.jvp_fd_eps
                    with no_tf32():
                        dudt = (fn(z_i + e * v_i, t_i + e, h_i + e) - fn(z_i - e * v_i, t_i - e, h_i - e)) / (2 * e)
            target[idx] = v_i - add_dims(h_i, v_i.ndim) * dudt

        return t, h, z, target, {"meanflow_frac_r_neq_t": idx.numel() / b}

    def _shortcut_targets(self, x: Tensor, eps: Tensor, u_fn: Callable):
        b = x.size(0)
        k = self.cfg.shortcut_log2_min_step
        nb = int(round(b * self.cfg.shortcut_bootstrap_frac))

        # Flow matching part: smallest step size
        t = self.sample_time(b)
        h = torch.full_like(t, self.min_step)
        z = self._interpolate(x, eps, t)
        target = eps - x

        if nb > 0:
            # Self-consistency part: a step of size 2d must match two steps of size d, d in {1/2, ..., 2^-k}
            level = torch.randint(0, k, (nb,), device=x.device).float()  # big step 2d = 2^-level
            big = 2.0**-level
            num_sections = 2.0**level
            t_b = ((torch.rand(nb, device=x.device) * num_sections).floor() + 1) * big  # t in {2d, 4d, ..., 1}
            z_b = self._interpolate(x[:nb], eps[:nb], t_b)
            d = big / 2
            idx = torch.arange(nb, device=x.device)
            with torch.no_grad():
                u1 = u_fn(z_b, t_b, d, idx)
                z_mid = (z_b - add_dims(d, z_b.ndim) * u1).clamp(-4, 4)
                u2 = u_fn(z_mid, t_b - d, d, idx)
                target_b = ((u1 + u2) / 2).clamp(-4, 4)
            t = torch.cat((t_b, t[nb:]))
            h = torch.cat((big, h[nb:]))
            z = torch.cat((z_b, z[nb:]))
            target = torch.cat((target_b, target[nb:]))

        return t, h, z, target, {"shortcut_frac_bootstrap": nb / b}


class no_tf32:
    """Finite differences need full fp32 precision."""

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *args):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.prev
