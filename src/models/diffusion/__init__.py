from typing import Union

from .base import SelfForcingConfig, WorldModelDenoiser
from .denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from .flow_denoiser import FlowDenoiser, FlowDenoiserConfig
from .inner_model import InnerModelConfig
from .diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from .flow_sampler import FlowSampler, FlowSamplerConfig


def make_denoiser(cfg: Union[DenoiserConfig, FlowDenoiserConfig]) -> WorldModelDenoiser:
    return FlowDenoiser(cfg) if isinstance(cfg, FlowDenoiserConfig) else Denoiser(cfg)


def make_sampler(
    denoiser: WorldModelDenoiser, cfg: Union[DiffusionSamplerConfig, FlowSamplerConfig]
) -> Union[DiffusionSampler, FlowSampler]:
    if isinstance(denoiser, FlowDenoiser):
        assert isinstance(cfg, FlowSamplerConfig), "Flow denoisers need a FlowSamplerConfig (use sampler=flow)"
        return FlowSampler(denoiser, cfg)
    assert isinstance(cfg, DiffusionSamplerConfig), "The EDM denoiser needs a DiffusionSamplerConfig (use sampler=edm)"
    return DiffusionSampler(denoiser, cfg)


def sampler_config_with_steps(
    cfg: Union[DiffusionSamplerConfig, FlowSamplerConfig], num_steps: int
) -> Union[DiffusionSamplerConfig, FlowSamplerConfig]:
    """Copy of a sampler config with another number of function evaluations."""
    from dataclasses import replace

    if isinstance(cfg, FlowSamplerConfig):
        return replace(cfg, num_steps=num_steps)
    return replace(cfg, num_steps_denoising=num_steps)
