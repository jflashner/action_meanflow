from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None
    num_extra_conds: int = 0  # extra scalar conditionings (e.g. MeanFlow interval, context noise level), set by the denoiser config


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
            nn.Flatten(),  # b t e -> b (t e)
        )
        # One embedding per extra scalar, summed with the noise embedding. Not created when unused, so that
        # the original DIAMOND checkpoints still load strictly.
        if cfg.num_extra_conds > 0:
            self.extra_embs = nn.ModuleList([FourierFeatures(cfg.cond_channels) for _ in range(cfg.num_extra_conds)])
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(
        self,
        noisy_next_obs: Tensor,
        c_noise: Tensor,
        obs: Tensor,
        act: Tensor,
        extra_conds: Optional[List[Tensor]] = None,
    ) -> Tensor:
        emb = self.noise_emb(c_noise) + self.act_emb(act)
        if extra_conds:
            assert len(extra_conds) == len(self.extra_embs)
            for c, extra_emb in zip(extra_conds, self.extra_embs):
                emb = emb + extra_emb(c)
        cond = self.cond_proj(emb)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
