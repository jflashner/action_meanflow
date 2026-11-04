"""
Evaluate a trained world model (any denoiser) on the test set of a static dataset:

  - next-frame quality from ground-truth context (PSNR, SSIM, LPIPS), for several numbers of sampling steps;
  - autoregressive rollouts with the recorded actions: PSNR / SSIM / LPIPS as a function of the rollout step;
  - action responsiveness: rollouts from the same starts and noise with uniformly random actions,
    delta-PSNR_t = PSNR(real actions) - PSNR(random actions) (Genie), and optionally the accuracy of an
    inverse dynamics model (src/train_idm.py) on generated transitions (MineWorld-style);
  - sampling speed (ms per frame and FPS at batch size 1).

    python src/eval_wm.py --run-dir runs/wm_meanflow/Boxing/seed1 --data-dir data/Boxing --nfe 1 2 --idm idm/Boxing/idm.pt

Writes <run-dir>/eval/<name>.json (and GIFs of a few rollouts).
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch import Tensor

from data import Dataset
from models.diffusion import make_denoiser, make_sampler, sampler_config_with_steps, WorldModelDenoiser
from utils import extract_state_dict, get_path_agent_ckpt
from wm_metrics import frame_metrics, LPIPS, next_frame_metrics, psnr, rollout, sample_segments


OmegaConf.register_new_resolver("eval", eval, replace=True)

SUMMARY_STEPS = (1, 5, 10, 25, 50, 100, 200, 300, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Static dataset folder (with a test/ subfolder).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--epoch", type=int, default=-1, help="Agent checkpoint epoch (-1: last).")
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--nfe", type=int, nargs="+", default=None, help="Sampling steps (default: from the run).")
    parser.add_argument("--num-next-frame", type=int, default=2048, help="Test transitions for next-frame metrics.")
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=300, help="Rollout length in frames.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--idm", type=Path, default=None, help="Inverse dynamics model checkpoint (train_idm.py).")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-speed", action="store_true")
    parser.add_argument("--num-gifs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None, help="Output name (default: derived from the options).")
    return parser.parse_args()


def load_world_model(run_dir: Path, epoch: int, weights: str, device: torch.device):
    cfg = OmegaConf.load(run_dir / "config" / "trainer.yaml")
    path_ckpt = get_path_agent_ckpt(run_dir / "checkpoints", epoch)
    sd = torch.load(path_ckpt, map_location=device, weights_only=False)
    prefix = "ema_denoiser" if weights == "ema" and any(k.startswith("ema_denoiser.") for k in sd) else "denoiser"
    if prefix != "ema_denoiser" and weights == "ema":
        print("No EMA weights in the checkpoint, using the raw weights.")
    sd = extract_state_dict(sd, prefix)
    denoiser_cfg = instantiate(cfg.agent.denoiser)
    denoiser_cfg.inner_model.num_actions = sd["inner_model.act_emb.0.weight"].size(0)
    denoiser = make_denoiser(denoiser_cfg).to(device).eval()
    denoiser.load_state_dict(sd)
    sampler_cfg = instantiate(cfg.world_model_env.diffusion_sampler)
    return cfg, denoiser, sampler_cfg, path_ckpt, prefix


def default_nfes(cfg) -> List[int]:
    gen = cfg.evaluation.get("generation")
    return list(gen.nfe) if gen is not None else [1]


@torch.no_grad()
def evaluate_rollouts(
    denoiser: WorldModelDenoiser,
    sampler_cfg,
    obs: Tensor,
    act: Tensor,
    horizon: int,
    num_actions: int,
    batch_size: int,
    lpips_fn: Optional[LPIPS],
    idm: Optional[torch.nn.Module],
    seed: int,
    num_gifs: int,
    gif_dir: Optional[Path],
) -> Dict:
    n = denoiser.cfg.inner_model.num_steps_conditioning
    device = denoiser.device
    rng = torch.Generator().manual_seed(seed)
    act_random = act.clone()
    act_random[:, n - 1 :] = torch.randint(0, num_actions, act_random[:, n - 1 :].shape, generator=rng)

    curves: Dict[str, List[Tensor]] = {}
    idm_correct: Dict[str, List[Tensor]] = {}
    for i in range(0, obs.size(0), batch_size):
        o = obs[i : i + batch_size].to(device)
        a = act[i : i + batch_size].to(device)
        a_rand = act_random[i : i + batch_size].to(device)
        gt = o[:, n : n + horizon]
        # Same noise for both rollouts (common random numbers), only the actions differ
        gen = rollout(denoiser, sampler_cfg, o[:, :n], a, horizon, seed=seed + i)
        gen_rand = rollout(denoiser, sampler_cfg, o[:, :n], a_rand, horizon, seed=seed + i)

        for k, v in frame_metrics(gen, gt, lpips_fn).items():
            curves.setdefault(k, []).append(v.cpu())
        curves.setdefault("psnr_random_actions", []).append(psnr(gen_rand, gt).cpu())

        if idm is not None:
            for key, frames, actions in (("gt", gt, a), ("real_actions", gen, a), ("random_actions", gen_rand, a_rand)):
                idm_correct.setdefault(key, []).append(idm_accuracy(idm, o[:, :n], frames, actions, n).cpu())

        if gif_dir is not None and i == 0:
            save_gifs(gt[:num_gifs], gen[:num_gifs], gen_rand[:num_gifs], gif_dir)

    curves = {k: torch.cat(v) for k, v in curves.items()}  # (num_rollouts, horizon)
    mean_curves = {k: v.mean(0) for k, v in curves.items()}
    mean_curves["delta_psnr"] = (curves["psnr"] - curves["psnr_random_actions"]).mean(0)

    summary = {}
    for k, v in mean_curves.items():
        for t in SUMMARY_STEPS:
            if t <= horizon:
                summary[f"{k}_t{t}"] = v[t - 1].item()
        for t in (16, 50, horizon):
            if t <= horizon:
                summary[f"{k}_mean{t}"] = v[:t].mean().item()
    out = {"summary": summary, "curves": {k: v.tolist() for k, v in mean_curves.items()}}
    if idm is not None:
        out["idm_accuracy"] = {k: torch.cat(v).float().mean().item() for k, v in idm_correct.items()}
    return out


def idm_accuracy(idm: torch.nn.Module, context: Tensor, frames: Tensor, act: Tensor, n: int) -> Tensor:
    """Whether the IDM recovers the action that produced each generated frame (frame n + j comes from act[n + j - 1])."""
    full = torch.cat((context, frames), dim=1)
    correct = []
    for start in range(0, frames.size(1), 50):  # chunks of 50 frames to bound memory
        js = range(start, min(start + 50, frames.size(1)))
        triplets = torch.stack([full[:, n + j - 2 : n + j + 1] for j in js], dim=1)  # (b, t, 3, c, h, w)
        labels = act[:, n - 1 + js.start : n - 1 + js.stop]
        logits = idm(triplets.flatten(0, 1))
        correct.append((logits.argmax(-1) == labels.flatten()).view(labels.shape))
    return torch.cat(correct, dim=1)


def save_gifs(gt: Tensor, gen: Tensor, gen_rand: Tensor, gif_dir: Path, scale: int = 4) -> None:
    """One GIF per rollout: ground truth | real actions | random actions."""
    gif_dir.mkdir(parents=True, exist_ok=True)
    to_uint8 = lambda x: x.add(1).div(2).mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    for k in range(gt.size(0)):
        rows = np.concatenate([to_uint8(gt[k]), to_uint8(gen[k]), to_uint8(gen_rand[k])], axis=2)  # (t, h, 3w, c)
        frames = [Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale), Image.Resampling.NEAREST) for f in rows]
        frames[0].save(gif_dir / f"rollout_{k}.gif", save_all=True, append_images=frames[1:], duration=67, loop=0)


@torch.no_grad()
def measure_speed(denoiser: WorldModelDenoiser, sampler_cfg, obs: Tensor, act: Tensor, iters: int = 100) -> Dict:
    n = denoiser.cfg.inner_model.num_steps_conditioning
    sampler = make_sampler(denoiser, sampler_cfg)
    o, a = obs[:1, :n].to(denoiser.device), act[:1, :n].to(denoiser.device)
    sync = torch.cuda.synchronize if o.is_cuda else (lambda: None)
    for _ in range(10):
        sampler.sample(o, a)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        sampler.sample(o, a)
    sync()
    ms = (time.perf_counter() - t0) / iters * 1000
    return {"ms_per_frame": ms, "fps": 1000 / ms}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training

    cfg, denoiser, sampler_cfg, path_ckpt, prefix = load_world_model(args.run_dir, args.epoch, args.weights, device)
    num_actions = denoiser.cfg.inner_model.num_actions
    n = denoiser.cfg.inner_model.num_steps_conditioning
    nfes = args.nfe if args.nfe is not None else default_nfes(cfg)

    dataset = Dataset(args.data_dir / args.split, f"{args.split}_dataset", cache_in_ram=True)
    dataset.load_from_default_path()
    assert dataset.num_episodes > 0, f"Empty dataset at {args.data_dir / args.split}"
    lpips_fn = None if args.no_lpips else LPIPS.create(device)
    idm = None
    if args.idm is not None:
        from train_idm import load_idm

        idm = load_idm(args.idm, device)

    next_frame_batch = sample_segments(dataset, args.num_next_frame, n + 1, seed=args.seed)
    horizon = min(args.horizon, int(max(dataset.lengths)) - n)
    rollout_batch = sample_segments(dataset, args.num_rollouts, n + horizon, seed=args.seed + 1)

    name = args.name or f"{args.split}_{'ema' if prefix == 'ema_denoiser' else 'raw'}_h{horizon}"
    out_dir = args.run_dir / "eval"
    results = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(path_ckpt),
        "weights": prefix,
        "game": cfg.env.train.id.replace("NoFrameskip-v4", ""),
        "seed": cfg.common.seed,
        "denoiser": OmegaConf.to_container(cfg.agent.denoiser, resolve=True),
        "self_forcing": OmegaConf.to_container(cfg.denoiser.training.get("self_forcing") or {}, resolve=True),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "num_next_frame": args.num_next_frame,
        "num_rollouts": args.num_rollouts,
        "horizon": horizon,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "nfe": {},
    }
    for nfe in nfes:
        print(f"\n=== {nfe} sampling step(s) ===")
        cfg_nfe = sampler_config_with_steps(sampler_cfg, nfe)
        res = {"next_frame": next_frame_metrics(denoiser, cfg_nfe, next_frame_batch, [nfe], args.batch_size, lpips_fn, args.seed)}
        res["next_frame"] = {k.replace(f"_nfe{nfe}", ""): v for k, v in res["next_frame"].items()}
        print("next frame:", {k: round(v, 4) for k, v in res["next_frame"].items()})
        res["rollout"] = evaluate_rollouts(
            denoiser, cfg_nfe, rollout_batch.obs, rollout_batch.act, horizon, num_actions, args.batch_size,
            lpips_fn, idm, args.seed, args.num_gifs, out_dir / f"{name}_gifs_nfe{nfe}" if args.num_gifs > 0 else None,
        )
        s = res["rollout"]["summary"]
        print("rollout:", {k: round(v, 3) for k, v in s.items() if k.startswith(("psnr_t", "delta_psnr_t", "psnr_mean"))})
        if "idm_accuracy" in res["rollout"]:
            print("IDM accuracy:", res["rollout"]["idm_accuracy"])
        if not args.no_speed:
            res["speed"] = measure_speed(denoiser, cfg_nfe, rollout_batch.obs, rollout_batch.act)
            print("speed:", res["speed"])
        results["nfe"][str(nfe)] = res

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.json").open("w") as f:
        json.dump(results, f)
    print(f"\nSaved {out_dir / f'{name}.json'}")


if __name__ == "__main__":
    main()
