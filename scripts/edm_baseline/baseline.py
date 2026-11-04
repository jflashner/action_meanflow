import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import ale_py
import gymnasium as gym
from paths import ROOT
from data import Batch
from envs.atari_preprocessing import AtariPreprocessing
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)



def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)



def load_episodes(run, records, split):
    episodes = []
    for record in records:
        if record["split"] == split:
            with np.load(run / record["path"]) as f:
                episodes.append(dict(id=record["id"], obs=f["obs"], act=f["act"]))
    return episodes



def normalized(a):
    return torch.as_tensor(a, device="cuda").float().div(127.5).sub(1).permute(0, 1, 4, 2, 3).contiguous()



def batch_from(obs, act):
    b, t = act.shape
    return Batch(obs, act, torch.zeros(b, t, device="cuda"),
        torch.zeros(b, t, device="cuda", dtype=torch.long),
        torch.zeros(b, t, device="cuda", dtype=torch.long),
        torch.ones(b, t, device="cuda", dtype=torch.bool), [], [])



def evaluation_windows(episodes, count, horizon, seed):
    rng = np.random.default_rng(seed)
    windows, actions, ids = [], [], []
    for e in episodes:
        available = len(e["act"]) - horizon - 2
        if available <= 0:
            continue
        for start in rng.choice(available, size=min(count, available), replace=False):
            windows.append(e["obs"][start:start + 4 + horizon])
            actions.append(e["act"][start:start + 3 + horizon])
            ids.append([e["id"], int(start)])
    assert windows, "No valid evaluation windows"
    return normalized(np.stack(windows)), torch.as_tensor(np.stack(actions), device="cuda"), ids



def scores(pred, target, previous):
    squared = ((pred - target) / 2).square()
    mse = squared.mean(dim=(1, 2, 3))
    changed = ((target - previous).abs().amax(dim=1, keepdim=True) > 2 / 255).expand_as(squared)
    count = changed.sum(dim=(1, 2, 3))
    moving = (squared * changed).sum(dim=(1, 2, 3)) / count.clamp_min(1)
    return {"mse": mse.cpu().tolist(), "psnr_db": (-10 * mse.clamp_min(1e-12).log10()).cpu().tolist(),
            "changed_pixel_mse": [v if n else None for v, n in zip(moving.cpu().tolist(), count.cpu().tolist())]}



def aggregate(values):
    return {k: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
            for k, v in values.items()}



def gif_comparison(path, truth, generated, initial):
    def pixels(x):
        return x.add(1).mul(127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    actual, predicted = pixels(truth), pixels(generated)
    copied = pixels(initial[None])[0]
    frames = []
    for a, b in zip(actual, predicted):
        frame = Image.new("RGB", (576, 216), "white")
        for j, (label, data) in enumerate(zip(["Ground truth", "DIAMOND", "Repeat frame"], [a, b, copied])):
            frame.paste(Image.fromarray(data).resize((192, 192), Image.Resampling.NEAREST), (j * 192, 24))
            ImageDraw.Draw(frame).text((j * 192 + 5, 5), label, fill="black")
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=67, loop=0)
