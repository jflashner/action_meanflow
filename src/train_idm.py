"""
Inverse dynamics model (IDM) for measuring action responsiveness of world models (MineWorld-style): it predicts
the action a_t from frames (o_{t-1}, o_t, o_{t+1}). Trained on the real train set; eval_wm.py then checks whether
it recovers the commanded actions from generated rollouts. Its accuracy on real test transitions is the reference
(some Atari actions have no visible effect in some states, so it is below 100%).

    python src/train_idm.py --data-dir data/Boxing --out idm/Boxing/idm.pt
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from data import Dataset


class IDM(nn.Module):
    def __init__(self, num_actions: int, img_channels: int = 3, channels: Tuple[int, ...] = (64, 128, 256, 256)) -> None:
        super().__init__()
        self.config = {"num_actions": num_actions, "img_channels": img_channels, "channels": list(channels)}
        layers, c_in = [], 3 * img_channels
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.GroupNorm(max(1, c // 32), c), nn.SiLU()]
            c_in = c
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(c_in, 256), nn.SiLU(), nn.Linear(256, num_actions))

    def forward(self, frames: Tensor) -> Tensor:
        """frames: (b, 3, c, h, w) in [-1, 1], the frames before, at and after the action."""
        b, t, c, h, w = frames.shape
        return self.head(self.encoder(frames.reshape(b, t * c, h, w)))


def load_idm(path: Path, device: torch.device) -> IDM:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = IDM(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model


def load_transitions(dataset: Dataset, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """All frames as uint8 on device, actions, and the indices t with t-1, t, t+1 in the same episode."""
    obs, act, idx = [], [], []
    offset = 0
    for episode_id in range(dataset.num_episodes):
        episode = dataset.load_episode(episode_id)
        obs.append(episode.obs.add(1).div(2).mul(255).round().byte())
        act.append(episode.act)
        idx.append(np.arange(1, len(episode) - 1) + offset)
        offset += len(episode)
    return torch.cat(obs).to(device), torch.cat(act).to(device), torch.from_numpy(np.concatenate(idx)).to(device)


def get_batch(obs: Tensor, act: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    frames = torch.stack((obs[t - 1], obs[t], obs[t + 1]), dim=1).float().div(255).mul(2).sub(1)
    return frames, act[t]


@torch.no_grad()
def evaluate(model: IDM, obs: Tensor, act: Tensor, idx: Tensor, num_actions: int, batch_size: int = 1024) -> dict:
    model.eval()
    preds = []
    for i in range(0, len(idx), batch_size):
        frames, _ = get_batch(obs, act, idx[i : i + batch_size])
        preds.append(model(frames).argmax(-1))
    preds, labels = torch.cat(preds), act[idx]
    recall = [(preds[labels == a] == a).float().mean().item() for a in range(num_actions) if (labels == a).any()]
    model.train()
    return {"accuracy": (preds == labels).float().mean().item(), "balanced_accuracy": float(np.mean(recall))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Train episodes held out to pick the checkpoint.")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {}
    for split in ("train", "test"):
        dataset = Dataset(args.data_dir / split, split, cache_in_ram=False)
        dataset.load_from_default_path()
        splits[split] = load_transitions(dataset, device)
    # Validation: the transitions of the last train episodes (indices are sorted by episode)
    obs, act, idx = splits["train"]
    n_val = max(1, int(len(idx) * args.val_frac))
    splits["train"], splits["val"] = (obs, act, idx[:-n_val]), (obs, act, idx[-n_val:])
    num_actions = int(json.loads((args.data_dir / "collection_info.json").read_text())["num_actions"])

    model = IDM(num_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    obs, act, idx = splits["train"]
    best, best_state = -1.0, None
    for step in trange(args.steps, desc="Training IDM"):
        t = idx[torch.randint(0, len(idx), (args.batch_size,), device=device)]
        frames, labels = get_batch(obs, act, t)
        loss = F.cross_entropy(model(frames), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val = evaluate(model, *splits["val"], num_actions)
            if val["accuracy"] > best:
                best, best_state = val["accuracy"], copy.deepcopy(model.state_dict())
            if (step + 1) % (5 * args.eval_every) == 0:
                print(f"step {step + 1}: loss {loss.item():.3f}, val {val}")

    model.load_state_dict(best_state)
    metrics = {split: evaluate(model, *splits[split], num_actions) for split in ("train", "val", "test")}
    print(metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "metrics": metrics}, args.out)
    with args.out.with_suffix(".json").open("w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
