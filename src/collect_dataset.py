"""
Collect a static dataset of Atari transitions for world model experiments, with DIAMOND's pretrained agents
(Hugging Face Hub, eloialonso/diamond). Every method is then trained and evaluated on the same data.

Each episode is played by an epsilon-greedy version of the agent, with epsilon drawn from --epsilons, to cover
both competent play and diverse (random) actions. Episodes are not cut at life loss, so test episodes are long
enough for long rollouts.

    python src/collect_dataset.py --game Boxing --out-dir data/Boxing

creates data/Boxing/{train,test}, usable as `static_dataset.path=data/Boxing`.
"""

import argparse
import json
from pathlib import Path
import random
from typing import List, Optional

from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from agent import Agent
from data import Dataset, Episode
from envs import make_atari_env
from utils import ATARI_100K_GAMES


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True, help="Atari 100k game, e.g. Boxing, Breakout, Pong.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=100_000, help="Number of transitions in the train set.")
    parser.add_argument("--test-episodes", type=int, default=30, help="Number of complete episodes in the test set.")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=4500, help="In agent steps (4 frames each).")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-ckpt", type=Path, default=None, help="Local agent checkpoint instead of DIAMOND's.")
    return parser.parse_args()


def load_pretrained_agent(game: str, num_actions: int, size: int, device: torch.device, ckpt: Optional[Path]) -> Agent:
    download = lambda f: Path(hf_hub_download(repo_id="eloialonso/diamond", filename=f))
    cfg = OmegaConf.create(
        {
            "agent": OmegaConf.load(download("atari_100k/config/agent/default.yaml")),
            "env": {"train": {"size": size}},
        }
    )
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(ckpt if ckpt is not None else download(f"atari_100k/models/{game}.pt"))
    return agent


@torch.no_grad()
def collect(
    agent: Agent,
    env_kwargs: dict,
    num_envs: int,
    dataset: Dataset,
    epsilons: List[float],
    seed: int,
    num_steps: Optional[int] = None,
    num_episodes: Optional[int] = None,
) -> List[dict]:
    assert (num_steps is None) != (num_episodes is None)
    device = agent.device
    env = make_atari_env(num_envs=num_envs, device=device, **env_kwargs)
    rng = np.random.RandomState(seed)
    ac = agent.actor_critic

    obs, _ = env.reset(seed=[seed + i for i in range(num_envs)])
    hx = torch.zeros(num_envs, ac.lstm_dim, device=device)
    cx = torch.zeros(num_envs, ac.lstm_dim, device=device)
    eps = torch.tensor(rng.choice(epsilons, size=num_envs), device=device, dtype=torch.float32)
    buffers = [[] for _ in range(num_envs)]
    manifest = []
    total_steps = 0
    pbar = tqdm(total=num_steps or num_episodes, desc=f"Collect {dataset.name}", unit="steps" if num_steps else "eps")

    def store(i: int, truncate: bool = False) -> None:
        obs_, act_, rew_, end_, trunc_ = (torch.stack(x) for x in zip(*buffers[i]))
        if truncate:
            trunc_[-1] = 1
        episode_id = dataset.add_episode(Episode(obs_, act_, rew_, end_, trunc_, info={}))
        manifest.append(
            {"episode_id": episode_id, "epsilon": eps[i].item(), "length": len(obs_), "return": rew_.sum().item()}
        )
        buffers[i] = []

    while True:
        logits_act, _, (hx, cx) = ac.predict_act_value(obs, (hx, cx))
        act = Categorical(logits=logits_act).sample()
        is_random = torch.rand(num_envs, device=device) < eps
        act = torch.where(is_random, torch.randint_like(act, env.num_actions), act)

        next_obs, rew, end, trunc, _ = env.step(act)
        for i in range(num_envs):
            buffers[i].append((obs[i].cpu(), act[i].cpu(), rew[i].cpu(), end[i].cpu(), trunc[i].cpu()))
        total_steps += num_envs
        if num_steps is not None:
            pbar.update(num_envs)

        dead = torch.logical_or(end, trunc)
        for i in dead.nonzero(as_tuple=True)[0].tolist():
            store(i)
            if num_episodes is not None:
                pbar.update(1)
                if len(manifest) >= num_episodes:
                    pbar.close()
                    env.close()
                    return manifest
            eps[i] = float(rng.choice(epsilons))
        if dead.any():
            hx[dead] = 0
            cx[dead] = 0

        if num_steps is not None and total_steps >= num_steps:
            for i in range(num_envs):
                if len(buffers[i]) > 0:
                    store(i, truncate=True)  # keep the unfinished episodes, marked as truncated
            pbar.close()
            env.close()
            return manifest

        obs = next_obs


def main() -> None:
    args = parse_args()
    assert args.game in ATARI_100K_GAMES, f"{args.game} is not an Atari 100k game (DIAMOND agents exist only for those)"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = dict(
        id=f"{args.game}NoFrameskip-v4",
        done_on_life_loss=False,
        size=args.size,
        max_episode_steps=args.max_episode_steps * 4,  # the gymnasium time limit counts emulator frames
    )
    probe = make_atari_env(num_envs=1, device=device, **env_kwargs)
    num_actions = int(probe.num_actions)
    probe.close()
    agent = load_pretrained_agent(args.game, num_actions, args.size, device, args.agent_ckpt)

    out = args.out_dir.expanduser()
    for split in ("train", "test"):
        if (out / split / "info.pt").is_file():
            raise FileExistsError(f"{out / split} already contains a dataset")

    info = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "num_actions": num_actions}
    for split, kwargs, seed in (
        ("train", dict(num_steps=args.train_steps), args.seed),
        ("test", dict(num_episodes=args.test_episodes), args.seed + 10_000),
    ):
        dataset = Dataset(out / split, f"{split}_dataset")
        manifest = collect(agent, env_kwargs, args.num_envs, dataset, args.epsilons, seed, **kwargs)
        dataset.save_to_default_path()
        lengths = np.array([m["length"] for m in manifest])
        returns = np.array([m["return"] for m in manifest])
        info[split] = {
            "num_episodes": dataset.num_episodes,
            "num_steps": dataset.num_steps,
            "episode_length_mean": float(lengths.mean()),
            "episode_length_min": int(lengths.min()),
            "return_mean": float(returns.mean()),
            "episodes": manifest,
        }
        print(f"{split}: {dataset}  mean length {lengths.mean():.0f}  mean return {returns.mean():.1f}")

    with (out / "collection_info.json").open("w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
