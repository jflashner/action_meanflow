import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import baseline as base
from paths import ROOT, CONFIG_ROOT
from baseline import DiffusionSampler, DiffusionSamplerConfig
from models.actor_critic import ActorCritic
from utils import extract_state_dict


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()



def prepare(cfg):
    folder = ROOT / "policies"
    folder.mkdir(exist_ok=True)
    metadata = {}
    for filename in ["config/agent/default.yaml", "models/Breakout.pt", "models/Boxing.pt"]:
        target = folder / Path(filename).name
        url = f"https://huggingface.co/{cfg['policy_repository']}/resolve/{cfg['policy_revision']}/atari_100k/{filename}"
        if not target.exists():
            temp = target.with_suffix(".download")
            with urllib.request.urlopen(url, timeout=120) as source, temp.open("wb") as out:
                shutil.copyfileobj(source, out)
            temp.replace(target)
        metadata[target.name] = dict(url=url, sha256=digest(target))
        base.emit("downloaded", filename=target.name)
    base.save_json(folder / "provenance.json", metadata)
    import lpips
    lpips.LPIPS(net="alex").eval()
    base.emit("prepare_complete")



def actor(game, num_actions):
    cfg = OmegaConf.create({"agent": OmegaConf.load(ROOT / "policies/default.yaml"), "env": {"train": {"size": 64}}})
    policy = ActorCritic(instantiate(cfg.agent.actor_critic, num_actions=int(num_actions))).cuda().eval()
    weights = torch.load(ROOT / f"policies/{game}.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(extract_state_dict(weights, "actor_critic"), strict=True)
    del weights
    return policy



@torch.no_grad()
def collect(game, folder, cfg):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "data").mkdir()
    env = base.AtariPreprocessing(base.gym.make(f"{game}NoFrameskip-v4", frameskip=1,
        full_action_space=False, render_mode="rgb_array"), 30, 4, 64)
    policy = actor(game, env.action_space.n)
    records = []
    started = time.monotonic()
    game_offset = 0 if game == "Breakout" else 1000000
    try:
        for split_id, split in enumerate(("train", "validation", "test")):
            steps, episode = 0, 0
            target = cfg["train_transition_target"] if split == "train" else 0
            minimum = cfg["minimum_train_episodes"] if split == "train" else cfg[f"{split}_episodes"]
            while episode < minimum or steps < target:
                seed = 11030000 + game_offset + split_id * 100000 + episode
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                epsilon = cfg["exploration_epsilons"][episode % len(cfg["exploration_epsilons"])]
                obs, info = env.reset(seed=seed)
                hx = torch.zeros(1, policy.lstm_dim, device="cuda")
                cx = torch.zeros_like(hx)
                frames, actions, rewards = [obs.copy()], [], []
                fire = game == "Breakout"
                for t in range(cfg["episode_cap"]):
                    tensor = torch.as_tensor(obs, device="cuda").float().div(127.5).sub(1).permute(2, 0, 1)[None]
                    logits, _, (hx, cx) = policy.predict_act_value(tensor, (hx, cx))
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
                    if rng.random() < epsilon:
                        action = int(rng.integers(int(env.action_space.n)))
                    if fire:
                        action = env.unwrapped.get_action_meanings().index("FIRE")
                    obs, reward, end, trunc, info = env.step(action)
                    if end or trunc:
                        break
                    frames.append(obs.copy())
                    actions.append(action)
                    rewards.append(float(reward))
                    fire = game == "Breakout" and bool(info["life_loss"])
                    if info["life_loss"]:
                        hx.zero_()
                        cx.zero_()
                if len(actions) < 4:
                    raise RuntimeError("Episode too short for a transition window")
                eid = split_id * 100000 + episode
                file = folder / "data" / f"episode-{eid:06d}.npz"
                np.savez_compressed(file, obs=np.stack(frames), act=np.asarray(actions, dtype=np.int64),
                                    rew=np.asarray(rewards, dtype=np.float32))
                records.append(dict(id=eid, seed=seed, split=split, steps=len(actions), epsilon=epsilon,
                    return_excluding_terminal=sum(rewards), ended=bool(end),
                    truncated=bool(trunc or t == cfg["episode_cap"] - 1),
                    path=str(file.relative_to(folder)), sha256=digest(file)))
                steps += len(actions)
                episode += 1
                if episode % 8 == 0:
                    base.emit("collection", game=game, split=split, episodes=episode, steps=steps)
            base.emit("split_complete", game=game, split=split, episodes=episode, steps=steps)
        manifest = dict(episodes=records, actions=env.unwrapped.get_action_meanings(), game=game,
            policy=json.loads((ROOT / "policies/provenance.json").read_text())[f"{game}.pt"],
            policy_revision=cfg["policy_revision"], collection_seconds=time.monotonic() - started,
            alignment="obs[t], act[t] -> obs[t+1]; episode split before windowing", protocol=cfg)
        base.save_json(folder / "data-manifest.json", manifest)
        base.save_json(folder / "complete.json", dict(status="complete", kind="dataset"))
    finally:
        env.close()



@torch.no_grad()
def predict(sampler, history, actions, chunk=16):
    return torch.cat([sampler.sample(history[i:i+chunk], actions[i:i+chunk])[0]
                      for i in range(0, len(history), chunk)])



def mean_metrics(values):
    return base.aggregate(values)



@torch.no_grad()
def metrics(pred, target, previous, perceptual):
    out = base.scores(pred, target, previous)
    out["lpips_alex"] = torch.cat([perceptual(pred[i:i+16], target[i:i+16]).flatten()
        for i in range(0, len(pred), 16)]).cpu().tolist()
    return out



@torch.no_grad()
def evaluate(model, cfg, run, records):
    import lpips
    perceptual = lpips.LPIPS(net="alex").cuda().eval()
    model.eval()
    ec = cfg["evaluation"]
    result = {}
    for split in ("validation", "test"):
        episodes = base.load_episodes(run, records, split)
        obs, act, ids = base.evaluation_windows(episodes, ec["next_frame_windows_per_episode"], 1, 301)
        raw = {"repeat_frame": metrics(obs[:, 3], obs[:, 4], obs[:, 3], perceptual)}
        summary = dict(window_ids=ids, repeat_frame=mean_metrics(raw["repeat_frame"]), sampling={})
        random_actions = act[:, :4].clone()
        generator = torch.Generator(device="cuda").manual_seed(404)
        random_actions[:, -1] = torch.randint(int(model.cfg.inner_model.num_actions), (len(act),), generator=generator, device="cuda")
        for steps in ec["sampling_steps"]:
            sampler = DiffusionSampler(model, DiffusionSamplerConfig(num_steps_denoising=steps))
            torch.manual_seed(505)
            correct = predict(sampler, obs[:, :4], act[:, :4])
            values = metrics(correct, obs[:, 4], obs[:, 3], perceptual)
            torch.manual_seed(505)
            random_pred = predict(sampler, obs[:, :4], random_actions)
            random_scores = base.scores(random_pred, obs[:, 4], obs[:, 3])
            raw[str(steps)] = values
            delta = (np.asarray(values["psnr_db"]) - np.asarray(random_scores["psnr_db"])).tolist()
            summary["sampling"][str(steps)] = dict(**mean_metrics(values), action_delta_psnr_db=float(np.mean(delta)),
                random_action_mse=float(np.mean(random_scores["mse"])))
            raw[f"action_delta_psnr_{steps}"] = delta
        summary["random_action_fraction_changed"] = float((act[:, 3] != random_actions[:, 3]).float().mean())
        long_obs, long_act, long_ids = base.evaluation_windows(episodes, ec["rollout_windows_per_episode"], ec["rollout_horizon"], 606)
        summary.update(rollout_window_ids=long_ids, rollout_eligible_episodes=len({p[0] for p in long_ids}),
                       total_episodes=len(episodes), rollout={})
        baseline_curve = [base.scores(long_obs[:, 3], long_obs[:, t+4], long_obs[:, t+3])
                          for t in range(ec["rollout_horizon"])]
        raw["rollout_repeat"] = baseline_curve
        for steps in ec["sampling_steps"]:
            sampler = DiffusionSampler(model, DiffusionSamplerConfig(num_steps_denoising=steps))
            context = long_obs[:, :4].clone()
            curve, selected_frames = [], []
            torch.manual_seed(707)
            for t in range(ec["rollout_horizon"]):
                pred = predict(sampler, context, long_act[:, t:t+4])
                values = base.scores(pred, long_obs[:, t+4], long_obs[:, t+3])
                if t+1 in ec["rollout_report_steps"]:
                    values["lpips_alex"] = metrics(pred, long_obs[:, t+4], long_obs[:, t+3], perceptual)["lpips_alex"]
                curve.append(values)
                selected_frames.append(pred[:2].clone())
                context = torch.cat([context[:, 1:], pred[:, None]], dim=1)
            generated = torch.stack(selected_frames, dim=1)
            for i in range(min(2, len(long_ids))):
                base.gif_comparison(run / f"{split}-steps{steps}-rollout{i}.gif", long_obs[i, 4:], generated[i], long_obs[i, 3])
            np.savez_compressed(run / f"{split}-steps{steps}-examples.npz", ids=np.asarray(long_ids[:2]),
                actions=long_act[:2].cpu().numpy(), truth=long_obs[:2].add(1).mul(127.5).clamp(0,255).byte().cpu().numpy(),
                generated=generated.add(1).mul(127.5).clamp(0,255).byte().cpu().numpy())
            summary["rollout"][str(steps)] = [mean_metrics(v) for v in curve]
            raw[f"rollout_{steps}"] = curve
        summary["rollout_repeat"] = [mean_metrics(v) for v in baseline_curve]
        base.save_json(run / f"{split}-per-window.json", raw)
        result[split] = summary
        base.emit("evaluation_split_complete", split=split, metrics=summary["sampling"], eligible=summary["rollout_eligible_episodes"])
        latency_obs, latency_act = obs[:1, :4].cpu(), act[:1, :4].cpu()
        del obs, act, long_obs, long_act, context, generated, selected_frames, pred, correct, random_pred, sampler
        del episodes
        gc.collect()
    del perceptual
    gc.collect()
    torch.cuda.empty_cache()
    result["latency"] = {}
    for steps in ec["sampling_steps"]:
        sampler = DiffusionSampler(model, DiffusionSamplerConfig(num_steps_denoising=steps))
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(ec["latency_warmup"] + ec["latency_repeats"]):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            output, _ = sampler.sample(latency_obs.cuda(), latency_act.cuda())
            output.add(1).mul(127.5).clamp(0,255).byte().cpu()
            torch.cuda.synchronize()
            if i >= ec["latency_warmup"]:
                timings.append(1000*(time.perf_counter()-tick))
        result["latency"][str(steps)] = dict(median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings,95)),
            measurements_ms=timings, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    result["latency_scope"] = "CPU normalized history/actions through GPU generation to CPU uint8 output; batch 1, uncompiled FP32; excludes environment, display, network and LPIPS"
    base.save_json(run / "evaluation.json", result)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "collect"])
    parser.add_argument("--game", choices=["Breakout", "Boxing"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((CONFIG_ROOT / "protocol.json").read_text())
    if args.stage == "prepare":
        prepare(cfg)
    else:
        if args.game is None:
            parser.error("collect requires --game")
        collect(args.game, ROOT / f"datasets/{args.game}-v1", cfg)


if __name__ == "__main__":
    main()
