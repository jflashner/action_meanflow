import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from paths import ROOT
from data import Dataset, Episode, SegmentId


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(game):
    source = ROOT / f"datasets/{game}-v1"
    target = ROOT / f"datasets/{game}-native-v2"
    if (target / "complete.json").exists():
        print(f"Verified dataset already exists: {target}", flush=True)
        return
    target.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "data-manifest.json").read_text())
    report = dict(game=game, source_manifest_sha256=digest(source / "data-manifest.json"), splits={})
    for split in ("train", "validation", "test"):
        dataset = Dataset(target / split, save_on_disk=False)
        mapping = []
        for record in manifest["episodes"]:
            if record["split"] != split:
                continue
            path = source / record["path"]
            assert digest(path) == record["sha256"]
            with np.load(path) as raw:
                pixels = torch.from_numpy(raw["obs"].copy()).permute(0, 3, 1, 2).contiguous()
                act = torch.from_numpy(np.append(raw["act"], 0)).long()
                rew = torch.from_numpy(np.append(raw["rew"], np.float32(0))).float()
            size = len(pixels)
            assert len(act) == len(rew) == size == record["steps"] + 1
            episode = Episode(pixels.div(255).mul(2).sub(1), act, rew,
                              torch.zeros(size, dtype=torch.uint8), torch.zeros(size, dtype=torch.uint8), {})
            episode.trunc[-1] = 1
            native_id = dataset.add_episode(episode)
            output = dataset._get_episode_path(native_id)
            output.parent.mkdir(parents=True, exist_ok=True)
            stored = dict(episode.__dict__, obs=pixels)
            torch.save(stored, output)
            loaded = dataset.load_episode(native_id)
            assert torch.equal(loaded.obs, episode.obs) and torch.equal(loaded.act, act)
            padded = dataset[SegmentId(native_id, -2, 4)]
            assert padded.obs.shape[0] == 6 and padded.mask_padding.tolist() == [False, False, True, True, True, True]
            assert torch.equal(padded.obs[2:], loaded.obs[:4])
            assert torch.equal(loaded.act[:-1], act[:-1])
            mapping.append(dict(native_id=native_id, source_id=record["id"], observations=size,
                                source_sha256=record["sha256"], native_sha256=digest(output)))
        dataset.save_to_default_path()
        report["splits"][split] = mapping
        print(json.dumps(dict(game=game, split=split, episodes=dataset.num_episodes, observations=dataset.num_steps)), flush=True)
    report["alignment"] = "All T+1 observations retained; T actions followed by an unused final placeholder. Denoiser uses actions strictly before each target. Reward/termination training is disabled."
    report["pixel_storage"] = "Original uint8 pixels written directly in upstream Episode format, avoiding an extra float-to-uint8 quantization."
    (target / "conversion-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (target / "complete.json").write_text(json.dumps(dict(status="complete", game=game)) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    for game in ("Breakout", "Boxing"):
        prepare(game)
