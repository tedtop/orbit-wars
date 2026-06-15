#!/usr/bin/env python3
"""PlanetPolicy — factored per-source actor-critic for Orbit Wars RL.

Consumes the feature tensors from rl/features.py and produces, per owned (source)
planet: a distribution over {no-op, K candidate targets} and over ship-count
buckets, plus a state value (PPO critic). Extends the RL tutorial's PlanetPolicy
(which had only a target head) with a no-op option and a ship-bucket head.

Aim is NOT predicted — at action time the chosen (source, target) is converted to
an angle by orbit_lite's lead-aim solver. The policy only decides which target and
how many ships.

Check:  python rl/policy.py   (runs a forward pass on a real replay obs)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ship-count buckets as fractions of the source's safe garrison launch.
SHIP_BUCKETS = (0.25, 0.5, 0.75, 1.0)


@dataclass
class PolicyOutput:
    target_logits: torch.Tensor   # [S, K+1]  index 0 = no-op, 1..K = candidates
    ship_logits: torch.Tensor     # [S, B]
    value: torch.Tensor           # scalar


def _mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, o)]
    return nn.Sequential(*layers)


class PlanetPolicy(nn.Module):
    def __init__(self, self_dim=8, candidate_dim=12, global_dim=8,
                 candidate_count=8, n_ship_buckets=len(SHIP_BUCKETS), hidden=128):
        super().__init__()
        self.K = candidate_count
        self.B = n_ship_buckets
        self.self_encoder = _mlp(self_dim, hidden, hidden)
        self.global_encoder = _mlp(global_dim, hidden, hidden)
        self.candidate_encoder = _mlp(candidate_dim, hidden, hidden)
        self.target_head = _mlp(hidden * 3, hidden, 1)          # per (source, candidate)
        self.noop_head = _mlp(hidden * 2, hidden, 1)            # per source (self+global)
        self.ship_head = _mlp(hidden * 3, hidden, n_ship_buckets)  # per source
        self.value_head = _mlp(hidden * 3, hidden, 1)

    def forward(self, self_feat, cand_feat, global_feat, cand_mask) -> PolicyOutput:
        # self_feat [S,Fs], cand_feat [S,K,Fc], global_feat [Fg], cand_mask [S,K] bool
        S = self_feat.shape[0]
        sh = self.self_encoder(self_feat)                       # [S,H]
        gh = self.global_encoder(global_feat.unsqueeze(0))      # [1,H]
        ch = self.candidate_encoder(cand_feat)                  # [S,K,H]
        gh_s = gh.expand(S, -1)                                 # [S,H]

        exp_self = sh.unsqueeze(1).expand(-1, self.K, -1)       # [S,K,H]
        exp_glob = gh_s.unsqueeze(1).expand(-1, self.K, -1)     # [S,K,H]
        joint = torch.cat([exp_self, exp_glob, ch], dim=-1)     # [S,K,3H]
        cand_logits = self.target_head(joint).squeeze(-1)       # [S,K]
        cand_logits = cand_logits.masked_fill(~cand_mask, torch.finfo(cand_logits.dtype).min)

        noop_logit = self.noop_head(torch.cat([sh, gh_s], dim=-1))  # [S,1]
        target_logits = torch.cat([noop_logit, cand_logits], dim=-1)  # [S,K+1]

        pooled = ch.mean(dim=1)                                 # [S,H]
        sgp = torch.cat([sh, gh_s, pooled], dim=-1)             # [S,3H]
        ship_logits = self.ship_head(sgp)                       # [S,B]
        value = self.value_head(sgp).mean()                     # scalar (mean over sources)
        return PolicyOutput(target_logits=target_logits, ship_logits=ship_logits, value=value)

    @torch.no_grad()
    def act(self, self_feat, cand_feat, global_feat, cand_mask, deterministic=False):
        """Sample a per-source action. Returns dict of tensors for env decode + PPO."""
        out = self.forward(self_feat, cand_feat, global_feat, cand_mask)
        tdist = torch.distributions.Categorical(logits=out.target_logits)
        sdist = torch.distributions.Categorical(logits=out.ship_logits)
        if deterministic:
            tgt = out.target_logits.argmax(-1)
            ship = out.ship_logits.argmax(-1)
        else:
            tgt = tdist.sample()
            ship = sdist.sample()
        return {
            "target": tgt, "ship": ship,
            "logp": tdist.log_prob(tgt) + sdist.log_prob(ship),  # [S]
            "value": out.value, "entropy": tdist.entropy().mean() + sdist.entropy().mean(),
        }

    def evaluate(self, self_feat, cand_feat, global_feat, cand_mask, tgt, ship):
        """Re-evaluate stored actions for the PPO update."""
        out = self.forward(self_feat, cand_feat, global_feat, cand_mask)
        tdist = torch.distributions.Categorical(logits=out.target_logits)
        sdist = torch.distributions.Categorical(logits=out.ship_logits)
        logp = tdist.log_prob(tgt) + sdist.log_prob(ship)
        ent = tdist.entropy().mean() + sdist.entropy().mean()
        return logp, out.value, ent


# ---------------------------------------------------------------------------
def _check():
    import json
    from pathlib import Path
    import numpy as np
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import build_features

    rp = next(iter(Path("replays").rglob("*.json")), None)
    if rp is None:
        print("no replay to test on"); return
    d = json.loads(rp.read_text())
    n = len(d["info"]["TeamNames"])
    obs = d["steps"][min(60, len(d["steps"]) - 1)][0]["observation"]
    f = build_features(obs, 0, n)
    t = lambda a: torch.tensor(a)
    pol = PlanetPolicy()
    out = pol.forward(t(f["self"]), t(f["cand"]), t(f["global"]), t(f["cand_mask"]))
    S = f["self"].shape[0]
    print(f"sources={S} | target_logits {tuple(out.target_logits.shape)} "
          f"ship_logits {tuple(out.ship_logits.shape)} value={out.value.item():.3f}")
    a = pol.act(t(f["self"]), t(f["cand"]), t(f["global"]), t(f["cand_mask"]))
    print(f"sampled targets (0=no-op): {a['target'].tolist()}  ship-buckets: {a['ship'].tolist()}")
    print(f"logp per source: {[round(x,2) for x in a['logp'].tolist()]}  entropy={a['entropy']:.3f}")
    print(f"params: {sum(p.numel() for p in pol.parameters()):,}")


if __name__ == "__main__":
    _check()
