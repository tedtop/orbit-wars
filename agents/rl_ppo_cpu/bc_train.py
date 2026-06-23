"""
Behavior cloning for Orbit Wars — per-planet ActorCritic architecture.

Trains the same ActorCritic used by PPO (train.py) via supervised learning
on prize-zone replay data extracted by pipeline/extract_moves.py.

Usage:
    python agents/rl_ppo_cpu/bc_train.py \
        --data training/moves_v8.jsonl.gz \
        --epochs 10 --lr 1e-3 --batch-size 512 \
        --out agents/rl_ppo_cpu/checkpoints/bc_best.pt
"""

import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── Mirror constants from train.py (keep in sync) ─────────────────────────────
MAX_PLANETS  = 20
MAX_FLEETS   = 12
PLANET_FEATS = 10
FLEET_FEATS  = 8
GLOBAL_FEATS = 6
OBS_DIM      = MAX_PLANETS * PLANET_FEATS + MAX_FLEETS * FLEET_FEATS + GLOBAL_FEATS
FRAC_BINS    = [0.25, 0.5, 0.75, 1.0]
N_FRACS      = len(FRAC_BINS)
MIN_SHIPS    = 3.0
HIDDEN       = 256
N_LAYERS     = 3
ANGLE_TOL    = 0.5   # radians — tolerance for matching launch angle to target planet


# ── Observation encoder (verbatim from train.py) ──────────────────────────────

def encode_obs(obs: dict, player: int) -> np.ndarray:
    planets = obs.get("planets", [])
    fleets  = obs.get("fleets", [])
    step    = int(obs.get("step", 0))

    pf = np.zeros((MAX_PLANETS, PLANET_FEATS), dtype=np.float32)
    for i, p in enumerate(planets[:MAX_PLANETS]):
        pid, owner, x, y, r, ships, prod = p[:7]
        dx, dy = x - 50.0, y - 50.0
        dist_sun = math.sqrt(dx*dx + dy*dy) / 70.0
        angle    = math.atan2(dy, dx) / math.pi
        pf[i] = [
            x / 100.0, y / 100.0, r / 10.0,
            ships / 200.0, prod / 20.0,
            1.0 if owner == player else 0.0,
            1.0 if (owner >= 0 and owner != player) else 0.0,
            1.0 if owner == -1 else 0.0,
            dist_sun, angle,
        ]

    ff = np.zeros((MAX_FLEETS, FLEET_FEATS), dtype=np.float32)
    for i, f in enumerate(fleets[:MAX_FLEETS]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        spd = (1.0 + 5.0 * (max(ships, 1) / 1000.0) ** 1.5) if ships > 0 else 0.0
        dx, dy = x - 50.0, y - 50.0
        dist_sun = math.sqrt(dx*dx + dy*dy) / 70.0
        ff[i] = [
            x / 100.0, y / 100.0, ships / 200.0, angle / math.pi,
            1.0 if owner == player else 0.0,
            1.0 if (owner >= 0 and owner != player) else 0.0,
            dist_sun, spd / 6.0,
        ]

    my_s = sum(p[5] for p in planets if p[1] == player) + \
           sum(f[6] for f in fleets  if f[1] == player)
    en_s = sum(p[5] for p in planets if p[1] >= 0 and p[1] != player) + \
           sum(f[6] for f in fleets  if f[1] >= 0 and f[1] != player)
    my_p = sum(1 for p in planets if p[1] == player)
    en_p = sum(1 for p in planets if p[1] >= 0 and p[1] != player)

    gf = np.array([
        step / 500.0,
        my_s / 1000.0, en_s / 1000.0,
        my_p / 20.0,   en_p / 20.0,
        (my_s - en_s) / 1000.0,
    ], dtype=np.float32)

    return np.concatenate([pf.flatten(), ff.flatten(), gf])


# ── ActorCritic (verbatim from train.py) ──────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        d = OBS_DIM
        for _ in range(N_LAYERS):
            layers += [nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN), nn.Tanh()]
            d = HIDDEN
        self.backbone = nn.Sequential(*layers)

        src_input_dim  = HIDDEN + PLANET_FEATS
        self.fire_head = nn.Linear(src_input_dim, 1)
        self.tgt_head  = nn.Linear(src_input_dim, MAX_PLANETS)
        self.frac_head = nn.Linear(src_input_dim, N_FRACS)
        self.val_head  = nn.Linear(HIDDEN, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
        for h in (self.fire_head, self.tgt_head, self.frac_head):
            nn.init.orthogonal_(h.weight, gain=0.01)
        nn.init.orthogonal_(self.val_head.weight, gain=1.0)

    def forward(self, obs_t, planet_feats_t, tgt_mask=None):
        h = self.backbone(obs_t)
        x = torch.cat([h, planet_feats_t], dim=-1)
        fire_logits = self.fire_head(x).squeeze(-1)
        tgt_logits  = self.tgt_head(x)
        frac_logits = self.frac_head(x)
        if tgt_mask is not None:
            tgt_logits = tgt_logits.masked_fill(~tgt_mask, -1e8)
        return fire_logits, tgt_logits, frac_logits


# ── Action parsing ─────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def parse_step(obs: dict, raw_action, player: int):
    """
    Return list of (obs_vec, planet_feats, fire_label, tgt_label, frac_label, has_launch).
    has_launch=False for fire=0 examples — target/frac labels are garbage, don't train on them.
    """
    planets = obs.get("planets", [])
    obs_vec = encode_obs(obs, player)

    # Extract pf slice for each planet (rows of the pf block in obs_vec)
    pf_block = obs_vec[:MAX_PLANETS * PLANET_FEATS].reshape(MAX_PLANETS, PLANET_FEATS)

    # Map planet_id -> list index and coordinates
    id_to_idx = {}
    id_to_xy  = {}
    for idx, p in enumerate(planets[:MAX_PLANETS]):
        pid = int(p[0])
        id_to_idx[pid] = idx
        id_to_xy[pid]  = (float(p[2]), float(p[3]))

    # Build launched_from: planet_id -> (angle, ships, ships_on_planet)
    launched_from = {}
    for mv in (raw_action or []):
        if not mv or len(mv) < 3:
            continue
        src_id, angle, ships = int(mv[0]), float(mv[1]), float(mv[2])
        launched_from[src_id] = (angle, ships)

    examples = []
    for idx, p in enumerate(planets[:MAX_PLANETS]):
        pid   = int(p[0])
        owner = int(p[1])
        if owner != player:
            continue
        p_ships = float(p[5])
        if p_ships < MIN_SHIPS:
            continue

        planet_feats = pf_block[idx]
        fire_label   = 1 if pid in launched_from else 0

        tgt_label  = 0
        frac_label = 0
        has_launch = False

        if fire_label == 1:
            angle, ships = launched_from[pid]
            src_x, src_y = id_to_xy[pid]

            # Find target planet by angle proximity
            best_tgt  = 0
            best_diff = math.pi + 1.0
            for jdx, q in enumerate(planets[:MAX_PLANETS]):
                qid = int(q[0])
                if qid == pid:
                    continue
                dx = float(q[2]) - src_x
                dy = float(q[3]) - src_y
                expected = math.atan2(dy, dx)
                diff = _angle_diff(angle, expected)
                if diff < best_diff:
                    best_diff = diff
                    best_tgt  = jdx

            if best_diff > ANGLE_TOL:
                # Couldn't match angle to any planet — skip this launch
                fire_label = 0
            else:
                tgt_label  = best_tgt
                frac       = min(ships / max(p_ships, 1.0), 1.0)
                frac_label = min(range(N_FRACS),
                                 key=lambda k: abs(FRAC_BINS[k] - frac))
                has_launch = True

        examples.append((obs_vec, planet_feats, fire_label, tgt_label, frac_label, has_launch))

    return examples


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(path: str, min_rating: float = 1400.0):
    examples = []
    n_records = n_skipped = n_steps = 0
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            rt  = rec.get("rating")
            if rt is None or rt < min_rating:
                n_skipped += 1
                continue
            obs_raw = rec.get("obs")
            if obs_raw is None:
                n_skipped += 1
                continue
            player = int(rec.get("player", 0))
            action = rec.get("action", [])
            n_records += 1

            step_examples = parse_step(obs_raw, action, player)
            examples.extend(step_examples)
            n_steps += len(step_examples)

    print(f"Loaded {n_records} records → {len(examples)} per-planet examples "
          f"({n_skipped} skipped, {sum(1 for e in examples if e[5])} launches)")
    return examples


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.data} (min_rating={args.min_rating}) …")
    examples = load_dataset(args.data, min_rating=args.min_rating)
    if not examples:
        print("ERROR: no examples loaded — check --data path and --min-rating")
        sys.exit(1)

    n_launch = sum(1 for e in examples if e[5])
    print(f"Total examples: {len(examples)} | with launch: {n_launch} ({100*n_launch/len(examples):.1f}%)")

    # Train/val split (last 10%)
    split = int(0.9 * len(examples))
    import random
    random.shuffle(examples)
    train_ex = examples[:split]
    val_ex   = examples[split:]

    def to_tensors(ex_list):
        obs_t    = torch.tensor(np.array([e[0] for e in ex_list]), dtype=torch.float32)
        pf_t     = torch.tensor(np.array([e[1] for e in ex_list]), dtype=torch.float32)
        fire_t   = torch.tensor([e[2] for e in ex_list], dtype=torch.float32)
        tgt_t    = torch.tensor([e[3] for e in ex_list], dtype=torch.long)
        frac_t   = torch.tensor([e[4] for e in ex_list], dtype=torch.long)
        launch_t = torch.tensor([e[5] for e in ex_list], dtype=torch.bool)
        return obs_t, pf_t, fire_t, tgt_t, frac_t, launch_t

    model = ActorCritic().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_ex)
        total_loss = fire_acc = tgt_acc = frac_acc = 0.0
        n_batches = n_launch_train = 0

        for i in range(0, len(train_ex), args.batch_size):
            batch = train_ex[i : i + args.batch_size]
            obs_t, pf_t, fire_t, tgt_t, frac_t, launch_t = to_tensors(batch)
            obs_t    = obs_t.to(device)
            pf_t     = pf_t.to(device)
            fire_t   = fire_t.to(device)
            tgt_t    = tgt_t.to(device)
            frac_t   = frac_t.to(device)
            launch_t = launch_t.to(device)

            fire_logits, tgt_logits, frac_logits = model(obs_t, pf_t)

            loss_fire = F.binary_cross_entropy_with_logits(fire_logits, fire_t)

            if launch_t.any():
                loss_tgt  = F.cross_entropy(tgt_logits[launch_t],  tgt_t[launch_t])
                loss_frac = F.cross_entropy(frac_logits[launch_t], frac_t[launch_t])
            else:
                loss_tgt  = torch.tensor(0.0, device=device)
                loss_frac = torch.tensor(0.0, device=device)

            loss = loss_fire + loss_tgt + loss_frac

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            fire_acc   += ((fire_logits > 0).float() == fire_t).float().mean().item()
            if launch_t.any():
                tgt_acc  += (tgt_logits[launch_t].argmax(-1)  == tgt_t[launch_t]).float().mean().item()
                frac_acc += (frac_logits[launch_t].argmax(-1) == frac_t[launch_t]).float().mean().item()
                n_launch_train += 1
            n_batches += 1

        scheduler.step()
        avg_loss  = total_loss / n_batches
        avg_fire  = fire_acc   / n_batches
        avg_tgt   = tgt_acc    / max(n_launch_train, 1)
        avg_frac  = frac_acc   / max(n_launch_train, 1)

        # Validation
        model.eval()
        val_loss = val_fire = val_tgt = val_frac = 0.0
        val_batches = val_launch = 0
        with torch.no_grad():
            for i in range(0, len(val_ex), args.batch_size):
                batch = val_ex[i : i + args.batch_size]
                obs_t, pf_t, fire_t, tgt_t, frac_t, launch_t = to_tensors(batch)
                obs_t    = obs_t.to(device)
                pf_t     = pf_t.to(device)
                fire_t   = fire_t.to(device)
                tgt_t    = tgt_t.to(device)
                frac_t   = frac_t.to(device)
                launch_t = launch_t.to(device)

                fire_logits, tgt_logits, frac_logits = model(obs_t, pf_t)
                l_fire = F.binary_cross_entropy_with_logits(fire_logits, fire_t)
                if launch_t.any():
                    l_tgt  = F.cross_entropy(tgt_logits[launch_t],  tgt_t[launch_t])
                    l_frac = F.cross_entropy(frac_logits[launch_t], frac_t[launch_t])
                    val_tgt  += (tgt_logits[launch_t].argmax(-1)  == tgt_t[launch_t]).float().mean().item()
                    val_frac += (frac_logits[launch_t].argmax(-1) == frac_t[launch_t]).float().mean().item()
                    val_launch += 1
                else:
                    l_tgt = l_frac = torch.tensor(0.0)
                val_loss    += (l_fire + l_tgt + l_frac).item()
                val_fire    += ((fire_logits > 0).float() == fire_t).float().mean().item()
                val_batches += 1

        vl   = val_loss    / val_batches
        vf   = val_fire    / val_batches
        vtgt = val_tgt     / max(val_launch, 1)
        vfrc = val_frac    / max(val_launch, 1)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"loss={avg_loss:.4f} fire={avg_fire:.3f} tgt={avg_tgt:.3f} frac={avg_frac:.3f} | "
              f"val: loss={vl:.4f} fire={vf:.3f} tgt={vtgt:.3f} frac={vfrc:.3f}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({
                "state": model.state_dict(),   # key matches eval_checkpoints.py
                "epoch": epoch,
                "val_loss": vl,
                "val_fire_acc": vf,
                "val_tgt_acc": vtgt,
                "val_frac_acc": vfrc,
                "args": vars(args),
            }, args.out)
            print(f"  -> saved best checkpoint (val_loss={vl:.4f})")

    print(f"\nDone. Best val_loss={best_val_loss:.4f} | checkpoint: {args.out}")
    print(f"Next: python agents/rl_ppo_cpu/eval_checkpoints.py --checkpoint-a {args.out} --vs-comet --n-games 200")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       default="training/moves_v8.jsonl.gz")
    ap.add_argument("--min-rating", type=float, default=1400.0)
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=512)
    ap.add_argument("--out",        default="agents/rl_ppo_cpu/checkpoints/bc_best.pt")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
