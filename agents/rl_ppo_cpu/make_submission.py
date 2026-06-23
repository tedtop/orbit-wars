"""
Package a trained checkpoint into a Kaggle submission tar.gz.
Inference uses pure numpy — no torch import at submission time.
This avoids the NumPy 2.x/PyTorch binary incompatibility on the runner.
"""
import math, io, os, sys, base64, tarfile, json
import numpy as np
import torch

# ── Must match train.py constants ─────────────────────────────────────────────
MAX_PLANETS  = 20
MAX_FLEETS   = 12
PLANET_FEATS = 10
FLEET_FEATS  = 8
GLOBAL_FEATS = 6
OBS_DIM      = MAX_PLANETS * PLANET_FEATS + MAX_FLEETS * FLEET_FEATS + GLOBAL_FEATS
N_FRACS      = 4
FRAC_BINS    = [0.25, 0.5, 0.75, 1.0]
MIN_SHIPS    = 3.0
HIDDEN       = 256
N_LAYERS     = 3


def export_weights_numpy(checkpoint_path: str) -> dict:
    """Load PyTorch checkpoint and export all weights as numpy arrays."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ck["state"] if "state" in ck else ck
    return {k: v.numpy() for k, v in sd.items()}


def make_submission(checkpoint_path: str = "best_model.pt",
                    output_path: str = "submission.tar.gz"):
    weights = export_weights_numpy(checkpoint_path)
    # Serialize as npz
    buf = io.BytesIO()
    np.savez_compressed(buf, **weights)
    buf.seek(0)
    weights_b64 = base64.b64encode(buf.read()).decode()

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    info = {"update": ck.get("update", "?"), "steps": ck.get("steps", "?")}
    print(f"Exporting checkpoint: update={info['update']} steps={info['steps']:,}")

    agent_code = f'''"""Orbit Wars RL agent — pure numpy inference."""
import math, base64, io, sys
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path("/kaggle_simulations/agent")
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np

# ── Constants (must match training) ──────────────────────────────────────────
MAX_PLANETS  = {MAX_PLANETS}
MAX_FLEETS   = {MAX_FLEETS}
PLANET_FEATS = {PLANET_FEATS}
FLEET_FEATS  = {FLEET_FEATS}
GLOBAL_FEATS = {GLOBAL_FEATS}
OBS_DIM      = {OBS_DIM}
N_FRACS      = {N_FRACS}
FRAC_BINS    = {FRAC_BINS}
MIN_SHIPS    = {MIN_SHIPS}
HIDDEN       = {HIDDEN}
N_LAYERS     = {N_LAYERS}

# ── Load weights ──────────────────────────────────────────────────────────────
_WEIGHTS_B64 = "{weights_b64}"

def _load_weights():
    raw = base64.b64decode(_WEIGHTS_B64)
    buf = io.BytesIO(raw)
    return dict(np.load(buf))

_W = _load_weights()

# ── Numpy MLP ─────────────────────────────────────────────────────────────────

def _tanh(x):
    return np.tanh(x)

def _layer_norm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return w * (x - m) / np.sqrt(v + eps) + b

def _linear(x, w, b):
    return x @ w.T + b

def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _backbone(x):
    """Run shared backbone: N_LAYERS x (Linear + LayerNorm + Tanh)."""
    for i in range(N_LAYERS):
        w  = _W[f"backbone.{{i*3}}.weight"]
        b  = _W[f"backbone.{{i*3}}.bias"]
        ln_w = _W[f"backbone.{{i*3+1}}.weight"]
        ln_b = _W[f"backbone.{{i*3+1}}.bias"]
        x = _linear(x, w, b)
        x = _layer_norm(x, ln_w, ln_b)
        x = _tanh(x)
    return x

def _forward(obs, pf, tgt_mask):
    """
    obs:      (OBS_DIM,)
    pf:       (PLANET_FEATS,)
    tgt_mask: (MAX_PLANETS,) bool
    Returns: fire_prob, tgt_probs, frac_probs
    """
    h = _backbone(obs)                    # (HIDDEN,)
    x = np.concatenate([h, pf])          # (HIDDEN + PLANET_FEATS,)

    fire_w = _W["fire_head.weight"]; fire_b = _W["fire_head.bias"]
    tgt_w  = _W["tgt_head.weight"];  tgt_b  = _W["tgt_head.bias"]
    frac_w = _W["frac_head.weight"]; frac_b = _W["frac_head.bias"]

    fire_logit = (_linear(x, fire_w, fire_b))[0]
    fire_prob  = _sigmoid(fire_logit)

    tgt_logits = _linear(x, tgt_w, tgt_b)   # (MAX_PLANETS,)
    tgt_logits[~tgt_mask] = -1e8
    tgt_probs = _softmax(tgt_logits)

    frac_logits = _linear(x, frac_w, frac_b)
    frac_probs  = _softmax(frac_logits)

    return fire_prob, tgt_probs, frac_probs


# ── Obs encoder ───────────────────────────────────────────────────────────────

def _encode_obs(obs):
    player  = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    fleets  = obs.get("fleets", [])
    step    = int(obs.get("step", 0))

    pf = np.zeros((MAX_PLANETS, PLANET_FEATS), dtype=np.float32)
    for i, p in enumerate(planets[:MAX_PLANETS]):
        pid, owner, x, y, r, ships, prod = p[:7]
        dx, dy = x - 50.0, y - 50.0
        dist_sun = (dx*dx + dy*dy)**0.5 / 70.0
        angle    = math.atan2(dy, dx) / math.pi
        pf[i] = [x/100, y/100, r/10, ships/200, prod/20,
                 1. if owner==player else 0.,
                 1. if (owner>=0 and owner!=player) else 0.,
                 1. if owner==-1 else 0.,
                 dist_sun, angle]

    ff = np.zeros((MAX_FLEETS, FLEET_FEATS), dtype=np.float32)
    for i, f in enumerate(fleets[:MAX_FLEETS]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        spd = (1. + 5.*(max(ships,1)/1000.)**1.5) if ships > 0 else 0.
        dx, dy = x-50., y-50.
        dist_sun = (dx*dx+dy*dy)**0.5 / 70.
        ff[i] = [x/100, y/100, ships/200, angle/math.pi,
                 1. if owner==player else 0.,
                 1. if (owner>=0 and owner!=player) else 0.,
                 dist_sun, spd/6.]

    my_s = sum(p[5] for p in planets if p[1]==player) + \
           sum(f[6] for f in fleets  if f[1]==player)
    en_s = sum(p[5] for p in planets if p[1]>=0 and p[1]!=player) + \
           sum(f[6] for f in fleets  if f[1]>=0 and f[1]!=player)
    my_p = sum(1 for p in planets if p[1]==player)
    en_p = sum(1 for p in planets if p[1]>=0 and p[1]!=player)

    gf = np.array([step/500., my_s/1000., en_s/1000.,
                   my_p/20., en_p/20., (my_s-en_s)/1000.], dtype=np.float32)
    return np.concatenate([pf.flatten(), ff.flatten(), gf])


# ── Agent entry point ─────────────────────────────────────────────────────────

def agent(obs):
    d = obs if isinstance(obs, dict) else {{
        "player": obs.player, "planets": obs.planets,
        "fleets": obs.fleets, "step": getattr(obs, "step", 0)}}
    player  = int(d.get("player", 0))
    planets = d.get("planets", [])

    obs_vec = _encode_obs(d)
    launches = []

    for src_idx, p in enumerate(planets[:MAX_PLANETS]):
        if p[1] != player or p[5] < MIN_SHIPS:
            continue

        pid, owner, x, y, r, ships, prod = p[:7]
        dx, dy = x-50., y-50.
        pf = np.array([x/100, y/100, r/10, ships/200, prod/20,
                       1., 0., 0.,
                       (dx*dx+dy*dy)**0.5/70.,
                       math.atan2(dy,dx)/math.pi], dtype=np.float32)

        tgt_mask = np.ones(MAX_PLANETS, dtype=bool)
        tgt_mask[src_idx] = False
        for j in range(len(planets), MAX_PLANETS):
            tgt_mask[j] = False

        fire_prob, tgt_probs, frac_probs = _forward(obs_vec, pf, tgt_mask)

        if fire_prob < 0.5:
            continue

        tgt_idx  = int(np.argmax(tgt_probs))
        frac_idx = int(np.argmax(frac_probs))

        if tgt_idx >= len(planets) or tgt_idx == src_idx:
            continue

        tgt = planets[tgt_idx]
        angle   = math.atan2(tgt[3]-p[3], tgt[2]-p[2])
        n_ships = max(1, int(p[5] * FRAC_BINS[frac_idx]))
        launches.append([int(p[0]), angle, n_ships])

    return launches
'''

    main_py = os.path.join(os.path.dirname(output_path) or ".", "main.py")
    with open(main_py, "w") as f:
        f.write(agent_code)

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(main_py, arcname="main.py")

    sz = os.path.getsize(output_path) / 1024
    print(f"Written: {output_path} ({sz:.0f} KB)")
    print("Submit this file to the competition.")


if __name__ == "__main__":
    cp = sys.argv[1] if len(sys.argv) > 1 else "best_model.pt"
    make_submission(cp)
