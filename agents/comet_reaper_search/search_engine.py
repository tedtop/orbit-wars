"""Monte-Carlo move evaluation for comet_reaper_search.

A flat-array Numba forward model (simplified *economics*: production, fleet flight by
straight-line distance, sequential-arrival combat — drops orbital motion/sun/comets) +
an aggressive rollout policy. We score each candidate wave-set this turn by averaging
the outcome of K full rollouts under that policy — the lookahead the 1-ply engine lacks.

Stage A (this file): the rollout sim + a self-test. Encoder + integration come next.
"""
from __future__ import annotations

import numpy as np
import numba as nb

MAXF = 512  # max simultaneously in-flight fleets per rollout


@nb.njit(cache=True, fastmath=True, inline="always")
def _speed(n):
    if n < 1.0:
        n = 1.0
    s = 1.0 + 5.0 * (np.log(n) / np.log(1000.0)) ** 1.5
    return 6.0 if s > 6.0 else s


@nb.njit(cache=True, fastmath=True)
def rollout(ships0, owner0, prod, D,
            f_ships0, f_owner0, f_target0, f_eta0, nf0,
            n_turns, n_players, frac, min_launch):
    """One rollout; returns per-player total ships (garrison + in-flight) after n_turns."""
    P = ships0.shape[0]
    ships = ships0.copy()
    owner = owner0.copy()
    fs = np.zeros(MAXF); fo = np.zeros(MAXF, np.int64)
    ft = np.zeros(MAXF, np.int64); fe = np.zeros(MAXF, np.int64)
    nf = 0
    for k in range(nf0):
        if nf < MAXF:
            fs[nf] = f_ships0[k]; fo[nf] = f_owner0[k]; ft[nf] = f_target0[k]; fe[nf] = f_eta0[k]; nf += 1

    for _ in range(n_turns):
        for i in range(P):                                  # production
            if owner[i] >= 0:
                ships[i] += prod[i]
        for i in range(P):                                  # aggressive rollout policy
            if owner[i] >= 0 and ships[i] >= min_launch:
                best = -1; bestd = 1e18
                for j in range(P):
                    if j != i and owner[j] != owner[i] and D[i, j] < bestd:
                        bestd = D[i, j]; best = j
                if best >= 0:
                    send = np.floor(frac * (ships[i] - min_launch))   # surplus above reserve only
                    if send >= 1.0 and nf < MAXF:
                        eta = int(np.ceil(bestd / _speed(send)))
                        if eta < 1:
                            eta = 1
                        ships[i] -= send
                        fs[nf] = send; fo[nf] = owner[i]; ft[nf] = best; fe[nf] = eta; nf += 1
        w = 0                                               # advance fleets + resolve arrivals
        for k in range(nf):
            fe[k] -= 1
            if fe[k] <= 0:
                tg = ft[k]
                if owner[tg] == fo[k]:
                    ships[tg] += fs[k]
                else:
                    ships[tg] -= fs[k]
                    if ships[tg] < 0.0:
                        owner[tg] = fo[k]; ships[tg] = -ships[tg]
            else:
                fs[w] = fs[k]; fo[w] = fo[k]; ft[w] = ft[k]; fe[w] = fe[k]; w += 1
        nf = w

    tot = np.zeros(n_players)
    for i in range(P):
        if owner[i] >= 0:
            tot[owner[i]] += ships[i]
    for k in range(nf):
        tot[fo[k]] += fs[k]
    return tot


@nb.njit(cache=True, fastmath=True)
def rollout_many(ships0, owner0, prod, D, f_ships0, f_owner0, f_target0, f_eta0, nf0,
                 n_turns, n_players, frac, min_launch, K):
    """K rollouts (the policy here is deterministic, so K averages model stochasticity
    once we add jitter); returns mean per-player totals."""
    acc = np.zeros(n_players)
    for _ in range(K):
        acc += rollout(ships0, owner0, prod, D, f_ships0, f_owner0, f_target0, f_eta0, nf0,
                       n_turns, n_players, frac, min_launch)
    return acc / K


import math


def _speed_py(n):
    n = max(float(n), 1.0)
    return min(1.0 + 5.0 * (math.log(n) / math.log(1000.0)) ** 1.5, 6.0)


def encode(obs, n_players):
    """Raw obs dict -> flat arrays for the rollout sim. Existing in-flight fleets are
    mapped to a target planet by best heading-alignment + straight-line ETA (approx)."""
    planets = obs["planets"]
    P = len(planets)
    ships = np.array([p[5] for p in planets], np.float64)
    owner = np.array([int(p[1]) for p in planets], np.int64)
    prod = np.array([p[6] for p in planets], np.float64)
    px = np.array([p[2] for p in planets], np.float64)
    py = np.array([p[3] for p in planets], np.float64)
    pr = np.array([p[4] for p in planets], np.float64)
    D = np.sqrt((px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2)
    id2slot = {int(p[0]): i for i, p in enumerate(planets)}
    fs, fo, ft, fe = [], [], [], []
    for fl in obs.get("fleets", []):
        if int(fl[0]) < 0:
            continue
        fx, fy, fang, fsh, fow = float(fl[2]), float(fl[3]), float(fl[4]), float(fl[6]), int(fl[1])
        dirx, diry = math.cos(fang), math.sin(fang)
        best, bestal = -1, -1.0
        for j in range(P):
            dx, dy = px[j] - fx, py[j] - fy
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            al = (dx * dirx + dy * diry) / dist
            if al > bestal:
                bestal, best = al, j
        if best >= 0 and bestal > 0.85:
            eta = int(max(1, math.ceil(math.hypot(px[best] - fx, py[best] - fy) / _speed_py(fsh))))
            fs.append(fsh); fo.append(fow); ft.append(best); fe.append(eta)
    return {
        "ships": ships, "owner": owner, "prod": prod, "D": D, "pr": pr, "px": px, "py": py,
        "f_ships": np.array(fs, np.float64), "f_owner": np.array(fo, np.int64),
        "f_target": np.array(ft, np.int64), "f_eta": np.array(fe, np.int64),
        "id2slot": id2slot, "P": P,
    }


def encode_from_tensors(obs_tensors, n_players):
    """Encode directly from the engine's tensor obs (slot order matches cand_src/cand_tgt_slot)."""
    pl = obs_tensors["planets"].detach().to("cpu").numpy()
    P = pl.shape[0]
    pid_col = pl[:, 0]
    owner = np.where(pid_col >= 0, pl[:, 1], -2).astype(np.int64)   # dead (id<0) -> inert
    px = pl[:, 2].astype(np.float64); py = pl[:, 3].astype(np.float64)
    ships = pl[:, 5].astype(np.float64); prod = pl[:, 6].astype(np.float64)
    D = np.sqrt((px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2)
    fs, fo, ft, fe = [], [], [], []
    fl = obs_tensors.get("fleets")
    if fl is not None:
        fl = fl.detach().to("cpu").numpy()
        for r in fl:
            if r[0] < 0:
                continue
            fx, fy, fang, fsh, fow = float(r[2]), float(r[3]), float(r[4]), float(r[6]), int(r[1])
            dirx, diry = math.cos(fang), math.sin(fang)
            best, bestal = -1, -1.0
            for j in range(P):
                dx, dy = px[j] - fx, py[j] - fy
                dist = math.hypot(dx, dy)
                if dist < 1e-6:
                    continue
                al = (dx * dirx + dy * diry) / dist
                if al > bestal:
                    bestal, best = al, j
            if best >= 0 and bestal > 0.85:
                eta = int(max(1, math.ceil(math.hypot(px[best] - fx, py[best] - fy) / _speed_py(fsh))))
                fs.append(fsh); fo.append(fow); ft.append(best); fe.append(eta)
    return {"ships": ships, "owner": owner, "prod": prod, "D": D, "P": P,
            "f_ships": np.array(fs, np.float64), "f_owner": np.array(fo, np.int64),
            "f_target": np.array(ft, np.int64), "f_eta": np.array(fe, np.int64)}


def eval_candidates(enc, pid, cand_src, cand_tgt, cand_ships, n_players, H=40, frac=0.5, min_launch=4.0):
    """Forward-sim value of each single-wave candidate (src->tgt, ships). Vectorizable later;
    deterministic rollout so K=1. Returns a numpy array of (my_net - best_opp_net) per candidate."""
    base_s, base_o, prod, D = enc["ships"], enc["owner"], enc["prod"], enc["D"]
    ef_s, ef_o, ef_t, ef_e = enc["f_ships"], enc["f_owner"], enc["f_target"], enc["f_eta"]
    n0 = len(ef_s)
    out = np.zeros(len(cand_src))
    for c in range(len(cand_src)):
        src, tgt, sh = int(cand_src[c]), int(cand_tgt[c]), float(cand_ships[c])
        s = base_s.copy()
        fs = np.concatenate([ef_s, np.zeros(1)]); fo = np.concatenate([ef_o, np.zeros(1, np.int64)])
        ft = np.concatenate([ef_t, np.zeros(1, np.int64)]); fe = np.concatenate([ef_e, np.zeros(1, np.int64)])
        nn = n0
        if 1 <= sh <= s[src] and src != tgt:
            s[src] -= sh
            eta = int(max(1, math.ceil(D[src, tgt] / _speed_py(sh))))
            fs[nn] = sh; fo[nn] = pid; ft[nn] = tgt; fe[nn] = eta; nn += 1
        tot = rollout(s, base_o, prod, D, fs[:nn], fo[:nn], ft[:nn], fe[:nn], nn,
                      H, n_players, frac, min_launch)
        opp = max((tot[o] for o in range(n_players) if o != pid), default=0.0)
        out[c] = tot[pid] - opp
    return out


def mc_eval(enc, pid, plans, n_players, K=64, H=40, frac=0.5, min_launch=4.0):
    """Score each candidate plan by mean rollout outcome. A plan = list of
    (source_slot, target_slot, ships). Score = my_net - best_opponent_net."""
    base_s, base_o, prod, D = enc["ships"], enc["owner"], enc["prod"], enc["D"]
    ef_s, ef_o, ef_t, ef_e = enc["f_ships"], enc["f_owner"], enc["f_target"], enc["f_eta"]
    scores = []
    for plan in plans:
        s = base_s.copy()
        nf_extra = len(plan)
        fs = np.concatenate([ef_s, np.zeros(nf_extra)])
        fo = np.concatenate([ef_o, np.zeros(nf_extra, np.int64)])
        ft = np.concatenate([ef_t, np.zeros(nf_extra, np.int64)])
        fe = np.concatenate([ef_e, np.zeros(nf_extra, np.int64)])
        n = len(ef_s)
        for (src, tgt, sh) in plan:
            sh = min(float(sh), s[src])
            if sh < 1 or src == tgt:
                continue
            s[src] -= sh
            eta = int(max(1, math.ceil(D[src, tgt] / _speed_py(sh))))
            fs[n] = sh; fo[n] = pid; ft[n] = tgt; fe[n] = eta; n += 1
        tot = rollout_many(s, base_o, prod, D, fs[:n], fo[:n], ft[:n], fe[:n], n,
                           H, n_players, frac, min_launch, K)
        opp = max((tot[o] for o in range(n_players) if o != pid), default=0.0)
        scores.append(tot[pid] - opp)
    return scores


def _precompile():
    """Trigger JIT compilation at import time so the ~1s compile is off the turn-0 clock."""
    try:
        z = np.zeros(0); zi = np.zeros(0, np.int64); D = np.ones((2, 2))
        rollout(np.array([10.0, 10.0]), np.array([0, 1], np.int64), np.array([1.0, 1.0]),
                D, z, zi, zi, zi, 0, 2, 2, 0.5, 4.0)
    except Exception:
        pass


_precompile()


def _self_test():
    import time
    rng = np.random.default_rng(0)
    P, n_players, H = 24, 2, 40        # short lookahead horizon, not the whole game
    px = rng.random(P) * 100; py = rng.random(P) * 100
    D = np.sqrt((px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2)
    prod = (rng.random(P) * 4 + 1)
    ships = rng.random(P) * 20
    owner = np.full(P, -1, np.int64); owner[0] = 0; owner[1] = 1   # one home each, rest neutral
    e = np.zeros(0); ei = np.zeros(0, np.int64)

    def roll(s):
        return rollout(s, owner, prod, D, e, ei, ei, ei, 0, H, n_players, 0.5, 4.0)

    t0 = time.time(); base = roll(ships)
    print(f"compile+1 rollout (H={H}): {time.time()-t0:.2f}s  -> p0={base[0]:.0f} p1={base[1]:.0f}")
    N = 4000; t0 = time.time()
    for _ in range(N):
        roll(ships)
    dt = time.time() - t0
    print(f"{N} {H}-turn rollouts: {dt:.2f}s -> {1e6*dt/N:.0f} us each, ~{int(0.8*N/dt):,} per 800ms turn")
    # sanity: a head start should raise our score MONOTONICALLY and smoothly (no cliff)
    print("graded head-start (p0 advantage = p0-p1):")
    for boost in (0, 50, 100, 200):
        s = ships.copy(); s[0] += boost; t = roll(s)
        print(f"  +{boost:>3}: p0={t[0]:6.0f} p1={t[1]:6.0f}  adv={t[0]-t[1]:+7.0f}")


def _test_eval():
    """Fidelity probe on a real replay state: MC eval should rank a sensible nearby
    capture > no-op > a wasteful far capture."""
    import json, glob, time
    rp = next(iter(sorted(glob.glob("replays/*/*.json"))), None)
    if rp is None:
        print("no replay to probe"); return
    d = json.load(open(rp))
    n = len(d["info"]["TeamNames"])
    obs = d["steps"][min(60, len(d["steps"]) - 1)][0]["observation"]
    pid = 0
    enc = encode(obs, n)
    owner, ships, D = enc["owner"], enc["ships"], enc["D"]
    mine = [i for i in range(enc["P"]) if owner[i] == pid]
    if not mine:
        print("focal owns nothing at this step"); return
    src = max(mine, key=lambda i: ships[i])
    notmine = [j for j in range(enc["P"]) if owner[j] != pid]
    near = min(notmine, key=lambda j: D[src, j])
    far = max(notmine, key=lambda j: D[src, j])
    send = float(ships[src] * 0.7)
    plans = {
        "capture_near": [(src, near, send)],
        "no_op": [],
        "capture_far": [(src, far, send)],
    }
    t0 = time.time()
    sc = mc_eval(enc, pid, list(plans.values()), n, K=64, H=40)
    dt = (time.time() - t0) * 1000
    print(f"\nfidelity probe ({rp.split('/')[-1]}, {n}P, step60, src ships={ships[src]:.0f}):")
    for name, s in zip(plans, sc):
        print(f"  {name:14s} score={s:+8.1f}")
    ok = sc[0] > sc[1] and sc[0] > sc[2]
    print(f"  -> near>{'noop' } and near>far ? {'YES (sim ranks moves sensibly)' if ok else 'NO (low fidelity!)'}")
    print(f"  eval of 3 plans x 64 rollouts: {dt:.0f}ms")


if __name__ == "__main__":
    _self_test()
    _test_eval()
