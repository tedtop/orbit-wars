"""
Champion-vs-challenger eval harness.

Mirrors the eval format used by top competitors (Baran Kucuk's eval_checkpoints.py):
  - Large-n head-to-head between two checkpoints
  - Seat-rotation: half games A as P0, half as P1
  - Reports slot_0/slot_1 win rates (symmetry check — target ~0.50/0.50)
  - Reports a_as_p0 / a_as_p1 (agent-seat consistency check)
  - Reports score gap, episode length by outcome, terminal win pct

Usage:
  python eval_checkpoints.py --checkpoint-a runs/job1/best_model.pt \
                              --checkpoint-b runs/job2/best_model.pt \
                              --n-games 200

  python eval_checkpoints.py --checkpoint-a runs/job1/best_model.pt \
                              --vs-greedy --n-games 200
"""

import argparse, json, logging, math, os, sys
import torch

# Suppress kaggle_environments INFO spam to stdout so --json output is clean JSON
logging.getLogger("kaggle_environments").setLevel(logging.WARNING)

from kaggle_environments import make

# Import shared model/inference code from train.py
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    ActorCritic, ActorCriticET, _build_policy,
    collect_planet_actions, make_launches,
    DEVICE, TOTAL_STEPS, MIN_SHIPS
)


def load_policy(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model_type = ckpt.get("model_type", "mlp")
    policy = _build_policy(model_type)
    policy.load_state_dict(ckpt["state"])
    policy.eval()
    meta = {"update": ckpt.get("update", "?"), "steps": ckpt.get("steps", "?"),
            "model_type": model_type}
    return policy, meta


def make_rl_agent(policy):
    # Use context manager not decorator — @torch.no_grad() changes the wrapper
    # signature, which kaggle_environments' _get_args() misreads as 0-arg when
    # both agents are RL closures, causing silent ep_length=2 draws.
    def agent(obs):
        with torch.no_grad():
            d = obs if isinstance(obs, dict) else {
                "player": obs.player, "planets": obs.planets,
                "fleets": obs.fleets, "step": getattr(obs, "step", 0)}
            actions, _ = collect_planet_actions(policy, [d])
            return make_launches(d, actions[0])
    return agent


def make_comet_agent():
    """Load comet_reaper as eval opponent B."""
    import importlib.util, sys as _sys
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "comet_reaper", "main.py")
    spec = importlib.util.spec_from_file_location("_comet_eval", path)
    mod  = importlib.util.module_from_spec(spec)
    _sys.modules["_comet_eval"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def make_greedy_agent():
    def agent(obs):
        d = obs if isinstance(obs, dict) else {
            "player": obs.player, "planets": obs.planets, "fleets": obs.fleets}
        player  = int(d.get("player", 0))
        planets = d.get("planets", [])
        my = [p for p in planets if p[1] == player and p[5] >= MIN_SHIPS]
        if not my:
            return []
        src = max(my, key=lambda p: p[5])
        targets = [p for p in planets if p[1] != player]
        if not targets:
            return []
        tgt = min(targets, key=lambda p: math.hypot(p[2]-src[2], p[3]-src[3]))
        angle = math.atan2(tgt[3]-src[3], tgt[2]-src[2])
        return [[int(src[0]), angle, int(src[5])]]
    return agent


def run_game(agent_a, agent_b, seed, agents_order):
    """Run one game. agents_order: [agent_a, agent_b] or [agent_b, agent_a]."""
    env = make("orbit_wars", debug=False,
               configuration={"seed": seed, "episodeSteps": TOTAL_STEPS})
    env.run(agents_order)
    steps = env.steps
    r0 = steps[-1][0].reward or 0
    r1 = steps[-1][1].reward or 0
    ep_len = len(steps)

    # Terminal = one player eliminated (reward ±1 from win, not timeout)
    # In orbit_wars, terminal wins have reward=1.0 vs timeout wins may differ
    # We detect terminal by checking if all planets belong to one player at end
    final_obs = steps[-1][0].observation if steps[-1][0].observation else None
    is_terminal = False
    if final_obs and hasattr(final_obs, 'planets'):
        planets = final_obs.planets or []
        owners = set(int(p[1]) for p in planets if p[1] is not None)
        is_terminal = len(owners) <= 1
    elif final_obs and isinstance(final_obs, dict):
        planets = final_obs.get('planets', [])
        owners = set(int(p[1]) for p in planets if p[1] is not None)
        is_terminal = len(owners) <= 1

    p0_score = steps[-1][0].observation.get('my_score', r0) if isinstance(
        getattr(steps[-1][0], 'observation', None), dict) else r0
    p1_score = steps[-1][1].observation.get('my_score', r1) if isinstance(
        getattr(steps[-1][1], 'observation', None), dict) else r1

    return r0, r1, ep_len, is_terminal


def evaluate(agent_a, agent_b, n_games=200, label_a="A", label_b="B"):
    """
    Run n_games head-to-head with seat rotation (half as P0, half as P1).
    Seat alternation is what makes slot_0/slot_1 a valid symmetry check:
    strong player is split evenly across both seats, so any slot deviation
    signals environment/encoding asymmetry, not skill.
    Returns stats dict matching the spec (Baran Kucuk eval_checkpoints.py format).
    """
    half = n_games // 2

    a_wins = 0; b_wins = 0; draws = 0
    slot0_wins = 0; slot1_wins = 0  # P0/P1 wins regardless of who is playing that seat
    a_as_p0_wins = 0; a_as_p1_wins = 0

    ep_lengths_a_win = []; ep_lengths_a_loss = []; ep_lengths_a_draw = []
    terminal_a_wins = 0; terminal_b_wins = 0
    a_rewards = []; b_rewards = []

    print(f"Running {half} games A=P0 + {half} games A=P1...", flush=True)

    # First half: A in seat 0, B in seat 1
    for seed in range(half):
        r0, r1, ep_len, is_term = run_game(agent_a, agent_b, seed, [agent_a, agent_b])
        a_rewards.append(r0); b_rewards.append(r1)
        if r0 > r1:
            a_wins += 1; a_as_p0_wins += 1; slot0_wins += 1
            ep_lengths_a_win.append(ep_len)
            if is_term: terminal_a_wins += 1
        elif r1 > r0:
            b_wins += 1; slot1_wins += 1
            ep_lengths_a_loss.append(ep_len)
            if is_term: terminal_b_wins += 1
        else:
            draws += 1; ep_lengths_a_draw.append(ep_len)
        if (seed + 1) % 20 == 0:
            print(f"  A=P0: {seed+1}/{half} done | A wins: {a_as_p0_wins}", flush=True)

    # Second half: B in seat 0, A in seat 1
    for seed in range(half, n_games):
        r0, r1, ep_len, is_term = run_game(agent_b, agent_a, seed, [agent_b, agent_a])
        a_rewards.append(r1); b_rewards.append(r0)
        if r1 > r0:
            a_wins += 1; a_as_p1_wins += 1; slot1_wins += 1
            ep_lengths_a_win.append(ep_len)
            if is_term: terminal_a_wins += 1
        elif r0 > r1:
            b_wins += 1; slot0_wins += 1
            ep_lengths_a_loss.append(ep_len)
            if is_term: terminal_b_wins += 1
        else:
            draws += 1; ep_lengths_a_draw.append(ep_len)
        if (seed - half + 1) % 20 == 0:
            print(f"  A=P1: {seed-half+1}/{half} done | A wins: {a_as_p1_wins}", flush=True)

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    slot0_wr = slot0_wins / n_games
    slot1_wr = slot1_wins / n_games
    a_p0_wr  = a_as_p0_wins / half
    a_p1_wr  = a_as_p1_wins / half

    # Hard gate thresholds:
    #   seat consistency: |a_as_p0 - a_as_p1| < 0.05 — the primary encoding symmetry test
    #   slot balance: among decisive games only (exclude draws), P0/P1 split should be ~50/50
    #   Raw slot rates can't reach [0.46,0.54] when draw_rate > ~8% — use decisive-only denom.
    decisive = slot0_wins + slot1_wins  # total non-draw games
    slot0_decisive = slot0_wins / decisive if decisive > 0 else 0.5
    seat_cons_ok = abs(a_p0_wr - a_p1_wr) < 0.05
    slot_sym_ok  = (0.46 <= slot0_decisive <= 0.54)

    stats = {
        "checkpoint_a":                  label_a,
        "checkpoint_b":                  label_b,
        "n_games":                       n_games,
        "a_win_rate":                    a_wins / n_games,
        "b_win_rate":                    b_wins / n_games,
        "draw_rate":                     draws  / n_games,
        "a_loss_rate":                   b_wins / n_games,
        # Slot symmetry (HARD gate — [0.46, 0.54] each; ref: 0.499/0.4997)
        "slot_0_win_rate":               slot0_wr,
        "slot_1_win_rate":               slot1_wr,
        # Agent-seat consistency (HARD gate — within 0.05)
        "a_as_p0_win_rate":              a_p0_wr,
        "a_as_p1_win_rate":              a_p1_wr,
        # Reward/score tracking
        "a_mean_reward":                 mean(a_rewards),
        "b_mean_reward":                 mean(b_rewards),
        "score_gap_mean":                mean(a_rewards) - mean(b_rewards),
        # Episode length by outcome
        "mean_episode_length":           mean(ep_lengths_a_win + ep_lengths_a_loss + ep_lengths_a_draw),
        "mean_episode_length_a_win":     mean(ep_lengths_a_win),
        "mean_episode_length_a_loss":    mean(ep_lengths_a_loss),
        # Terminal win rate (decisive close-out vs timeout)
        "terminal_state_win_pct_a":      terminal_a_wins / max(a_wins, 1),
        "terminal_state_win_pct_b":      terminal_b_wins / max(b_wins, 1),
        # Gate results
        "slot_0_decisive_wr":            slot0_decisive,   # draw-adjusted: slot0/(slot0+slot1)
        "slot_symmetry_ok":              slot_sym_ok,
        "seat_consistency_ok":           seat_cons_ok,
        "symmetry_hard_gate_passed":     slot_sym_ok and seat_cons_ok,
    }
    return stats


def print_report(stats):
    gate = "✅ SYMMETRY GATE PASSED" if stats['symmetry_hard_gate_passed'] else "🛑 SYMMETRY GATE FAILED — win rate UNRELIABLE"
    print("\n" + "="*60)
    print(f"EVAL: {stats['checkpoint_a']}")
    print(f"  vs: {stats['checkpoint_b']}")
    print(f"  n={stats['n_games']} games  |  {gate}")
    print("="*60)

    if not stats['symmetry_hard_gate_passed']:
        print("  ⛔ DO NOT PROMOTE — symmetry gate failed.")
        print("     slot_0_decisive must be in [0.46, 0.54]; |a_as_p0 - a_as_p1| < 0.05.")
        print()

    dec_flag = "✓ [0.46–0.54]" if stats['slot_symmetry_ok'] else "⚠ OUT OF RANGE [0.46–0.54]"
    print(f"  slot_0_decisive:   {stats['slot_0_decisive_wr']:.3f}  {dec_flag}  (draw-adj)")
    sym_flag = "✓" if stats['slot_symmetry_ok'] else "⚠"
    print(f"  slot_0_win_rate:   {stats['slot_0_win_rate']:.3f}  (raw, includes draws)")
    print(f"  slot_1_win_rate:   {stats['slot_1_win_rate']:.3f}")
    print(f"  slot_1_win_rate:   {stats['slot_1_win_rate']:.3f}")
    seat_flag = "✓ <0.05" if stats['seat_consistency_ok'] else "⚠ SEAT BIAS (>0.05)"
    print(f"  a_as_p0_win_rate:  {stats['a_as_p0_win_rate']:.3f}  {seat_flag}")
    print(f"  a_as_p1_win_rate:  {stats['a_as_p1_win_rate']:.3f}")
    print()
    print(f"  A win rate:        {stats['a_win_rate']:.3f}  ({stats['a_win_rate']*100:.1f}%)")
    print(f"  B win rate:        {stats['b_win_rate']:.3f}")
    print(f"  Draw rate:         {stats['draw_rate']:.3f}")
    print(f"  A mean reward:     {stats['a_mean_reward']:.4f}  (B: {stats['b_mean_reward']:.4f})  gap: {stats['score_gap_mean']:+.4f}")
    print()
    print(f"  Mean ep length:    {stats['mean_episode_length']:.0f} steps")
    print(f"  Mean ep (A win):   {stats['mean_episode_length_a_win']:.0f} steps")
    print(f"  Mean ep (A loss):  {stats['mean_episode_length_a_loss']:.0f} steps")
    term_flag = "" if stats['terminal_state_win_pct_a'] > 0.40 else "  ⚠ LOW (policy winning on timeout, not close-out)"
    print(f"  Terminal win% A:   {stats['terminal_state_win_pct_a']:.1%}{term_flag}")
    print(f"  Terminal win% B:   {stats['terminal_state_win_pct_b']:.1%}")
    print("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", default=None)
    p.add_argument("--vs-greedy", action="store_true")
    p.add_argument("--vs-comet",  action="store_true", help="Use comet_reaper as opponent B")
    p.add_argument("--n-games", type=int, default=200)
    p.add_argument("--json", action="store_true", help="Output JSON only")
    args = p.parse_args()

    agent_a, meta_a = load_policy(args.checkpoint_a)
    label_a = f"{os.path.basename(args.checkpoint_a)} (U{meta_a['update']})"

    if args.vs_comet:
        agent_b = make_comet_agent()
        label_b = "comet_reaper"
    elif args.vs_greedy or args.checkpoint_b is None:
        agent_b = make_greedy_agent()
        label_b = "greedy"
    else:
        agent_b, meta_b = load_policy(args.checkpoint_b)
        label_b = f"{os.path.basename(args.checkpoint_b)} (U{meta_b['update']})"

    stats = evaluate(make_rl_agent(agent_a), agent_b, n_games=args.n_games,
                     label_a=label_a, label_b=label_b)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats)
