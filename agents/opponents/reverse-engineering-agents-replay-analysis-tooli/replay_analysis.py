# Reference code extracted from reverse-engineering-agents-replay-analysis-tooli.ipynb (NOT a playable agent).
# May contain plotting/analysis cells; kept verbatim for study.

import json
import glob
import math
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Configure your replay folder and agent name(s) here
REPLAY_FOLDER = '/kaggle/input/datasets/aidensong123/replays'   # change this if your replays live elsewhere
TARGET_AGENT = 'AidenSong123'    # the agent whose play you want to analyze
ONLY_WINS = True            # set False to analyze all games (wins+losses)

# %% ---- next cell ----

def load_replays(folder, agent_name):
    """Load all replays where agent_name played, plus their seat and outcome."""
    out = []
    for fp in sorted(glob.glob(os.path.join(folder, '*.json'))):
        with open(fp) as f:
            d = json.load(f)
        names = [a.get('Name','?') for a in d.get('info',{}).get('Agents', [])]
        seat = next((i for i,n in enumerate(names) if n == agent_name), -1)
        if seat == -1:
            continue
        rewards = d.get('rewards', [])
        won = (rewards[seat] == 1) if seat < len(rewards) else False
        n_p = len(d['steps'][0])
        out.append({'path': fp, 'data': d, 'seat': seat, 'won': won, 'n_players': n_p,
                    'agents': names})
    return out

def classify_target(action, planets, agent_seat):
    """Identify what planet a (src,angle,ships) action would hit, by ray-tracing."""
    if not isinstance(action, (list, tuple)) or len(action) != 3:
        return None, None, None
    try:
        src_id = int(action[0]); angle = float(action[1]); ships = int(action[2])
    except Exception:
        return None, None, None
    src = next((p for p in planets if p[0] == src_id), None)
    if src is None:
        return None, None, None
    sx, sy = src[2], src[3]
    cosA, sinA = math.cos(angle), math.sin(angle)
    best = None; best_t = 1e18
    for p in planets:
        if p[0] == src_id: continue
        dx, dy = p[2]-sx, p[3]-sy
        proj = dx*cosA + dy*sinA
        if proj <= 0: continue
        perp_sq = dx*dx + dy*dy - proj*proj
        if perp_sq < p[4]**2 and proj < best_t:
            best, best_t = p, proj
    return src, best, ships

replays = load_replays(REPLAY_FOLDER, TARGET_AGENT)
print(f'Loaded {len(replays)} replays where {TARGET_AGENT} played')
wins = [r for r in replays if r['won']]
losses = [r for r in replays if not r['won']]
print(f'  Wins: {len(wins)}, Losses: {len(losses)}')
print(f'  2P: {sum(1 for r in replays if r["n_players"]==2)}, 4P: {sum(1 for r in replays if r["n_players"]==4)}')
games = wins if ONLY_WINS else replays

# %% ---- next cell ----

first_action_turns = []
first_n_targets = Counter()
first_n_ships = []
FIRST_N = 5

for r in games:
    found = []
    for t, step in enumerate(r['data']['steps']):
        if t == 0: continue
        if not isinstance(step, list): continue
        if r['seat'] >= len(step): continue
        obs = step[0].get('observation', {})
        actions = step[r['seat']].get('action', [])
        if not isinstance(actions, list): continue
        for a in actions:
            src, target, ships = classify_target(a, obs.get('planets', []) or [], r['seat'])
            if ships is None: continue
            found.append((t, src, target, ships))
            if len(found) >= FIRST_N: break
        if len(found) >= FIRST_N: break
    if found:
        first_action_turns.append(found[0][0])
        for t, src, target, ships in found:
            if target is None:
                first_n_targets[('unknown', 0)] += 1
            else:
                owner = target[1]
                cls = 'own' if owner == r['seat'] else 'neutral' if owner == -1 else 'enemy'
                first_n_targets[(cls, target[6])] += 1
            first_n_ships.append(ships)

print(f'First action turn: avg={sum(first_action_turns)/len(first_action_turns):.1f}, '
      f'min={min(first_action_turns)}, max={max(first_action_turns)}')
print(f'Median ships in first {FIRST_N} launches: {sorted(first_n_ships)[len(first_n_ships)//2]}')
print(f'\nTarget classification (first {FIRST_N} launches per game):')
for (cls, prod), c in first_n_targets.most_common(10):
    print(f'  {cls:8s} prod={prod}: {c:>4d}x')

# %% ---- next cell ----

reaction_times = []
for r in games[:30]:
    steps = r['data']['steps']
    seat = r['seat']
    prev_enemy_ids = set()
    for t in range(1, len(steps)-1):
        if not isinstance(steps[t], list): continue
        obs = steps[t][0].get('observation', {})
        cur_ids = set(f[0] for f in obs.get('fleets', []) or [] if f[1] != seat and f[1] != -1)
        new_ids = cur_ids - prev_enemy_ids
        if new_ids:
            for dt in range(1, 6):
                if t+dt >= len(steps): break
                if not isinstance(steps[t+dt], list): break
                acts = steps[t+dt][seat].get('action', [])
                if isinstance(acts, list) and acts:
                    reaction_times.append(dt)
                    break
        prev_enemy_ids = cur_ids

rc = Counter(reaction_times)
print(f'n={len(reaction_times)} enemy fleet detections')
for k in sorted(rc.keys()):
    print(f'  +{k} turn: {rc[k]:>5d}x ({rc[k]/len(reaction_times)*100:.1f}%)')

# Plot
plt.figure(figsize=(8,3))
ks = sorted(rc.keys())
plt.bar(ks, [rc[k] for k in ks])
plt.xlabel('Turns until counter-launch')
plt.ylabel('Count')
plt.title(f'{TARGET_AGENT} reaction time to enemy fleet appearance')
plt.show()

# %% ---- next cell ----

phase_volumes = []  # [(early, mid_open, mid, midlate, late) per game]
for r in games:
    counts = [0]*5
    for t, step in enumerate(r['data']['steps']):
        if t == 0: continue
        if not isinstance(step, list): continue
        if r['seat'] >= len(step): continue
        actions = step[r['seat']].get('action', [])
        n = len(actions) if isinstance(actions, list) else 0
        if t < 50: counts[0] += n
        elif t < 100: counts[1] += n
        elif t < 200: counts[2] += n
        elif t < 300: counts[3] += n
        else: counts[4] += n
    phase_volumes.append(counts)

if phase_volumes:
    avgs = [sum(g[i] for g in phase_volumes)/len(phase_volumes) for i in range(5)]
    labels = ['0-50', '50-100', '100-200', '200-300', '300+']
    print('Avg launches per game by phase:')
    for l, v in zip(labels, avgs):
        print(f'  {l:>10s}: {v:>6.1f}')
    plt.figure(figsize=(8,3))
    plt.bar(labels, avgs, color=['#74a9cf','#3690c0','#0570b0','#045a8d','#023858'])
    plt.ylabel('Avg launches/game')
    plt.title(f'{TARGET_AGENT} launch volume by phase')
    plt.show()

# %% ---- next cell ----

own_sizes = []; neut_sizes = []; enemy_sizes = []
for r in games[:50]:
    for t, step in enumerate(r['data']['steps']):
        if t == 0 or not isinstance(step, list): continue
        if r['seat'] >= len(step): continue
        obs = step[0].get('observation', {})
        planets = obs.get('planets', []) or []
        actions = step[r['seat']].get('action', [])
        if not isinstance(actions, list): continue
        for a in actions:
            _, target, ships = classify_target(a, planets, r['seat'])
            if ships is None or target is None: continue
            owner = target[1]
            if owner == r['seat']: own_sizes.append(ships)
            elif owner == -1: neut_sizes.append(ships)
            else: enemy_sizes.append(ships)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, sizes, label in zip(axes, [own_sizes, neut_sizes, enemy_sizes],
                            ['own (reinforce)', 'neutral', 'enemy (attack)']):
    if sizes:
        ax.hist([min(s, 200) for s in sizes], bins=40)
        ax.set_xlabel('Ships per fleet (capped at 200)')
        ax.set_ylabel('Count')
        sizes_sorted = sorted(sizes)
        median = sizes_sorted[len(sizes_sorted)//2]
        ax.set_title(f'{label} (n={len(sizes)}, median={median})')
plt.tight_layout(); plt.show()

# %% ---- next cell ----

coordinated_turns = 0
multi_action_turns = 0
for r in games[:30]:
    for t, step in enumerate(r['data']['steps']):
        if t == 0 or not isinstance(step, list): continue
        if r['seat'] >= len(step): continue
        obs = step[0].get('observation', {})
        planets = obs.get('planets', []) or []
        actions = step[r['seat']].get('action', [])
        if not isinstance(actions, list) or len(actions) < 2: continue
        multi_action_turns += 1
        target_ids = []
        for a in actions:
            _, target, _ = classify_target(a, planets, r['seat'])
            if target is not None:
                target_ids.append(target[0])
        if target_ids:
            tc = Counter(target_ids)
            if tc.most_common(1)[0][1] >= 2:
                coordinated_turns += 1

print(f'Coordinated multi-source turns: {coordinated_turns}/{multi_action_turns} = {coordinated_turns/max(1,multi_action_turns)*100:.1f}%')

# %% ---- next cell ----

phase_gains = defaultdict(int)
phase_losses = defaultdict(int)
for r in games:
    seat = r['seat']
    prev_count = 0
    for t, step in enumerate(r['data']['steps']):
        if t == 0 or not isinstance(step, list): continue
        obs = step[0].get('observation', {})
        my_count = sum(1 for p in obs.get('planets', []) or [] if p[1] == seat)
        delta = my_count - prev_count
        phase = '0-50' if t < 50 else '50-200' if t < 200 else '200+'
        if delta > 0: phase_gains[phase] += delta
        elif delta < 0: phase_losses[phase] += -delta
        prev_count = my_count

print('Total planet captures and losses by phase (across all games):')
for phase in ['0-50', '50-200', '200+']:
    g = phase_gains[phase]; l = phase_losses[phase]
    net = g - l
    print(f'  {phase:>10s}: gained={g:>4d}, lost={l:>4d}, net={net:>+4d}')

# %% ---- next cell ----

comet_targets = 0; total = 0
for r in games:
    for t, step in enumerate(r['data']['steps']):
        if t == 0 or not isinstance(step, list): continue
        if r['seat'] >= len(step): continue
        obs = step[0].get('observation', {})
        comet_ids = set(obs.get('comet_planet_ids', []) or [])
        planets = obs.get('planets', []) or []
        actions = step[r['seat']].get('action', [])
        if not isinstance(actions, list): continue
        for a in actions:
            _, target, _ = classify_target(a, planets, r['seat'])
            if target is None: continue
            total += 1
            if target[0] in comet_ids:
                comet_targets += 1

print(f'Comet-targeting fraction: {comet_targets}/{total} = {comet_targets/max(1,total)*100:.2f}%')