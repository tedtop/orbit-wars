#!/usr/bin/env bash
# Behavior-clone individual top bots into sparring opponents for the self-play league.
# Each clone is our PlanetPolicy trained ONLY on that team's prize-zone moves.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATA=training/moves_prizezone.jsonl.gz

# team name  ->  output checkpoint stem
names=("Jake Will"      "Xiangyu Liu"      "M & J & M.ver2"   "TonyK")
outs=( "clone_jake_will" "clone_xiangyu_liu" "clone_mjm"        "clone_tonyk")

for i in "${!names[@]}"; do
  echo "=== cloning '${names[$i]}' -> training/${outs[$i]}.pt ==="
  $PY rl/bc_train.py --data "$DATA" --min-rating 1500 --team "${names[$i]}" \
      --limit 60000 --steps 8000 --noop-weight 0.12 \
      --out "training/${outs[$i]}.pt" 2>&1 | grep -vE "cabt|open_spiel|INFO:"
done
echo "=== all clones trained ==="
ls -1 training/clone_*.pt
