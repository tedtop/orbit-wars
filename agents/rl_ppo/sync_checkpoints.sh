#!/usr/bin/env bash
# Pull checkpoints from fleet, then run champion-vs-challenger eval.
#
# Promotion rules (spec):
#   1. Symmetry HARD gate must pass first (slot_0 and slot_1 each in [0.46,0.54],
#      |a_as_p0 - a_as_p1| < 0.05). If this fails, encoding is broken — no promote.
#   2. Challenger must beat current champion >0.50 win rate over n_games.
#   3. Previous champions kept in checkpoints/pool/ — never discarded.
#
# Usage: bash agents/rl_ppo/sync_checkpoints.sh
set -e

HOSTS=(
    "149.165.175.182"   # m3.quad
    "149.165.174.18"    # m3.2xl
    "149.165.174.133"
    "149.165.171.142"
    "149.165.170.73"
    "149.165.171.248"
    "149.165.175.105"   # m3.xl
    "149.165.170.84"
    "149.165.175.177"
)
USER="exouser"
REPO_DIR="/home/${USER}/orbit_wars"
LOCAL_OUT="agents/rl_ppo/runs/remote"
CHAMPION="agents/rl_ppo/checkpoints/champion.pt"
POOL_DIR="agents/rl_ppo/checkpoints/pool"
STATE_FILE="ORCHESTRATOR_STATE.md"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $HOME/.ssh/id_rsa"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../.venv/bin/python3"
N_GAMES=1000  # promotion + symmetry gate need large n: at 200, ±0.04 band is inside noise

mkdir -p "$LOCAL_OUT" "$(dirname "$CHAMPION")" "$POOL_DIR"

echo "=== Syncing checkpoints from ${#HOSTS[@]} hosts ==="

for HOST in "${HOSTS[@]}"; do
    HOST_DIR="${LOCAL_OUT}/${HOST}"
    mkdir -p "$HOST_DIR"
    rsync -az \
        --include='*/' --include='best_model.pt' --include='*.log' \
        --include='metrics.jsonl' --exclude='*' \
        -e "ssh $SSH_OPTS" \
        "${USER}@${HOST}:${REPO_DIR}/agents/rl_ppo/runs/" \
        "$HOST_DIR/" 2>/dev/null && echo "✓ $HOST synced" || echo "✗ $HOST unreachable"
done

# ── Champion selection ──────────────────────────────────────────────────────────
echo ""
echo "=== Champion selection ==="

CANDIDATES=()
while IFS= read -r line; do CANDIDATES+=("$line"); done < <(find "$LOCAL_OUT" -name "best_model.pt" 2>/dev/null | sort)

# Also sync opp_log.jsonl files (logging only — no eval impact)
for H in "${HOSTS[@]}"; do
    rsync -az -e "ssh $SSH_OPTS" \
      --include="*/" --include="opp_log.jsonl" --exclude="*" \
      "${USER}@${H}:${REPO_DIR}/agents/rl_ppo/runs/" \
      "${LOCAL_OUT}/${H}/" 2>/dev/null || true
done

if [ ${#CANDIDATES[@]} -eq 0 ]; then
    echo "No candidates yet — skipping."
    exit 0
fi

echo "Candidates: ${#CANDIDATES[@]}"

get_update() {
    "$PYTHON" -c "
import torch, sys
try:
    c = torch.load('$1', map_location='cpu')
    print(c.get('update', 0))
except:
    print(0)
" 2>/dev/null || echo 0
}

get_best_wr() {
    local CKPT="$1"
    local RUN_DIR; RUN_DIR=$(dirname "$CKPT")
    local METRICS="$RUN_DIR/metrics.jsonl"
    "$PYTHON" -c "
import torch, json, sys
ckpt, metrics = '$CKPT', '$METRICS'
printed = False
try:
    c = torch.load(ckpt, map_location='cpu')
    wr = c.get('best_wr', None)
    if wr is not None:
        print(float(wr))
        printed = True
except Exception:
    pass
if not printed:
    try:
        best = 0.0
        with open(metrics) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    if 'eval_as_p0' in d and 'eval_as_p1' in d:
                        wr = (d['eval_as_p0'] + d['eval_as_p1']) / 2
                        if wr > best:
                            best = wr
                except Exception:
                    pass
        print(best)
    except Exception:
        print(0.0)
" 2>/dev/null || echo 0.0
}

# Seed champion from best-eval checkpoint if none exists
if [ ! -f "$CHAMPION" ]; then
    BEST_SEED=""; BEST_WR_SEED=0
    for C in "${CANDIDATES[@]}"; do
        [[ "$C" == *"175.182"* ]] && continue
        WR=$(get_best_wr "$C")
        if awk "BEGIN{exit !($WR > $BEST_WR_SEED)}" 2>/dev/null; then
            BEST_WR_SEED=$WR; BEST_SEED=$C
        fi
    done
    if [ -n "$BEST_SEED" ]; then
        cp "$BEST_SEED" "$CHAMPION"
        echo "Seeded champion from $BEST_SEED (WR=$BEST_WR_SEED)"
    else
        echo "No trained checkpoints yet."; exit 0
    fi
fi

CURRENT_U=$(get_update "$CHAMPION")
CURRENT_WR=$(get_best_wr "$CHAMPION")

# Find challenger: best-eval 64-env candidate at least U+50 deeper than champion
# Ranked by greedy eval WR (not deepest U) — quality, not clock time
# Exclude m3.quad (175.182) — entropy-collapsed, 8-env seeds are not valid challengers
CHALLENGER=""; CHALLENGER_U=0; CHALLENGER_WR=0; CHALLENGER_LABEL=""
for C in "${CANDIDATES[@]}"; do
    [[ "$C" == *"175.182"* ]] && continue   # skip quad
    U=$(get_update "$C")
    WR=$(get_best_wr "$C")
    if [ "${U:-0}" -gt "$((CURRENT_U + 50))" ] 2>/dev/null; then
        # Primary sort: best WR; tiebreak: deepest U
        BETTER_WR=$(awk "BEGIN{print ($WR > $CHALLENGER_WR) ? 1 : 0}" 2>/dev/null)
        EQUAL_WR=$(awk "BEGIN{print ($WR == $CHALLENGER_WR) ? 1 : 0}" 2>/dev/null)
        DEEPER_U=0; [ "${U:-0}" -gt "$CHALLENGER_U" ] && DEEPER_U=1
        if [ "$BETTER_WR" = "1" ] || { [ "$EQUAL_WR" = "1" ] && [ "$DEEPER_U" = "1" ]; }; then
            CHALLENGER_WR=$WR
            CHALLENGER_U=$U
            CHALLENGER=$C
            CHALLENGER_LABEL=$(dirname "$C" | sed "s|$LOCAL_OUT/||")
        fi
    fi
done

if [ -z "$CHALLENGER" ]; then
    echo "Champion (U=$CURRENT_U WR=$CURRENT_WR) is best or no challenger >U=$((CURRENT_U+50)). Nothing to promote."
    RESULT_LINE="$(date '+%Y-%m-%d %H:%M') | champion=U${CURRENT_U} WR=${CURRENT_WR} | no challenger"
else
    echo "Champion: U=$CURRENT_U WR=$CURRENT_WR"
    echo "Challenger: $CHALLENGER_LABEL (U=$CHALLENGER_U WR=$CHALLENGER_WR)"
    echo "Running $N_GAMES-game eval (seat-balanced, symmetry hard gate)..."

    RAW=$("$PYTHON" "${SCRIPT_DIR}/eval_checkpoints.py" \
        --checkpoint-a "$CHALLENGER" \
        --checkpoint-b "$CHAMPION" \
        --n-games "$N_GAMES" \
        --json 2>/dev/null | "$PYTHON" -c "
import sys
data = sys.stdin.read()
start = data.rfind('{')
end   = data.rfind('}') + 1
print(data[start:end] if start >= 0 and end > start else '')
")

    WR=$(    echo "$RAW" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['a_win_rate']:.3f}\")")
    SYM_OK=$(echo "$RAW" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(d['symmetry_hard_gate_passed'])")
    SLOT0=$( echo "$RAW" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['slot_0_win_rate']:.3f}\")")
    SLOT1=$( echo "$RAW" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['slot_1_win_rate']:.3f}\")")
    SEAT=$( echo "$RAW"  | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(f\"{abs(d['a_as_p0_win_rate']-d['a_as_p1_win_rate']):.3f}\")")
    TERM=$( echo "$RAW"  | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['terminal_state_win_pct_a']:.2f}\")")

    echo "  WR=$WR  slot0=$SLOT0 slot1=$SLOT1  seat_delta=$SEAT  term=$TERM  sym_gate=$SYM_OK"

    PROMOTED="false"
    if [ "$SYM_OK" != "True" ]; then
        echo "  🛑 SYMMETRY GATE FAILED — encoding asymmetry detected. No promotion."
        echo "  ⚠ CHECK #1047 (trig/[Y,X] transpose in encode_obs). Fix before submitting."
        RESULT_LINE="$(date '+%Y-%m-%d %H:%M') | sym_FAIL | challenger=U${CHALLENGER_U} WR=${WR} slot0=${SLOT0} slot1=${SLOT1} seat_delta=${SEAT}"
    else
        WR_OK=$("$PYTHON" -c "print('yes' if float('$WR') > 0.50 else 'no')")
        if [ "$WR_OK" = "yes" ]; then
            # Save old champion to pool before overwriting
            POOL_NAME="champion_U${CURRENT_U}_$(date +%Y%m%d_%H%M).pt"
            cp "$CHAMPION" "$POOL_DIR/$POOL_NAME"
            cp "$CHALLENGER" "$CHAMPION"
            PROMOTED="true"
            echo "  *** PROMOTED: $CHALLENGER_LABEL U=$CHALLENGER_U → champion (WR=$WR)"
            echo "  Old champion saved to pool: $POOL_NAME"

            # Push new champion back to all fleet hosts so their OpponentPool can load it
            echo "  Pushing champion to fleet..."
            for H in "${HOSTS[@]}"; do
                rsync -az -e "ssh $SSH_OPTS" "$CHAMPION" \
                    "${USER}@${H}:${REPO_DIR}/agents/rl_ppo/checkpoints/champion.pt" 2>/dev/null &
            done
            wait
            echo "  Champion pushed to ${#HOSTS[@]} hosts"

            RESULT_LINE="$(date '+%Y-%m-%d %H:%M') | PROMOTED U${CHALLENGER_U} | WR=${WR} slot0=${SLOT0} slot1=${SLOT1} term=${TERM}"
        else
            echo "  Challenger didn't clear >0.50 (WR=$WR). Champion retained at U=$CURRENT_U."
            RESULT_LINE="$(date '+%Y-%m-%d %H:%M') | no_promote | champion=U${CURRENT_U} challenger=U${CHALLENGER_U} WR=${WR}"
        fi
    fi
fi

# ── comet_reaper eval on current champion ──────────────────────────────────────
# Only if champion exists and comet_reaper is loadable
COMET_RESULT=""
if [ -f "$CHAMPION" ]; then
    echo ""
    echo "=== comet_reaper eval (current champion) ==="
    COMET_RAW=$("$PYTHON" "${SCRIPT_DIR}/eval_checkpoints.py" \
        --checkpoint-a "$CHAMPION" \
        --vs-comet \
        --n-games 20 \
        --json 2>/dev/null | "$PYTHON" -c "
import sys
data = sys.stdin.read()
start = data.rfind('{')
end   = data.rfind('}') + 1
print(data[start:end] if start >= 0 and end > start else '')
") || true
    if [ -n "$COMET_RAW" ]; then
        COMET_WR=$(echo "$COMET_RAW" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['a_win_rate']:.3f}\")" 2>/dev/null || echo "?")
        COMET_SYM=$(echo "$COMET_RAW" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print('ok' if d['symmetry_hard_gate_passed'] else 'FAIL')" 2>/dev/null || echo "?")
        echo "  Champion vs comet_reaper: WR=$COMET_WR  sym=$COMET_SYM"
        if "$PYTHON" -c "exit(0 if float('$COMET_WR') > 0.50 else 1)" 2>/dev/null; then
            echo "  *** ALERT: Champion beats comet_reaper ($COMET_WR)! Consider submission."
        fi
        COMET_RESULT="vs_comet_WR=${COMET_WR}"
    else
        echo "  comet_reaper eval skipped (not yet loadable via this interface)"
        COMET_RESULT="vs_comet=skipped"
    fi
fi

# ── Write ranking to ORCHESTRATOR_STATE.md ────────────────────────────────────
POOL_SIZE=$(ls "$POOL_DIR"/*.pt 2>/dev/null | wc -l | tr -d ' ')
{
    echo ""
    echo "### Champion eval — $RESULT_LINE  $COMET_RESULT"
    echo "Pool size:        $POOL_SIZE checkpoints"
} >> "$STATE_FILE"

# ── Write league_state.json for dashboard ─────────────────────────────────────
CHAMP_U=$(get_update "$CHAMPION")
CHAMP_WR=$(get_best_wr "$CHAMPION")
LEAGUE_JSON="$(dirname "$CHAMPION")/league_state.json"
CHAMP_VER=$(ls "$POOL_DIR"/*.pt 2>/dev/null | wc -l | tr -d ' ')
IS_PROMOTED="False"; [ "$PROMOTED" = "true" ] && IS_PROMOTED="True"

# Sanitize COMET_WR: must be float or None for valid Python
COMET_WR_PY="None"
if echo "$COMET_WR" | grep -qE '^[0-9.]+$' 2>/dev/null; then
    COMET_WR_PY="$COMET_WR"
fi
# Sanitize WR for promotion event
WR_PY="0.0"
if echo "${WR:-0}" | grep -qE '^[0-9.]+$' 2>/dev/null; then
    WR_PY="${WR:-0.0}"
fi
# Build sync label in shell to avoid strftime issues in heredoc
SYNC_LABEL="$(date '+%Y-%m-%d %H:%M') MT"

"$PYTHON" << PYEOF
import json, time

league_json   = '$LEAGUE_JSON'
champ_u       = $CHAMP_U
champ_wr      = $CHAMP_WR
champ_ver     = $CHAMP_VER
pool_size     = $POOL_SIZE
comet_wr      = $COMET_WR_PY
promoted      = $IS_PROMOTED
chall_label   = '${CHALLENGER_LABEL:-seeded}'
chall_u       = ${CHALLENGER_U:-0}
promo_wr      = $WR_PY
sync_label    = '$SYNC_LABEL'

state = {
    'champion_path':      '$CHAMPION',
    'champion_u':         champ_u,
    'champion_wr_greedy': champ_wr,
    'champion_label':     chall_label,
    'champion_version':   champ_ver,
    'pool_size':          pool_size,
    'vs_comet_wr':        comet_wr,
    'last_promoted_at':   None,
    'last_sync_ts':       int(time.time()),
    'last_sync_label':    sync_label,
    'promoted':           promoted,
    'promotion_event':    '${RESULT_LINE:-no_event}',
    'promotions':         [],
}

# Preserve promotion history
try:
    old = json.load(open(league_json))
    state['promotions'] = old.get('promotions', [])
    if old.get('last_promoted_at') and not promoted:
        state['last_promoted_at'] = old['last_promoted_at']
except:
    pass

if promoted:
    state['last_promoted_at'] = sync_label
    state['promotions'].append({
        'ts': sync_label, 'label': chall_label,
        'u': chall_u, 'wr': promo_wr, 'version': champ_ver,
    })

with open(league_json, 'w') as f:
    json.dump(state, f, indent=2)
print(f'league_state.json written (v{champ_ver}, U={champ_u}, comet={comet_wr})')
PYEOF

# Push league_state.json to all fleet hosts so monitor.sh can show league header
if [ -f "$LEAGUE_JSON" ]; then
    PUSH_OK=0
    for H in "${HOSTS[@]}"; do
        scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$HOME/.ssh/id_rsa" \
            "$LEAGUE_JSON" \
            "${USER}@${H}:${REPO_DIR}/agents/rl_ppo/checkpoints/league_state.json" 2>/dev/null && PUSH_OK=$((PUSH_OK+1)) &
    done
    wait
    echo "  league_state pushed to ${PUSH_OK}/${#HOSTS[@]} hosts"
fi

echo ""
echo "Champion: $CHAMPION (U=$CHAMP_U)"
echo "Pool: $POOL_SIZE archived checkpoints"
