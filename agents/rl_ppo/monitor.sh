#!/usr/bin/env bash
# Compact refreshing status table for one Jetstream instance.
# Called by tmux_fleet.sh:  bash ~/orbit_wars/agents/rl_ppo/monitor.sh LABEL IP
LABEL="${1:-$(hostname -s)}"
IP="${2:-}"
LOGDIR="$HOME/orbit_wars/agents/rl_ppo/runs"
CKPT_DIR="$HOME/orbit_wars/agents/rl_ppo/checkpoints"
LEAGUE_JSON="$CKPT_DIR/league_state.json"

# Set tmux pane title
printf '\033]2;%s\033\\' "$LABEL"

# Wait for training logs to appear (logs live in per-run subdirs: runs/ctrl_job1/ctrl_job1.log)
until find "$LOGDIR" -name "*.log" 2>/dev/null | grep -q .; do
    clear
    printf '\033[1;33m%-12s\033[0m  \033[2m%s\033[0m\n' "$LABEL" "$(date +%H:%M:%S)"
    printf '\033[2m  waiting for logs...\033[0m\n'
    sleep 6
done

cf_color() {
    local n
    n=$(awk "BEGIN{printf \"%.3f\", $1+0}" 2>/dev/null || echo "0.000")
    if awk "BEGIN{exit !($n>=0.05 && $n<=0.30)}" 2>/dev/null; then
        printf '\033[1;32m%s\033[0m' "$n"
    elif awk "BEGIN{exit !($n>=0.01)}" 2>/dev/null; then
        printf '\033[1;33m%s\033[0m' "$n"
    else
        printf '\033[1;31m%s\033[0m' "$n"
    fi
}

ev_color() {
    local n
    n=$(awk "BEGIN{printf \"%.2f\", $1+0}" 2>/dev/null || echo "0.00")
    if awk "BEGIN{exit !($n>=0.1)}" 2>/dev/null; then
        printf '\033[1;32m%s\033[0m' "$n"
    elif awk "BEGIN{exit !($n>=-0.1)}" 2>/dev/null; then
        printf '\033[1;33m%s\033[0m' "$n"
    else
        printf '\033[1;31m%s\033[0m' "$n"
    fi
}

show() {
    clear

    # ── Title ──────────────────────────────────────────────────────────────────
    printf '\033[1;36m%-12s\033[0m  \033[2m%s\033[0m\n' "$LABEL" "$(date +%H:%M:%S)"

    # ── Table header — OPP column shows current training opponent ──────────────
    printf '\033[2m%-4s  %4s  %5s  %5s  %5s  %4s  %3s  %4s  %-8s\033[0m\n' \
        "JOB" "U" "STEPS" "CF" "EV" "ENT" "EP" "SPS" "OPP"
    printf '\033[2m%s\033[0m\n' "─────────────────────────────────────────────────────"

    # Get OPP map + eval data in one Python call
    # OPP: which champion each job last loaded (from opp_log.jsonl)
    # eval: latest greedy WR per job (from metrics.jsonl)
    eval "$( python3 - << 'PYEOF'
import json, os, glob

runs_dir = os.path.expanduser("~/orbit_wars/agents/rl_ppo/runs")
opp_map  = {}
eval_map = {}

# League state for champion version
league_json = os.path.expanduser("~/orbit_wars/agents/rl_ppo/checkpoints/league_state.json")
champ_ver = 0
champ_u   = "?"
try:
    ls = json.load(open(league_json))
    champ_ver = ls.get("champion_version", 0)
    champ_u   = ls.get("champion_u", "?")
except:
    pass

for opp_path in glob.glob(os.path.join(runs_dir, "*/opp_log.jsonl")):
    run = os.path.basename(os.path.dirname(opp_path))
    try:
        lines = [l for l in open(opp_path).read().splitlines() if l.strip()]
        if lines:
            ev = json.loads(lines[-1])
            opp_map[run] = str(ev.get("champion_u", "?"))
    except:
        pass

for mf in glob.glob(os.path.join(runs_dir, "*/metrics.jsonl")):
    run = os.path.basename(os.path.dirname(mf))
    best_wr = None; best_u = 0
    try:
        for line in open(mf):
            d = json.loads(line)
            if "eval_vs_greedy" in d:
                wr = d["eval_vs_greedy"]
                u  = d.get("update", 0)
                cr = d.get("comet_reaper_WR")
                if best_wr is None or wr > best_wr:
                    best_wr = wr; best_u = u
                    eval_map[run] = (wr, u, cr)
    except:
        pass

# Emit as shell assignments
print(f"CHAMP_VER={champ_ver}")
for k, v in opp_map.items():
    safe_k = k.replace("-", "_").replace(".", "_")
    print(f"OPP_{safe_k}={v}")
for k, (wr, u, cr) in eval_map.items():
    safe_k = k.replace("-", "_").replace(".", "_")
    cr_s = f"{cr:.0%}" if cr is not None else ""
    print(f"EVAL_{safe_k}={wr:.0%}@U{u}_{cr_s}")
PYEOF
    )" 2>/dev/null || true

    for f in "$LOGDIR"/*/*.log; do
        [ -f "$f" ] || continue
        job=$(basename "$f" .log | grep -o 'job[0-9]*' | sed 's/job/j/' 2>/dev/null || basename "$f" .log | cut -c1-4)
        line=$(grep '^U' "$f" 2>/dev/null | tail -1)
        if [ -z "$line" ]; then
            printf "%-4s  starting...\n" "${job:-?}"
            continue
        fi
        read -r u s cf ev ent ep sps <<< "$(echo "$line" | awk '
        {
            if($1 == "U")              { u=$2 }
            else if($1 ~ /^U[0-9]+$/) { gsub("U","",$1); u=$1 }
            for(i=1;i<=NF;i++){
                if($i == "S:")          { gsub(",","",$( i+1)); s=$(i+1) }
                if($i ~ /^CF:/)  { split($i,a,":"); cf=a[2] }
                if($i ~ /^EV:/)  { split($i,a,":"); ev=a[2] }
                if($i ~ /^Ent:/) { split($i,a,":"); ent=a[2] }
                if($i ~ /^EP:/)  { split($i,a,":"); ep=a[2] }
                if($i ~ /^SPS:/) { split($i,a,":"); sps=a[2] }
            }
            steps_m = (s+0)/1000000
            printf "%s %.1fM %s %s %s %s %s", u, steps_m, cf, ev, ent, ep, sps
        }')"
        ent_short=$(awk "BEGIN{printf \"%.2f\", ${ent:-0}+0}" 2>/dev/null || echo "?")

        # OPP label: look up which champion this job last loaded
        run_name=$(basename "$f" .log)
        safe_run=$(echo "$run_name" | tr '-.' '_')
        opp_champ_u_var="OPP_${safe_run}"
        opp_champ_u="${!opp_champ_u_var}"
        if [ -n "$opp_champ_u" ]; then
            OPP_LABEL="v${CHAMP_VER:-0}@${opp_champ_u}"
        elif [ "${u:-0}" -lt 300 ] 2>/dev/null; then
            OPP_LABEL="v${CHAMP_VER:-0}^"  # startup load, not yet reloaded
        else
            OPP_LABEL="v${CHAMP_VER:-0}?"
        fi

        printf "%-4s  %4s  %5s  " "${job:-?}" "U${u:-?}" "${s:-0.0M}"
        cf_color "${cf:-0}"
        printf "  "
        ev_color "${ev:-0}"
        printf "  %4s  %3s  %4s  \033[35m%-8s\033[0m\n" \
            "$ent_short" "${ep:-?}" "${sps:-?}" "$OPP_LABEL"
    done

    # ── Evals ──────────────────────────────────────────────────────────────────
    EVAL_LINES=""
    for run_dir in "$LOGDIR"/*/; do
        [ -d "$run_dir" ] || continue
        mf="$run_dir/metrics.jsonl"
        [ -f "$mf" ] || continue
        run_name=$(basename "$run_dir")
        job_num=$(echo "$run_name" | grep -o 'job[0-9]*' | sed 's/job//' 2>/dev/null)
        [ -z "$job_num" ] && continue
        # Get best eval from this run
        ev_data=$(python3 -c "
import json
best = None; best_u = 0
for line in open('$mf'):
    try:
        d = json.loads(line)
        if 'eval_vs_greedy' in d:
            wr = d['eval_vs_greedy']; u = d.get('update', 0)
            if best is None or wr > best: best=wr; best_u=u
    except: pass
if best is not None:
    print(f'j$job_num U={best_u}: \033[1;35m{best:.0%}\033[0m')
" 2>/dev/null)
        [ -n "$ev_data" ] && EVAL_LINES="$EVAL_LINES$ev_data"$'\n'
    done

    if [ -n "$EVAL_LINES" ]; then
        printf '\033[2m─────────────────────────────────────────────────────\033[0m\n'
        printf '\033[2mEvals (best per job):\033[0m\n'
        echo "$EVAL_LINES" | sort -t'j' -k2 -n 2>/dev/null | while IFS= read -r el; do
            [ -n "$el" ] && printf '  %b\n' "$el"
        done
    fi

    # ── League status (bottom — always visible after table) ────────────────────
    printf '\033[2m─────────────────────────────────────────────────────\033[0m\n'
    if [ -f "$LEAGUE_JSON" ]; then
        python3 - << 'PYEOF'
import json
try:
    ls = json.load(open("/home/exouser/orbit_wars/agents/rl_ppo/checkpoints/league_state.json"))
    ver   = ls.get("champion_version", 0)
    u     = ls.get("champion_u", "?")
    wr    = ls.get("champion_wr_greedy")
    cr_bot_wr = ls.get("comet_reaper_WR")
    prom  = ls.get("last_promoted_at") or "never"
    pool  = ls.get("pool_size", 0)
    wr_s  = f"{wr:.0%}" if wr is not None else "?"
    cc    = "\033[1;32m" if (cr_bot_wr or 0) > 0 else "\033[0;33m"
    cr_s  = f"{cr_bot_wr:.0%}" if cr_bot_wr is not None else "?"
    promos = ls.get("promotions", [])
    last_p = promos[-1] if promos else None
    prom_s = ""
    if last_p:
        prom_s = f" | last: {last_p.get('ts','?')[:16]} → v{last_p.get('version','?')}"
    print(f"\033[1;35m🏆 champ-v{ver}\033[0m U={u} WR={wr_s} vs_cr_bot:{cc}{cr_s}\033[0m pool:{pool}{prom_s}")
except Exception as e:
    print(f"\033[2m[league: {e}]\033[0m")
PYEOF
    else
        printf '\033[2m[league_state.json not yet pushed — run sync]\033[0m\n'
    fi
}

while true; do
    show 2>/dev/null
    sleep 8
done
