#!/usr/bin/env bash
# Refreshing status display for one Jetstream instance — v9 ET layout.
# Usage: bash ~/orbit_wars/agents/rl_ppo_cpu/monitor.sh LABEL IP
LABEL="${1:-$(hostname -s)}"
IP="${2:-}"
LOGDIR="$HOME/orbit_wars/agents/rl_ppo_cpu/runs"
CKPT_DIR="$HOME/orbit_wars/agents/rl_ppo_cpu/checkpoints"
LEAGUE_JSON="$CKPT_DIR/league_state.json"

printf '\033]2;%s\033\\' "$LABEL"

until find "$LOGDIR" -name "*.log" 2>/dev/null | grep -q .; do
    clear
    printf '\033[1;33m%-12s\033[0m  \033[2m%s\033[0m\n' "$LABEL" "$(date +%H:%M:%S)"
    printf '\033[2m  waiting for logs...\033[0m\n'
    sleep 6
done

# EV: green >= 0.90 (ET diagnostic pass), yellow >= 0.85, red below
ev_color() {
    local n
    n=$(awk "BEGIN{printf \"%.3f\", $1+0}" 2>/dev/null || echo "0.000")
    if awk "BEGIN{exit !($n>=0.90)}" 2>/dev/null; then
        printf '\033[1;32m%s\033[0m' "$n"
    elif awk "BEGIN{exit !($n>=0.85)}" 2>/dev/null; then
        printf '\033[1;33m%s\033[0m' "$n"
    else
        printf '\033[1;31m%s\033[0m' "$n"
    fi
}

# ENT: green >= 2.0, yellow >= 1.5, red below (collapse warning)
ent_color() {
    local n
    n=$(awk "BEGIN{printf \"%.2f\", $1+0}" 2>/dev/null || echo "0.00")
    if awk "BEGIN{exit !($n>=2.0)}" 2>/dev/null; then
        printf '\033[1;32m%s\033[0m' "$n"
    elif awk "BEGIN{exit !($n>=1.5)}" 2>/dev/null; then
        printf '\033[1;33m%s\033[0m' "$n"
    else
        printf '\033[1;31m%s!\033[0m' "$n"
    fi
}

show() {
    clear
    printf '\033[1;36m%-12s\033[0m  \033[2m%s\033[0m\n' "$LABEL" "$(date +%H:%M:%S)"

    # Pull eval data + champion info in one Python call
    eval "$( python3 - << 'PYEOF'
import json, os, glob

runs_dir = os.path.expanduser("~/orbit_wars/agents/rl_ppo_cpu/runs")
league_json = os.path.expanduser("~/orbit_wars/agents/rl_ppo_cpu/checkpoints/league_state.json")

champ_ver, champ_u, champ_wr, champ_cr = 0, "?", None, None
try:
    ls = json.load(open(league_json))
    champ_ver = ls.get("champion_version", 0)
    champ_u   = ls.get("champion_u", "?")
    champ_wr  = ls.get("champion_wr_greedy")
    champ_cr  = ls.get("comet_reaper_WR")
    promos    = ls.get("promotions", [])
    last_p    = promos[-1] if promos else None
    prom_ts   = last_p.get("ts","?")[:16] if last_p else "never"
    prom_ver  = last_p.get("version","?") if last_p else "?"
    print(f"CHAMP_VER={champ_ver}")
    print(f"CHAMP_U={champ_u}")
    wr_s = f"{champ_wr:.0%}" if champ_wr is not None else "?"
    cr_s = f"{champ_cr:.0%}" if champ_cr is not None else "?"
    print(f"CHAMP_WR={wr_s}")
    print(f"CHAMP_CR={cr_s}")
    print(f"CHAMP_POOL={ls.get('pool_size',0)}")
    print(f"PROM_TS={prom_ts}")
    print(f"PROM_VER={prom_ver}")
except:
    print("CHAMP_VER=0"); print("CHAMP_U=?"); print("CHAMP_WR=?")
    print("CHAMP_CR=?"); print("CHAMP_POOL=0"); print("PROM_TS=never"); print("PROM_VER=?")

# Latest eval per job: greedy_WR | comet_reaper_WR | update
for mf in glob.glob(os.path.join(runs_dir, "*/metrics.jsonl")):
    run = os.path.basename(os.path.dirname(mf))
    try:
        latest_eval = None
        for line in open(mf):
            d = json.loads(line)
            if "eval_vs_greedy" in d:
                latest_eval = d
        if latest_eval:
            wr = latest_eval["eval_vs_greedy"]
            u  = latest_eval.get("update", 0)
            cr = latest_eval.get("comet_reaper_WR")
            safe = run.replace("-","_").replace(".","_")
            wr_s = f"{wr:.0%}"
            cr_s = f"{cr:.0%}" if cr is not None else "?"
            print(f"EVAL_{safe}='{wr_s}|{cr_s}|U{u}'")
    except:
        pass
PYEOF
    )" 2>/dev/null || true

    printf '\n'
    printf '\033[2m%-4s  %4s  %5s  %4s  %9s  %5s  %4s\033[0m\n' \
        "JOB" "U" "EV" "ENT" "greedy_WR" "cr_WR" "SPS"
    printf '\033[2m%s\033[0m\n' "────────────────────────────────────────────"

    for f in "$LOGDIR"/*/*.log; do
        [ -f "$f" ] || continue
        job=$(basename "$f" .log | grep -o 'job[0-9]*' | sed 's/job/j/' 2>/dev/null || basename "$f" .log | cut -c1-4)
        line=$(grep '^U' "$f" 2>/dev/null | tail -1)
        if [ -z "$line" ]; then
            printf "%-4s  \033[2mstarting...\033[0m\n" "${job:-?}"
            continue
        fi

        read -r u ev ent sps <<< "$(echo "$line" | awk '
        {
            for(i=1;i<=NF;i++){
                if($i == "U")          { u=$(i+1) }
                if($i ~ /^U[0-9]+$/)  { gsub("U","",$i); u=$i }
                if($i ~ /^EV:/)  { split($i,a,":"); ev=a[2] }
                if($i ~ /^Ent:/) { split($i,a,":"); ent=a[2] }
                if($i ~ /^SPS:/) { split($i,a,":"); sps=a[2] }
            }
            printf "%s %s %s %s", u, ev, ent, sps
        }')"

        run_name=$(basename "$f" .log)
        safe_run=$(echo "$run_name" | tr -- '-.' '_')
        eval_var="EVAL_${safe_run}"
        eval_val="${!eval_var}"
        if [ -n "$eval_val" ]; then
            IFS='|' read -r greedy_wr cr_wr eval_u <<< "$eval_val"
        else
            greedy_wr="-"; cr_wr="-"
        fi

        ent_short=$(awk "BEGIN{printf \"%.1f\", ${ent:-0}+0}" 2>/dev/null || echo "?")

        printf "%-4s  %4s  " "${job:-?}" "U${u:-?}"
        ev_color "${ev:-0}"
        printf "  %4s  " "$ent_short"
        # greedy WR
        if [ "$greedy_wr" = "-" ] || [ "$greedy_wr" = "?" ]; then
            printf '\033[2m%9s\033[0m' "$greedy_wr"
        else
            printf '\033[1;35m%9s\033[0m' "$greedy_wr"
        fi
        printf "  "
        # cr_WR — bold green if non-zero, dim if zero/unknown
        if [ "$cr_wr" = "-" ] || [ "$cr_wr" = "?" ] || [ "$cr_wr" = "0%" ]; then
            printf '\033[2m%5s\033[0m' "${cr_wr:--}"
        else
            printf '\033[1;32m%5s\033[0m' "$cr_wr"
        fi
        printf "  %4s\n" "${sps:-?}"
    done

    printf '\033[2m────────────────────────────────────────────\033[0m\n'
    if [ "${CHAMP_CR:-?}" != "?" ] && [ "${CHAMP_CR:-?}" != "0%" ]; then
        CR_COLOR='\033[1;32m'
    else
        CR_COLOR='\033[0;33m'
    fi
    printf "\033[1;35m🏆 v${CHAMP_VER:-0}\033[0m U=${CHAMP_U:-?} g=${CHAMP_WR:-?} ${CR_COLOR}cr=${CHAMP_CR:-?}\033[0m p:${CHAMP_POOL:-0}\n"
    printf "\033[2m   promoted: ${PROM_TS:-never} -> v${PROM_VER:-?}\033[0m\n"
}

while true; do
    show 2>/dev/null
    sleep 8
done
