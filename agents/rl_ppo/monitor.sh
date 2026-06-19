#!/usr/bin/env bash
# Compact refreshing status table for one Jetstream instance.
# Called by tmux_fleet.sh:  bash ~/orbit_wars/agents/rl_ppo/monitor.sh LABEL IP
LABEL="${1:-$(hostname -s)}"
IP="${2:-}"
LOGDIR="$HOME/orbit_wars/agents/rl_ppo/runs"

# Set tmux pane title
printf '\033]2;%s\033\\' "$LABEL"

# Wait for training logs to appear
until ls "$LOGDIR"/*.log 2>/dev/null | grep -q .; do
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
    # Title: label + time only (no IP — saves 18 chars, panes are narrow)
    printf '\033[1;36m%-12s\033[0m  \033[2m%s\033[0m\n' "$LABEL" "$(date +%H:%M:%S)"
    printf '\033[2m%-4s  %4s  %5s  %5s  %5s  %4s  %3s  %4s\033[0m\n' \
        "JOB" "U" "STEPS" "CF" "EV" "ENT" "EP" "SPS"
    printf '\033[2m%s\033[0m\n' "───────────────────────────────────────────────"

    for f in "$LOGDIR"/*.log; do
        [ -f "$f" ] || continue
        job=$(basename "$f" .log | grep -o 'job[0-9]*' | sed 's/job/j/' || basename "$f" .log | cut -c1-4)
        line=$(grep '^U' "$f" 2>/dev/null | tail -1)
        if [ -z "$line" ]; then
            printf "%-4s  starting...\n" "${job:-?}"
            continue
        fi
        # Parse: "U   20 | S:    81,920 | L:... | CF:... | EV:... | Ent:... | EP:... | SPS:..."
        read -r u s cf ev ent ep sps <<< "$(echo "$line" | awk '
        {
            if($1 == "U")              { u=$2 }          # "U   20 |" space-padded
            else if($1 ~ /^U[0-9]+$/) { gsub("U","",$1); u=$1 }  # "U0020 |" zero-padded
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
        printf "%-4s  %4s  %5s  " "${job:-?}" "U${u:-?}" "${s:-0.0M}"
        cf_color "${cf:-0}"
        printf "  "
        ev_color "${ev:-0}"
        printf "  %4s  %3s  %4s\n" "$ent_short" "${ep:-?}" "${sps:-?}"
    done

    eval_line=$(grep 'Eval vs greedy' "$LOGDIR"/*.log 2>/dev/null | tail -1 | grep -o '[0-9]*%')
    [ -n "$eval_line" ] && printf '\033[2m─────────────────────────────────────────\033[0m\n' && \
        printf 'eval → \033[1;35m%s\033[0m vs greedy\n' "$eval_line"
}

while true; do
    show 2>/dev/null
    sleep 8
done
