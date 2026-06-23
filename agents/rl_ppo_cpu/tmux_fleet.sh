#!/usr/bin/env bash
# Orbit Wars fleet monitor — v9 ET, 9-instance fleet.
# Usage:    bash agents/rl_ppo_cpu/tmux_fleet.sh
# Reattach: tmux attach -t jetstream2_fleet
# Detach:   Ctrl-b d  |  Zoom: Ctrl-b z  |  Navigate: Ctrl-b arrow

SESSION="jetstream2_fleet"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=20"

# label:IP — fill in IPs from Jetstream2 before running
REMOTES=(
    "ppo-1:149.165.169.195"
    "ppo-2:149.165.173.145"
    "ppo-3:149.165.174.28"
    "ppo-4:149.165.175.244"
    "ppo-5:149.165.174.123"
    "ppo-6:149.165.154.56"
    "ppo-7:149.165.155.119"
    "ppo-8:149.165.150.220"
    "ppo-9:149.165.171.224"
)

# Build the per-pane command: SSH with auto-reconnect so the pane never dies
pane_cmd() {
    local LABEL="$1" IP="$2"
    # Retry loop keeps the pane alive if SSH drops
    printf "while true; do ssh -tt %s exouser@%s 'bash ~/orbit_wars/agents/rl_ppo_cpu/monitor.sh %s %s'; echo '[%s] disconnected — retrying in 8s...'; sleep 8; done" \
        "$SSH_OPTS" "$IP" "$LABEL" "$IP" "$LABEL"
}

tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Pane 0: first remote (creates the session with explicit large size) ───────
FIRST="${REMOTES[0]}"
LABEL="${FIRST%%:*}"
IP="${FIRST##*:}"
tmux new-session -d -s "$SESSION" -n "fleet" -x 220 -y 60 "$(pane_cmd "$LABEL" "$IP")"

# ── Panes 1-7: remaining remotes, tile after each split ──────────────────────
for ENTRY in "${REMOTES[@]:1}"; do
    LABEL="${ENTRY%%:*}"
    IP="${ENTRY##*:}"
    tmux split-window -h -t "$SESSION:fleet" "$(pane_cmd "$LABEL" "$IP")"
    tmux select-layout -t "$SESSION:fleet" tiled   # keep panes large enough to split again
done

# ── Final layout ──────────────────────────────────────────────────────────────
tmux select-layout -t "$SESSION:fleet" tiled

# Pane titles in top border
tmux set -t "$SESSION" pane-border-status    top
tmux set -t "$SESSION" pane-border-format    " #{pane_title} "
tmux set -t "$SESSION" pane-border-style     "fg=#45475a"
tmux set -t "$SESSION" pane-active-border-style "fg=#f9e2af"

# Status bar
tmux set -t "$SESSION" status-style    "bg=#1e1e2e,fg=#cdd6f4"
tmux set -t "$SESSION" status-right    "#[fg=#f9e2af]jetstream2_fleet  #[fg=#a6e3a1]%H:%M"
tmux set -t "$SESSION" status-interval 15

tmux select-pane -t "$SESSION:fleet.0"

echo "=== jetstream2_fleet: ${#REMOTES[@]} active panes ==="
echo "  Ctrl-b d  detach    Ctrl-b z  zoom pane"
echo "  Ctrl-b o  next      Ctrl-b ;  last active"
echo ""
exec tmux attach -t "$SESSION"
