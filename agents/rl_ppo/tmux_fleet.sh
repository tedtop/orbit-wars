#!/usr/bin/env bash
# Orbit Wars fleet monitor — 9 panes (3×3 grid), one per Jetstream instance.
# v7 fleet: 9× m3.xl (homogeneous for clean A/B)
# Usage:    bash agents/rl_ppo/tmux_fleet.sh
# Reattach: tmux attach -t orbit_fleet
# Detach:   Ctrl-b d  |  Zoom: Ctrl-b z  |  Navigate: Ctrl-b arrow

SESSION="orbit_fleet"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=20"

# label:IP  (no spaces around colon)
# All m3.xl — 9 of 9 instances ready
REMOTES=(
    "ppo-1:149.165.175.228"
    "ppo-2:149.165.175.188"
    "ppo-3:149.165.172.107"
    "ppo-4:149.165.159.11"
    "ppo-5:149.165.174.192"
    "ppo-6:149.165.169.105"
    "ppo-7:149.165.174.44"
    "ppo-8:149.165.150.186"
    "ppo-9:149.165.169.255"
)

# Build the per-pane command: SSH with auto-reconnect so the pane never dies
pane_cmd() {
    local LABEL="$1" IP="$2"
    # Retry loop keeps the pane alive if SSH drops
    printf "while true; do ssh -tt %s exouser@%s 'bash ~/orbit_wars/agents/rl_ppo/monitor.sh %s %s'; echo '[%s] disconnected — retrying in 8s...'; sleep 8; done" \
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
tmux set -t "$SESSION" status-right    "#[fg=#f9e2af]orbit_fleet  #[fg=#a6e3a1]%H:%M"
tmux set -t "$SESSION" status-interval 15

tmux select-pane -t "$SESSION:fleet.0"

echo "=== orbit_fleet: 9 panes 3×3 (9× m3.xl homogeneous — v7 A/B fleet) ==="
echo "  Ctrl-b d  detach    Ctrl-b z  zoom pane"
echo "  Ctrl-b o  next      Ctrl-b ;  last active"
echo ""
exec tmux attach -t "$SESSION"
