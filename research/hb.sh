#!/bin/bash
# Heartbeat: one line per checkpoint. Usage: ./hb.sh <agent-name> "<short note>"
# Only a live model can call this, so the LAST line marks when requests stopped succeeding.
printf '%s | %-12s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${1:-unknown}" "${2:-checkpoint}" \
  >> "$(dirname "$0")/HEARTBEAT.log"
