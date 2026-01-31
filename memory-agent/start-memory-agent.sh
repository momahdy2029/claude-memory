#!/bin/bash
# Start Memory Agent in background
# Uses relative path detection - works on any system after cloning

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Load environment if .env exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Start the server
nohup python main.py > memory-agent.log 2>&1 &
echo "Memory Agent started (PID: $!)"
echo "Log file: $SCRIPT_DIR/memory-agent.log"
echo "Dashboard: http://localhost:${PORT:-8102}/dashboard"
