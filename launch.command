#!/bin/bash

# Kitchen Floor Plan Viewer Launcher
cd "$(dirname "$0")"

echo "Starting Kitchen Floor Plan Server..."
echo "Opening http://localhost:8000 in your browser..."
echo ""
echo "Press Ctrl+C to stop the server when done."
echo ""

# Open browser after a short delay (give server time to start)
(sleep 1 && open "http://localhost:8000") &

# Start the custom server (supports saving)
python3 server.py
