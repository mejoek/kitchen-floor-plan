#!/bin/bash

# Kitchen Floor Plan Viewer Launcher (Docker)
cd "$(dirname "$0")"

echo "Starting Kitchen Floor Plan Server (Docker)..."
echo "Opening http://localhost:8000 in your browser..."
echo ""
echo "Press Ctrl+C to stop the server when done."
echo ""

# Open browser after a short delay (give server time to start)
(sleep 2 && open "http://localhost:8000") &

# Start the Docker container
docker-compose up

# Cleanup when stopped
echo ""
echo "Stopping container..."
docker-compose down
