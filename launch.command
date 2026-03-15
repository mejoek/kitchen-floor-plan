#!/bin/bash

# Kitchen Floor Plan Viewer Launcher (Docker)
cd "$(dirname "$0")"

echo "Starting Kitchen Floor Plan Server (Docker)..."

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
  echo ""
  echo "ERROR: Docker is not running. Please start Docker Desktop and try again."
  echo ""
  read -n 1 -s -r -p "Press any key to exit..."
  exit 1
fi

echo "Opening http://localhost:8000 in your browser..."
echo ""
echo "Press Ctrl+C to stop the server when done."
echo ""

# Open browser after a short delay (give server time to start)
(sleep 2 && open "http://localhost:8000") &

# Start the Docker container
docker compose up

# Cleanup when stopped
echo ""
echo "Stopping container..."
docker compose down
