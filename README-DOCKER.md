# Kitchen Floor Plan - Docker Setup

This directory contains a Dockerized version of the Kitchen Floor Plan design tool.

## Quick Start

### Using Docker Compose (Recommended)

1. **Start the application:**
   ```bash
   docker-compose up -d
   ```

2. **Open in browser:**
   Navigate to [http://localhost:8000](http://localhost:8000)

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Using Docker Commands

1. **Build the image:**
   ```bash
   docker build -t kitchen-floor-plan .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 -v "$(pwd):/app" --name kitchen-floor-plan kitchen-floor-plan
   ```

3. **Stop the container:**
   ```bash
   docker stop kitchen-floor-plan
   docker rm kitchen-floor-plan
   ```

## Features

- **Port 8000**: Access the web interface at http://localhost:8000
- **Volume Mount**: Current directory is mounted to persist YAML file changes
- **Auto-restart**: Container restarts automatically unless stopped manually

## File Persistence

All YAML files saved through the web interface are stored in the current directory and persist even when the container is stopped or removed.

## Viewing Logs

```bash
docker-compose logs -f
```

Or with Docker:
```bash
docker logs -f kitchen-floor-plan
```

## Rebuilding After Changes

If you modify `index.html` or `server.py`:

```bash
docker-compose down
docker-compose up -d --build
```

## Troubleshooting

**Port already in use:**
- Stop any existing process on port 8000
- Or change the port in `docker-compose.yml` (e.g., `"8080:8000"`)

**Permission issues:**
- Ensure the current directory has write permissions for saving YAML files
