FROM python:3.11-slim

WORKDIR /app

# Copy application files
COPY index.html .
COPY kitchen.yaml .
COPY server.py .

# Expose the port the server runs on
EXPOSE 8000

# Run the server
CMD ["python3", "server.py"]
