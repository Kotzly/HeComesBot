FROM python:3.11-slim

# System dependencies: ffmpeg for video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source to leverage layer caching
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[instagram]"

# Copy the web UI (Flask app + static assets)
COPY web/ ./web/

# Expose the Flask web UI port
EXPOSE 5000

# Default: launch the web UI. Override CMD to run a CLI instead, e.g.:
#   docker run hecomes hecomes-image --help
#   docker run hecomes hecomes-video -n 1 -d 5
#   docker run hecomes hecomes-instagram --type image --caption "hello"
CMD ["python", "web/app.py"]
