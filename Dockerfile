# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (fail fast, minimal deps, and clean up in same layer)
RUN set -eux; \
    apt-get update -qq && apt-get install -y --no-install-recommends \
        ffmpeg \
        imagemagick \
        fontconfig \
        fonts-dejavu \
        fonts-dejavu-core \
        fonts-liberation \
        wget \
        curl; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Configure ImageMagick policy to allow PDF/video operations and reading text via @file
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true
RUN sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@*"/' /etc/ImageMagick-6/policy.xml || true
# Also patch ImageMagick 7 policy if present
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-7/policy.xml || true
RUN sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@*"/' /etc/ImageMagick-7/policy.xml || true
# Refresh font cache to ensure newly installed fonts are discoverable
RUN fc-cache -f -v || true

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY create_video.py .

# Create directories for input/output
RUN mkdir -p /app/clips /app/output

# Set the ImageMagick binary path for MoviePy (Linux path)
ENV IMAGEMAGICK_BINARY=/usr/bin/convert

# Default command
CMD ["python", "create_video.py", "--help"]