# Multi-stage Dockerfile for ELVIS Trading Bot
FROM python:3.14-slim AS builder

# buildx sets TARGETARCH to amd64/arm64 - used to pick the right TA-Lib package
ARG TARGETARCH

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gcc \
    g++ \
    libffi-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install the TA-Lib C library from the official prebuilt per-arch package
# (0.6.x; avoids the ancient 0.4.0 autotools that can't detect arm64).
RUN wget -q "https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_${TARGETARCH}.deb" \
    && dpkg -i "ta-lib_0.6.4_${TARGETARCH}.deb" \
    && rm "ta-lib_0.6.4_${TARGETARCH}.deb" \
    && ldconfig

WORKDIR /app

# Copy source before dependency install because requirements.txt uses -e .
COPY . /app

# Install Python dependencies
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include
# Install Python dependencies (prefer wheels to reduce build time/RAM)
# NOTE: building some scientific packages from source on arm64 can OOM; prefer binaries.
RUN pip install --no-cache-dir --upgrade pip \
    && ldconfig \
    && CFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib" pip install --no-cache-dir --prefer-binary ta-lib \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Production stage
FROM python:3.14-slim AS production

ARG TARGETARCH

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    wget \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install the TA-Lib C runtime (same package as the builder) so the talib
# Python wheel can load its shared library at runtime.
RUN wget -q "https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_${TARGETARCH}.deb" \
    && dpkg -i "ta-lib_0.6.4_${TARGETARCH}.deb" \
    && rm "ta-lib_0.6.4_${TARGETARCH}.deb" \
    && ldconfig

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy ELVIS source code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include

# Expose ports
EXPOSE 5050 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5050/health || exit 1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash elvis
RUN chown -R elvis:elvis /app
USER elvis

# Start ELVIS with proper signal handling
CMD ["python", "main.py", "--mode", "paper", "--log-level", "INFO"]
