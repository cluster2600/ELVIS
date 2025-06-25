# Multi-stage Dockerfile for ELVIS Trading Bot
FROM python:3.11-slim AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    unzip \
    gcc \
    g++ \
    musl-dev \
    libffi-dev \
    openssl \
    python3-dev \
    cargo \
    pkg-config \
    cmake \
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib with explicit build target
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    ldconfig

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install Python dependencies
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include
RUN pip install --no-cache-dir --upgrade pip \
    && ldconfig \
    && CFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib" pip install --no-cache-dir --no-binary :all: ta-lib \
    && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
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
