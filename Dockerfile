# Dockerfile
FROM python:3.11-slim

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    wget -O config.guess 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' && \
    wget -O config.sub 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    ldconfig

# Set working directory
WORKDIR /app

# Copy ELVIS source
COPY . /app

# Install Python dependencies
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include
RUN pip install --no-cache-dir --upgrade pip \
    && CFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib" pip install --no-cache-dir --no-binary :all: ta-lib \
    && pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start ELVIS
CMD ["python", "main.py"]
