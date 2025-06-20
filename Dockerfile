# Dockerfile
FROM python:3.11-alpine

# Install system dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev python3-dev cargo

# Set working directory
WORKDIR /app

# Copy ELVIS source
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start ELVIS
CMD ["uvicorn", "elvis.main:app", "--host", "0.0.0.0", "--port", "8000"]
