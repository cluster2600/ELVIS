FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY trading ./trading
COPY scripts ./scripts
RUN pip install --no-cache-dir "psycopg2-binary==2.9.12"

ENTRYPOINT ["python", "-m", "scripts.postgres_bootstrap"]
