FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1

LABEL org.opencontainers.image.title="ELVIS V2 operator preview" \
      org.opencontainers.image.description="Paper-migration operator; ACTIVE remains NO-GO" \
      org.opencontainers.image.licenses="BTC_BOT" \
      org.elvis.v2.scope="paper-migration-preview" \
      org.elvis.v2.active="NO-GO"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN mkdir -p \
        trading/application \
        trading/domain \
        trading/persistence \
        trading/persistence/sql_migrations \
        scripts \
    && touch trading/application/__init__.py trading/domain/__init__.py
COPY trading/__init__.py ./trading/
COPY trading/application/fresh_target_cutover.py \
     trading/application/fresh_opening.py \
     trading/application/legacy_snapshot_import.py \
     trading/application/legacy_snapshot_reconciliation.py \
     trading/application/paper_account_readiness.py \
     ./trading/application/
COPY trading/domain/_decimal.py \
     trading/domain/_validation.py \
     trading/domain/order_lifecycle.py \
     trading/domain/orders.py \
     trading/domain/paper_accounting.py \
     trading/domain/paper_economics.py \
     trading/domain/paper_settlement.py \
     trading/domain/positions.py \
     ./trading/domain/
COPY trading/persistence/__init__.py \
     trading/persistence/journal_codec.py \
     trading/persistence/migration_runner.py \
     trading/persistence/order_position_journal.py \
     trading/persistence/paper_account_journal.py \
     trading/persistence/paper_account_journal_codec.py \
     trading/persistence/paper_account_readiness.py \
     trading/persistence/postgres_bootstrap.py \
     trading/persistence/postgres_cutover_preflight.py \
     trading/persistence/postgres_legacy_snapshot_import.py \
     trading/persistence/postgres_legacy_snapshot_reconciliation.py \
     ./trading/persistence/
COPY trading/persistence/sql_migrations/*.sql \
     trading/persistence/sql_migrations/__init__.py \
     ./trading/persistence/sql_migrations/
COPY scripts/__init__.py \
     scripts/v2_operator.py \
     scripts/v2_opening_plan.py \
     scripts/postgres_bootstrap.py \
     scripts/postgres_cutover_preflight.py \
     scripts/postgres_legacy_snapshot_import.py \
     scripts/postgres_legacy_snapshot_reconciliation.py \
     ./scripts/
COPY deploy/v2/requirements.operator.txt ./requirements.operator.txt
COPY LICENSE /licenses/ELVIS-LICENSE

RUN python -m pip install \
        --no-cache-dir \
        --only-binary=:all: \
        --require-hashes \
        -r requirements.operator.txt \
    && python -m pip check

USER 65532:65532

ENTRYPOINT ["python", "-m", "scripts.v2_operator"]
CMD ["--help"]
