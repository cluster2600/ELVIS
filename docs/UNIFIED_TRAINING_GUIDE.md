# Python 3.14 paper-training guide

> **Compatibility paper-training only.** Training does not deploy a model,
> change runtime authority, or authorise live execution. `ACTIVE` remains a
> **NO-GO**.

The supported entry point is a thin Python 3.14 wrapper around
`training.train_models`:

```bash
python3.14 -m pip install -e '.[ml]'
./scripts/run_training.sh --help
```

The wrapper performs no package installation, secret migration, Vault startup,
database fallback, or runtime launch. It forwards arguments exactly and returns
the training process exit status.

## Inputs and outputs

The canonical CLI accepts:

- `--config`, default `training/config/model_config.yaml`;
- `--data`, default `data/processed/training_data.csv`;
- `--output`, default `models`;
- `--resume`, an explicit checkpoint path or `latest`/`best`;
- `--debug`; and
- `--local_rank` for an explicitly configured distributed run.

Inspect `./scripts/run_training.sh --help` on the checked-out version before a
run. Use reviewed, causal paper data and write artifacts to an operator-owned
directory. Do not use production exchange credentials.

Example:

```bash
./scripts/run_training.sh \
  --config training/config/model_config.yaml \
  --data data/processed/training_data.csv \
  --output models/paper-review
```

The optional ML container uses the same entry point:

```bash
docker compose --profile ml run --rm elvis-ml-trainer --help
```

The container exchanges only model artifacts through the `models` volume. The
root Compose file remains compatibility evidence, not the V2 installer or a
production deployment.

## Verification boundary

Before retaining an artifact, record the source-data provenance, configuration
hash, code commit, Python/package environment, metrics, and validation split.
Model files alone are not activation evidence. The V2 operator preview does not
include this trainer and cannot load or activate its output.
