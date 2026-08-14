# Changelog

Release history is curated from tagged source. Older development diaries and
completion claims remain available through Git tags, especially `v0.3.0`; they
are not operational documentation.

## v2.0.0-alpha.2 — 2026-08-14

- Preserve the `v2.0.0-alpha.1` tag from the failed release attempt and publish
  the correction under a new immutable prerelease identity.
- Remove each platform-specific local image reference after its anonymous
  multi-architecture smoke, including best-effort cleanup on failure, so the
  next platform pull cannot conflict with the previous digest reference.
- Gate a run-unique candidate digest before compare-or-create promotion of both
  immutable commit and prerelease installation tags.
- Keep the Python 3.14-only operator scope and `ACTIVE` **NO-GO** boundary
  unchanged.

## v2.0.0-alpha.1 — 2026-08-14

- Package the offline V2 PostgreSQL bootstrap, cut-over preflight, bounded raw
  snapshot import, and reconciliation tools for Python 3.14.
- Publish only the dedicated V2 operator image from the tag workflow; CI keeps
  the compatibility paper image as a build-only regression check.
- Keep every operator result non-activating; `ACTIVE` remains a **NO-GO**.
- Retire the unsupported older-interpreter trainer boundary.
- Remove unverified Apple-container and Ansible deployment helpers.
- Remove 50 ambient-network/database V1 diagnostics that were named as pytest
  tests but provided no hermetic release contract.
- Replace duplicated V1 reports with a compact tagged restore manifest.
- Condense migration status around current gates instead of historical command
  transcripts.

## v0.3.0 — 2026-07-12

- Reorganised the repository and documented the compatibility runtime.
- Added profitability-roadmap wiring, checkpoint lifecycle work, optional model
  guards, and a split ML experiment container.
- This tag is the forensic restore point for retired V1 documentation and
  deployment experiments.

## v0.2.0 — historical release

- Intermediate tagged release retained in GitHub Releases and Git history.

## v0.1.0 — 2026-07-02

- First tagged release and GHCR image.
- Migrated the main image to Python 3.14 and hardened paper-runtime startup.

For exact changes and authorship, use `git log`, compare the release tags, or
open the corresponding GitHub release. No changelog entry is evidence of a
deployed or activated V2 runtime.
