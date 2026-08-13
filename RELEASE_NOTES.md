# ELVIS V2 operator preview — v2.0.0-alpha.1

This is the first installable **paper/migration preview** of the ELVIS V2
operator tooling. It is intentionally a GitHub prerelease and is not promoted
to a floating `latest`, `2`, or `2.0` container tag.

> **ACTIVE: NO-GO.** No V2 trading runtime is included or activated. This
> release does not authorize live trading, production cut-over, or use against
> an unstopped legacy source.

## Included

- public, anonymously pullable Python 3.14 operator image for Linux `amd64`
  and `arm64`
- four source-dispatched commands: `bootstrap`, `cutover-preflight`,
  `import-snapshot`, and `reconcile-snapshot`
- hardened Compose configuration for read-only operator inputs
- bounded example inputs for fresh-target paper migration work
- a clean-directory installation and verification guide

The image is published only as:

- `ghcr.io/cluster2600/elvis-v2-operator:2.0.0-alpha.1`
- `ghcr.io/cluster2600/elvis-v2-operator:sha-<full-release-commit>`

The release asset `IMAGE_DIGEST.txt` provides the preferred immutable manifest
reference.

## Supply-chain evidence

The release workflow verifies that the exact tag is attached to the current
`origin/main` commit and that the project version is `2.0.0a1` before building.
It then performs per-architecture command-import smokes and a clean-directory
Compose config/pull/help smoke before creating the release.

Release assets include SHA-256 checksums, platform-specific SPDX JSON image
SBOMs for Linux `amd64` and `arm64`, and GitHub build-provenance attestations.
The container manifest also carries BuildKit SBOM and provenance attestations.

## Not included

- no Python wheel or PyPI publication
- no V2 trading-runtime launcher
- no automatic database cut-over or activation
- no live-exchange execution path
- no compatibility promise for V1 deployment scripts

Follow the [versioned installation guide](https://github.com/cluster2600/ELVIS/blob/v2.0.0-alpha.1/INSTALL_V2.md)
exactly. Use stopped clones and disposable
targets, retain every JSON receipt, and keep **ACTIVE: NO-GO** until a separate
review explicitly changes that state.
