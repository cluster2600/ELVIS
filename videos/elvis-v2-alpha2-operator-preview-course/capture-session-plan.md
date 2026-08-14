# Capture session plan — SS02 to SS14

Status: capture protocol only. It does not authorise a final render,
publication, PostgreSQL operation, runtime start, or authority transition.

## Safety boundary

Use a dedicated demonstration shell with a neutral prompt and a 2560x1440
canvas. The release tag, source commit, public image digest and public
checksums may remain visible. Hide or exclude every user name, host name, IP or
MAC address, home path, registry credential, notification, shell history,
clipboard, `.env`, DSN, `pgpass`, certificate, key, private network and
unrelated container.

The operator input directory must be empty. Do not create or mount a service
file, CA, password file or JSON input. Do not execute `bootstrap --apply`,
`cutover-preflight --inspect`, `import-snapshot` or
`reconcile-snapshot --assess`. Help and version output are the only container
executions allowed in this course session.

## Session variables

Populate these variables outside the recorded frame. Never display their host
paths:

```bash
TAG=v2.0.0-alpha.2
PROJECT=elvis-v2-alpha2-course-proof
IMAGE='ghcr.io/cluster2600/elvis-v2-operator@sha256:04465358e0c9e230272fb587f0f01da3d859d79bf68be8cf704b95548be4f919'
ARCHIVE="elvis-v2-operator-${TAG}.tar.gz"
BUNDLE_ROOT="elvis-v2-operator-${TAG}"
BUNDLE="$PWD/$BUNDLE_ROOT"
COMPOSE_FILE="$BUNDLE/compose.preview.yml"
ENV_FILE="$BUNDLE/.env"
OPERATOR_DIR="$BUNDLE/operator"
```

Use an exact project name on every Compose command. Never run `down -v`, a
Docker prune, a daemon restart, or a command against another Compose project.
Before SS02, execute the sole entry in
`required_session_setup_commands` from the manifest outside the recorded frame.
It first rejects any pre-existing container, network or volume carrying the
exact project label, then records and exports the unrelated-container baseline
hash used by SS14. A collision aborts before any Compose cleanup can run.

## Original and derivative custody

1. Record the unedited terminal frame or screen capture outside Git.
2. Hash it immediately and store the raw SHA-256 with UTC capture time.
3. Review the original for the forbidden fields above.
4. Create a separate cropped/redacted derivative; never overwrite the raw.
5. Hash the derivative and record every edit in the capture manifest.
6. A real terminal capture must remain labelled `original-terminal-capture`.
   A frame rendered later from the raw transcript must be labelled
   `derived-transcript-visual`, never `screenshot` or `runtime evidence`.
7. Keep `publication_allowed=false` until the exact Studio preview and its hash
   receive explicit user approval.

## Shot protocol

`capture-manifest.template.json` is the single executable command source. Each
SS02–SS14 record has exactly one non-empty `exact_command`. This prose specifies
framing, expected output and redaction only: it never duplicates or overrides a
shot command. The contract test rejects missing commands, unsafe resolved
Compose output, unsorted asset/archive evidence, a missing baseline initializer
or a command fragment copied back into this protocol.

### SS02 — prerequisites

Execute only the manifest's SS02 command and record its bounded output.

The frame must show `x86_64`, Docker/Compose versions and
`gh-attestation: available`. Crop the prompt and neighbouring terminal output.

### SS03 — six release assets

Use the public release download command from `INSTALL_V2.md`, then display a
bounded, sorted list containing exactly the tarball, two SPDX files,
`IMAGE_DIGEST.txt`, `SHA256SUMS` and the in-toto bundle. Work in a neutral
directory and keep its path outside the frame.

### SS04 — checksums

Execute only the manifest's SS04 command.

Every listed subject must be `OK`. An error or omitted subject invalidates the
capture session.

### SS05 — attestation

Capture the strict verification of the tarball against the public repository,
release tag, source commit and release workflow. Do not display authentication
state or a token. The success result must identify the exact subject.

### SS06 — archive audit

Capture the sorted archive inventory before extraction. The manifest command
compares it to a closed expected list and rejects every entry type except a
regular file or directory, so the list has one bounded root and no absolute
path, `..` segment, link escape or unexpected entry.

### SS07 — immutable image

Execute the manifest's post-attestation extraction command. It requires a fresh
destination, extracts without archived ownership or permissions, and proves
that the outer release asset, the authenticated bundle copy and the manifest's
literal image digest are identical. Display only that verified inner digest.

### SS08 — Compose validation

The manifest's SS08 command installs the example environment with mode `0600`,
creates `operator/` with mode `0700`, proves that directory empty, and then
validates Compose. It explicitly fixes the operator directory and UID/GID on
every Compose invocation so ambient shell variables cannot override the proven
inputs. No hidden preparation command is allowed.

Show one `operator` service and the exact digest. Never show resolved host
paths or environment values.

### SS09 — public pull

Capture the targeted pull for `operator` and a final
`docker image inspect --format '{{.Os}}/{{.Architecture}}' "$IMAGE"`. The
expected platform is `linux/amd64`; registry credentials and other images stay
out of frame.

### SS10 — main help and safety boundary

Execute only the manifest's SS10 command. It runs the image without network or
mounts.

The frame must show exactly the four operator commands and `ACTIVE NO-GO`.

### SS11 — release and Python versions

Use the same no-network hardening for `--version`. For Python, override the
entrypoint and execute `python --version` with `--network none`, no mounts and
the same read-only/capability restrictions. Show only `2.0.0-alpha.2` and
`Python 3.14.6`.

### SS12 — four help contracts

For each of `bootstrap`, `cutover-preflight`, `import-snapshot` and
`reconcile-snapshot`, run only `<command> --help` with the SS10 no-network
container flags. Capture one bounded synopsis per command. No configuration or
receipt may be supplied.

### SS13 — hardened service surface

From the resolved Compose document, assert rather than merely display non-root
execution, read-only root filesystem, exactly dropped `ALL` capabilities,
`no-new-privileges:true`, PIDs `64`, the exact bounded `/tmp` tmpfs, no
published port, exact user `65532:65532`, and exactly one read-only bind from
the proven-empty operator directory to `/run/operator`. The expected source
path is supplied privately to the parser and compared without being printed.
Retain only the reviewed public-safe fields; never publish the host bind path
or environment values.

### SS14 — cleanup and zero residue

Execute only the manifest's SS14 command. It performs the exact scoped cleanup,
counts only resources with the project label, and compares the current
unrelated-container hash to the baseline initialized before SS02.
The pre-session collision guards are what make this cleanup ownership-safe;
without their PASS, SS14 must not run.

Then count resources bearing the exact Compose project label. The capture must
show zero containers, networks and volumes, plus an unchanged pre/post hash of
the unrelated running-container inventory. Do not display that inventory.

## Session acceptance

`capture-manifest.template.json` is the machine-readable command source for
SS02–SS14. This prose explains framing and redaction but may not change a
command. Before any future recapture, verify the manifest SHA-256 recorded in
the session record; a command change creates a new session ID and invalidates
the prior transcript for that shot.

The capture session passes only if all raw hashes are recorded, all derivatives
are reviewed, no sensitive-pattern hit remains, the image digest and versions
match, the operator directory stayed empty, no database-capable input was
mounted, the help commands ran with networking disabled, cleanup left zero
project resources, and the unrelated-stack hash remained unchanged.

The private raw transcript `KALI-SESSION-01` is useful corroborating evidence
for SS02–SS14, but it is not a screenshot. Its metadata remains in
`captures/KALI-SESSION-01.json`; future graphical captures must receive their
own manifest records and SHA-256 values.
