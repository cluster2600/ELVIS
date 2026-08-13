# ELVIS V1 restore manifest

Historical V1 prose and unverified deployment helpers are intentionally absent
from the active tree. Git tags preserve them more accurately than a second,
stale documentation copy.

## Forensic source

Tag `v0.3.0` is the restore point for the retired surface:

- `ansible/` playbooks, inventory, templates, and setup wrapper;
- Apple-container guides and shell helpers;
- superseded completion reports under `docs/archive/`;
- the old 4,700-line migration execution log;
- historical release notes and diagrams; and
- removed unused configuration and vendored Galaxy roles.
- manual network/database diagnostics formerly collected as release tests.

Inspect a path without changing the worktree:

```bash
git show v0.3.0:docs/APPLE_CONTAINER_SETUP.md
git show v0.3.0:ansible/playbook.yml
git ls-tree -r --name-only v0.3.0 docs/archive
```

Restore only into a disposable review branch, never into an operating checkout:

```bash
git switch --detach v0.3.0
```

Those files contain obsolete credentials, leverage, interpreter, and
deployment assumptions. The tag is provenance, not an install or rollback
procedure. Runtime rollback remains a separately rehearsed database and
authority operation described by the
[V2 roadmap](../../architecture_migration/04-migration-roadmap.md).
