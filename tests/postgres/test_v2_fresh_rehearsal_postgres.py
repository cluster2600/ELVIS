"""Disposable PostgreSQL 15 rehearsal for the dormant V2 bootstrap Compose."""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest

_REPOSITORY = Path(__file__).resolve().parents[2]
_DEPLOYMENT = _REPOSITORY / "deploy" / "v2"
_COMPOSE = _DEPLOYMENT / "compose.bootstrap.yml"
_POSTGRES_IMAGE = (
    "postgres:15-alpine@"
    "sha256:3d0f7584ed7d04e27fa050d6683a74746608faf21f202be78460d679cc56461f"
)
_DATABASE = "elvis_paper_v2_rehearsal"
_ADMIN_ROLE = "elvis_bootstrap_admin"
_POSTGRES_ADDRESS = "10.254.90.2"
_LOGIN_ROLE_KEYS = (
    "migrator",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_REQUIRED_ENV = "ELVIS_TEST_V2_FRESH_REHEARSAL_REQUIRED"

pytestmark = pytest.mark.skipif(
    os.getenv(_REQUIRED_ENV) != "1",
    reason=f"set {_REQUIRED_ENV}=1 to run the disposable Docker rehearsal",
)


def _redact(value: str, sensitive_values: tuple[str, ...]) -> str:
    for sensitive in sensitive_values:
        value = value.replace(sensitive, "<redacted>")
    return value


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    input_text: str | None = None,
    expected_exit_codes: tuple[int, ...] = (0,),
    sensitive_values: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    safe_environment = (
        None
        if environment is None
        else {
            "ELVIS_V2_OPERATOR_DIR": environment.get("ELVIS_V2_OPERATOR_DIR", ""),
            "ELVIS_V2_OPERATOR_GID": environment.get("ELVIS_V2_OPERATOR_GID", ""),
            "ELVIS_V2_OPERATOR_UID": environment.get("ELVIS_V2_OPERATOR_UID", ""),
            "PATH": environment.get("PATH", ""),
        }
    )
    result = subprocess.run(
        command,
        cwd=_REPOSITORY,
        env=safe_environment,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    if any(
        sensitive and (sensitive in result.stdout or sensitive in result.stderr)
        for sensitive in sensitive_values
    ):
        pytest.fail(
            "subprocess output exposed a generated rehearsal secret",
            pytrace=False,
        )
    if result.returncode not in expected_exit_codes:
        rendered_command = " ".join(command)
        details = (
            f"command exited {result.returncode}, expected {expected_exit_codes}\n"
            f"command: {rendered_command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        pytest.fail(_redact(details, sensitive_values), pytrace=False)
    return result


def _write_private(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o600)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@dataclass(frozen=True)
class Rehearsal:
    compose: list[str]
    environment: dict[str, str] = field(repr=False)
    operator_directory: Path
    admin_password: str = field(repr=False)
    roles: dict[str, str]
    role_passwords: dict[str, str] = field(repr=False)

    def run_compose(
        self,
        *arguments: str,
        input_text: str | None = None,
        expected_exit_codes: tuple[int, ...] = (0,),
        sensitive_values: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        all_sensitive_values = (
            self.admin_password,
            *self.role_passwords.values(),
            *sensitive_values,
        )
        return _run(
            [*self.compose, *arguments],
            environment=self.environment,
            input_text=input_text,
            expected_exit_codes=expected_exit_codes,
            sensitive_values=all_sensitive_values,
        )

    def run_bootstrap(
        self,
        manifest_name: str,
        *,
        expected_exit_code: int,
    ) -> dict[str, object]:
        result = self.run_compose(
            "run",
            "--rm",
            "--no-deps",
            "bootstrap",
            "--config",
            f"/run/operator/{manifest_name}",
            "--apply",
            "--confirm-exclusive-ddl-role-window",
            expected_exit_codes=(expected_exit_code,),
            sensitive_values=(self.admin_password, *self.role_passwords.values()),
        )
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(output_lines) == 1
        payload = json.loads(output_lines[0])
        assert isinstance(payload, dict)
        return payload

    def prove_operator_file_boundary_and_admin_connection(self) -> None:
        script = (
            "import json, os, stat\n"
            "import psycopg2\n"
            "paths = ('/run/operator/pgpass', '/run/operator/pg_service.conf', "
            "'/run/operator/bootstrap-stage-v1.json', "
            "'/run/operator/bootstrap-complete-v1.json')\n"
            "assert all(stat.S_IMODE(os.stat(path).st_mode) == 0o600 for path in paths)\n"
            "connection = psycopg2.connect(service='elvis_v2_admin', "
            "application_name='elvis-v2-rehearsal-boundary-test', connect_timeout=5)\n"
            "with connection.cursor() as cursor:\n"
            "    cursor.execute('SELECT current_database(), session_user')\n"
            "    identity = cursor.fetchone()\n"
            "connection.close()\n"
            "print(json.dumps({'database': identity[0], 'role': identity[1]}))\n"
        )
        result = self.run_compose(
            "run",
            "--rm",
            "--no-deps",
            "--entrypoint",
            "python",
            "bootstrap",
            "-c",
            script,
            sensitive_values=(self.admin_password, *self.role_passwords.values()),
        )
        assert json.loads(result.stdout.splitlines()[-1]) == {
            "database": _DATABASE,
            "role": _ADMIN_ROLE,
        }

    def admin_psql(self, sql_input: str) -> subprocess.CompletedProcess[str]:
        return self.run_compose(
            "exec",
            "-T",
            "postgres",
            "sh",
            "-ceu",
            (
                'export PGPASSWORD="$(cat /run/secrets/postgres_admin_password)"; '
                "exec psql -X --no-psqlrc --set ON_ERROR_STOP=1 "
                "--tuples-only --no-align --quiet "
                f"--host {_POSTGRES_ADDRESS} --port 5432 "
                f"--dbname {_DATABASE} --username {_ADMIN_ROLE}"
            ),
            input_text=sql_input,
            sensitive_values=(self.admin_password, *self.role_passwords.values()),
        )

    def role_psql(
        self,
        role: str,
        password: str,
        *,
        database: str = _DATABASE,
        expected_exit_code: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        assert re.fullmatch(r"[a-z][a-z0-9_]{0,62}", role)
        assert re.fullmatch(r"[a-z][a-z0-9_]{0,62}", database)
        return self.run_compose(
            "exec",
            "-T",
            "postgres",
            "sh",
            "-ceu",
            (
                "IFS= read -r PGPASSWORD; export PGPASSWORD; "
                "exec psql -X --no-psqlrc --tuples-only --no-align --quiet "
                f"--host {_POSTGRES_ADDRESS} --port 5432 "
                f"--dbname {database} --username {role} "
                "--command \"SELECT session_user || ':' || current_user\""
            ),
            input_text=f"{password}\n",
            expected_exit_codes=(expected_exit_code,),
            sensitive_values=(self.admin_password, *self.role_passwords.values()),
        )

    def assert_rehearsal_marker(self) -> None:
        result = self.run_compose(
            "exec",
            "-T",
            "postgres",
            "sh",
            "-ceu",
            (
                'marker="$PGDATA/.elvis-v2-fresh-rehearsal-v1"; '
                'test "$(cat "$marker")" = "elvis-v2-fresh-rehearsal:v1"; '
                'test "$(stat -c %a "$marker")" = "600"; '
                "printf '%s\\n' marker-ok"
            ),
        )
        assert result.stdout.strip() == "marker-ok"

    def restart_postgres_and_wait(self) -> None:
        self.run_compose("restart", "--no-deps", "postgres")
        self.run_compose(
            "up",
            "--detach",
            "--wait",
            "--wait-timeout",
            "120",
            "postgres",
        )


def _compose_command(project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(_COMPOSE),
        "--profile",
        "v2-rehearsal",
        "--profile",
        "v2-operator",
    ]


def _assert_project_absent(project: str) -> None:
    for resource, subcommand in (
        ("containers", ["container", "ls", "--all"]),
        ("volumes", ["volume", "ls"]),
        ("networks", ["network", "ls"]),
    ):
        result = _run(
            [
                "docker",
                *subcommand,
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ]
        )
        assert result.stdout.strip() == "", f"residual {resource} for {project}"


def _volume_fingerprint(volume_name: str) -> str:
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--volume",
            f"{volume_name}:/target:ro",
            "--entrypoint",
            "sh",
            _POSTGRES_IMAGE,
            "-ceu",
            (
                "test -f /target/preexisting; "
                "test ! -e /target/PG_VERSION; "
                "test ! -e /target/.elvis-v2-fresh-rehearsal-v1; "
                "stat -c '%a:%s' /target/preexisting; "
                "sha256sum /target/preexisting"
            ),
        ]
    )
    return result.stdout


@pytest.fixture
def fresh_rehearsal(tmp_path: Path) -> Rehearsal:
    project = f"elvis-v2-rehearsal-{uuid4().hex[:12]}"
    operator_directory = tmp_path / "operator"
    operator_directory.mkdir(mode=0o700)
    admin_password = secrets.token_hex(24)
    stage = json.loads(
        (_DEPLOYMENT / "bootstrap-stage-v1.example.json").read_text(encoding="utf-8")
    )
    complete = json.loads(
        (_DEPLOYMENT / "bootstrap-complete-v1.example.json").read_text(encoding="utf-8")
    )
    roles = {key: complete["roles"][key] for key in _LOGIN_ROLE_KEYS}
    role_passwords = {key: secrets.token_hex(24) for key in _LOGIN_ROLE_KEYS}

    _write_private(operator_directory / "postgres_admin_password", admin_password)
    passfile_lines = [
        f"postgres:5432:{_DATABASE}:{_ADMIN_ROLE}:{admin_password}",
        *(
            f"postgres:5432:{_DATABASE}:{roles[key]}:{role_passwords[key]}"
            for key in _LOGIN_ROLE_KEYS
        ),
    ]
    _write_private(operator_directory / "pgpass", "\n".join(passfile_lines) + "\n")
    _write_private(
        operator_directory / "pg_service.conf",
        (_DEPLOYMENT / "pg_service.conf.example").read_text(encoding="utf-8"),
    )
    _write_private(
        operator_directory / "bootstrap-stage-v1.json",
        json.dumps(stage, separators=(",", ":")),
    )
    _write_private(
        operator_directory / "bootstrap-complete-v1.json",
        json.dumps(complete, separators=(",", ":")),
    )

    environment = dict(os.environ)
    environment["ELVIS_V2_OPERATOR_DIR"] = str(operator_directory)
    environment["ELVIS_V2_OPERATOR_UID"] = str(os.getuid())
    environment["ELVIS_V2_OPERATOR_GID"] = str(os.getgid())
    compose = _compose_command(project)
    rehearsal = Rehearsal(
        compose=compose,
        environment=environment,
        operator_directory=operator_directory,
        admin_password=admin_password,
        roles=roles,
        role_passwords=role_passwords,
    )

    try:
        rehearsal.run_compose("build", "bootstrap")
        rehearsal.run_compose(
            "up",
            "--detach",
            "--wait",
            "--wait-timeout",
            "120",
            "postgres",
        )
        yield rehearsal
    finally:
        rehearsal.run_compose(
            "down",
            "--volumes",
            "--remove-orphans",
            "--rmi",
            "local",
            expected_exit_codes=(0,),
            sensitive_values=(admin_password, *role_passwords.values()),
        )
        _assert_project_absent(project)


def _provision_roles(rehearsal: Rehearsal) -> None:
    variables = ["\\set ON_ERROR_STOP on", "\\set QUIET on"]
    value_rows = []
    for index, key in enumerate(_LOGIN_ROLE_KEYS):
        role = rehearsal.roles[key]
        password = rehearsal.role_passwords[key]
        assert re.fullmatch(r"[a-z][a-z0-9_]{0,62}", role)
        assert re.fullmatch(r"[0-9a-f]{48}", password)
        variables.extend(
            (
                f"\\set role_{index} '{role}'",
                f"\\set password_{index} '{password}'",
            )
        )
        value_rows.append(f"(:'role_{index}', :'password_{index}')")

    statement = (
        "SELECT format('ALTER ROLE %I LOGIN PASSWORD %L', role_name, role_password)\n"
        f"FROM (VALUES {', '.join(value_rows)}) AS credentials(role_name, role_password)\n"
        "\\gexec\n"
    )
    rehearsal.admin_psql("\n".join(variables) + "\n" + statement)


def test_fresh_compose_rehearsal_stages_provisions_and_completes(
    fresh_rehearsal: Rehearsal,
) -> None:
    rehearsal = fresh_rehearsal
    fixture_representation = repr(rehearsal)
    assert "environment=" not in fixture_representation
    assert "admin_password=" not in fixture_representation
    assert "role_passwords=" not in fixture_representation
    rehearsal.prove_operator_file_boundary_and_admin_connection()
    first = rehearsal.run_bootstrap(
        "bootstrap-stage-v1.json",
        expected_exit_code=10,
    )
    assert first == {
        "status": "CREDENTIALS_REQUIRED",
        "migration_versions": [],
        "verified_role_probes": [],
        "pending_role_credentials": [rehearsal.roles[key] for key in _LOGIN_ROLE_KEYS],
        "old_shared_runtime_demoted": False,
    }

    _provision_roles(rehearsal)

    completed = rehearsal.run_bootstrap(
        "bootstrap-complete-v1.json",
        expected_exit_code=0,
    )
    assert completed == {
        "status": "COMPLETE",
        "migration_versions": [1, 2, 3, 4, 5, 6],
        "verified_role_probes": [rehearsal.roles[key] for key in _LOGIN_ROLE_KEYS],
        "pending_role_credentials": [],
        "old_shared_runtime_demoted": False,
    }
    assert (
        rehearsal.run_bootstrap(
            "bootstrap-complete-v1.json",
            expected_exit_code=0,
        )
        == completed
    )

    scram_check = rehearsal.admin_psql(
        "SELECT count(*)::text || ':' || "
        "COALESCE(bool_and(rolpassword LIKE 'SCRAM-SHA-256$%'), false)::text "
        "FROM pg_authid "
        "WHERE rolname IN ("
        + ",".join(f"'{rehearsal.roles[key]}'" for key in _LOGIN_ROLE_KEYS)
        + ");\n"
    )
    assert scram_check.stdout.strip() == "6:true"

    server_policy = rehearsal.admin_psql(
        "SHOW password_encryption;\n"
        "SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL;\n"
    )
    assert server_policy.stdout.splitlines() == ["scram-sha-256", "0"]

    for key in _LOGIN_ROLE_KEYS:
        role = rehearsal.roles[key]
        own_credential = rehearsal.role_psql(
            role,
            rehearsal.role_passwords[key],
        )
        assert own_credential.stdout.strip() == f"{role}:{role}"

    rehearsal.assert_rehearsal_marker()
    rehearsal.restart_postgres_and_wait()
    rehearsal.assert_rehearsal_marker()

    for index, key in enumerate(_LOGIN_ROLE_KEYS):
        crossed_key = _LOGIN_ROLE_KEYS[(index + 1) % len(_LOGIN_ROLE_KEYS)]
        crossed = rehearsal.role_psql(
            rehearsal.roles[key],
            rehearsal.role_passwords[crossed_key],
            expected_exit_code=2,
        )
        assert "password authentication failed" in crossed.stderr.lower()

    rejected_other_database = rehearsal.role_psql(
        _ADMIN_ROLE,
        rehearsal.admin_password,
        database="postgres",
        expected_exit_code=2,
    )
    assert "pg_hba.conf rejects connection" in rejected_other_database.stderr

    postgres_log_result = rehearsal.run_compose(
        "logs",
        "--no-color",
        "postgres",
    )
    postgres_logs = (
        postgres_log_result.stdout + "\n" + postgres_log_result.stderr
    ).splitlines()
    fatal_lines = [line for line in postgres_logs if "FATAL:" in line]
    expected_fatal_fragments = [
        *(
            f'password authentication failed for user "{rehearsal.roles[key]}"'
            for key in _LOGIN_ROLE_KEYS
        ),
        (
            "pg_hba.conf rejects connection for host "
            f'"10.254.90.2", user "{_ADMIN_ROLE}", database "postgres"'
        ),
    ]
    assert len(fatal_lines) == len(expected_fatal_fragments)
    for fragment in expected_fatal_fragments:
        assert sum(fragment in line for line in fatal_lines) == 1


def test_rehearsal_refuses_an_unmarked_nonempty_volume_without_modifying_it(
    tmp_path: Path,
) -> None:
    project = f"elvis-v2-unmarked-{uuid4().hex[:12]}"
    volume_name = f"{project}_rehearsal-data"
    operator_directory = tmp_path / "operator"
    operator_directory.mkdir(mode=0o700)
    admin_password = secrets.token_hex(24)
    _write_private(operator_directory / "postgres_admin_password", admin_password)

    environment = dict(os.environ)
    environment["ELVIS_V2_OPERATOR_DIR"] = str(operator_directory)
    environment["ELVIS_V2_OPERATOR_UID"] = str(os.getuid())
    environment["ELVIS_V2_OPERATOR_GID"] = str(os.getgid())
    rehearsal = Rehearsal(
        compose=_compose_command(project),
        environment=environment,
        operator_directory=operator_directory,
        admin_password=admin_password,
        roles={},
        role_passwords={},
    )

    try:
        rehearsal.run_compose("create", "postgres")
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--volume",
                f"{volume_name}:/target",
                "--entrypoint",
                "sh",
                _POSTGRES_IMAGE,
                "-ceu",
                (
                    "umask 077; "
                    "printf '%s\\n' unmarked-preexisting-volume "
                    "> /target/preexisting"
                ),
            ]
        )
        before = _volume_fingerprint(volume_name)

        rehearsal.run_compose(
            "start",
            "--wait",
            "--wait-timeout",
            "15",
            "postgres",
            expected_exit_codes=(1,),
        )
        log_result = rehearsal.run_compose("logs", "--no-color", "postgres")
        logs = log_result.stdout + "\n" + log_result.stderr
        assert "rehearsal volume is not empty" in logs
        assert _volume_fingerprint(volume_name) == before
    finally:
        rehearsal.run_compose(
            "down",
            "--volumes",
            "--remove-orphans",
            expected_exit_codes=(0,),
        )
        _assert_project_absent(project)
