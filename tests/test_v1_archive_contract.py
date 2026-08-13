"""Repository contract for retired V1 documentation and deployment surfaces."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

_REPOSITORY = Path(__file__).resolve().parents[1]
_ARCHIVE_INDEX = _REPOSITORY / "docs" / "archive" / "v1" / "README.md"

_RETIRED_PATHS = {
    _REPOSITORY / "docs" / "APPLE_CONTAINER_SETUP.md",
    _REPOSITORY / "docs" / "APPLE_NATIVE_CONTAINER_GUIDE.md",
    _REPOSITORY / "docs" / "README_APPLE_CONTAINERS.md",
    _REPOSITORY / "docs" / "comprehensive_improvements.md",
    _REPOSITORY / "docs" / "future_improvements.md",
    _REPOSITORY / "docs" / "implementation_summary.md",
    _REPOSITORY / "docs" / "VAULT_INTEGRATION.md",
    _REPOSITORY / "docs" / "VAULT_SETUP.md",
    _REPOSITORY / "diagrams" / "v1-retirement-boundary.mmd",
    _REPOSITORY / "diagrams" / "v1-retirement-boundary.svg",
    _REPOSITORY / "diagrams" / "v1-retirement-boundary.png",
    _REPOSITORY / "diagrams" / "v1-retirement-boundary.excalidraw",
    _REPOSITORY / "scripts" / "apple_container_elvis.sh",
    _REPOSITORY / "scripts" / "apple_container_native.sh",
    _REPOSITORY / "scripts" / "fix_apple_container_build.sh",
    _REPOSITORY / "scripts" / "setup_apple_containers.sh",
    _REPOSITORY / "scripts" / "test_apple_container.sh",
    _REPOSITORY / "scripts" / "test_container_setup.sh",
    _REPOSITORY / "scripts" / "setup_secure_config.sh",
    _REPOSITORY / "scripts" / "start_bot_with_vault.sh",
    _REPOSITORY / "scripts" / "run_console_dashboard.sh",
    _REPOSITORY / "docker" / "Dockerfile.simple",
    _REPOSITORY / "trading" / "config" / "data_config.yaml",
    _REPOSITORY / "trading" / "config" / "model_config.yaml",
    _REPOSITORY / "trading" / "config" / "risk_config.yaml",
}

_RETIRED_MANUAL_TESTS = {
    "test_api_connection.py",
    "test_balanced_simple.py",
    "test_balanced_strategy.py",
    "test_bnb_btc_integration.py",
    "test_bnbbtc_fix.py",
    "test_bug_fixes.py",
    "test_candlestick_dashboard.py",
    "test_cooldown_fix.py",
    "test_dashboard_errors.py",
    "test_dashboard_layout.py",
    "test_dashboard_market_depth.py",
    "test_dashboard_positions.py",
    "test_dashboard_stats.py",
    "test_database_connection.py",
    "test_database_integration.py",
    "test_enhanced_rf.py",
    "test_enhanced_trading.py",
    "test_executor_initialization_fix.py",
    "test_fixed_strategies.py",
    "test_force_trading.py",
    "test_fresh_balance.py",
    "test_futures_leverage.py",
    "test_immediate_trading.py",
    "test_leverage_settings.py",
    "test_live_pnl.py",
    "test_live_signals.py",
    "test_live_trade_stats.py",
    "test_llm_strategy.py",
    "test_market_depth.py",
    "test_market_depth_debug.py",
    "test_multi_exchange.py",
    "test_paper_trading_bnb.py",
    "test_position_management.py",
    "test_position_risk.py",
    "test_positions_display.py",
    "test_price_fetcher_fix.py",
    "test_profit_optimization.py",
    "test_recalibrated_bot.py",
    "test_recent_trades_display.py",
    "test_research_strategy.py",
    "test_right_pane.py",
    "test_rl_integration.py",
    "test_rl_live_trading.py",
    "test_signal_generation.py",
    "test_specific_cooldown.py",
    "test_trading_execution.py",
    "test_vault_connection.py",
    "test_vault_integration.py",
    "test_warnings_fixed.py",
    "test_working_llm.py",
}

_ACTIVE_DOCS = (
    _REPOSITORY / "README.md",
    _REPOSITORY / "CHANGELOG.md",
    _REPOSITORY / "INSTALL_V2.md",
    _REPOSITORY / "RELEASE_NOTES.md",
    _REPOSITORY / "scripts" / "README.md",
    *(_REPOSITORY / "docs").rglob("*.md"),
)

_OLD_MINOR = "3" + "." + "10"
_OLD_TOOL_TARGET = "py" + "3" + "10"
_OLD_ML_SUFFIX = "ml" + "3" + "10"
_OBSOLETE_INTERPRETER = re.compile(
    rf"(?:Python\s*{re.escape(_OLD_MINOR)}|python{re.escape(_OLD_MINOR)}"
    rf"|{_OLD_TOOL_TARGET}|{_OLD_ML_SUFFIX})",
    re.IGNORECASE,
)
_INLINE_MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")


def test_archive_is_a_single_tagged_restore_manifest() -> None:
    archive_files = {
        path for path in (_REPOSITORY / "docs" / "archive").rglob("*") if path.is_file()
    }
    assert archive_files == {_ARCHIVE_INDEX}

    manifest = _ARCHIVE_INDEX.read_text(encoding="utf-8")
    assert "v0.3.0" in manifest
    assert "git show v0.3.0:" in manifest
    assert "forensic" in manifest.lower()
    assert "not an install" in manifest.lower()


def test_unverified_v1_deployment_surfaces_are_absent() -> None:
    assert all(not path.exists() for path in _RETIRED_PATHS)
    ansible = _REPOSITORY / "ansible"
    assert not ansible.exists() or not any(
        path.is_file() for path in ansible.rglob("*")
    )


def test_manual_v1_diagnostics_are_not_collected_as_release_tests() -> None:
    assert len(_RETIRED_MANUAL_TESTS) == 50
    assert all(
        not (_REPOSITORY / "tests" / name).exists() for name in _RETIRED_MANUAL_TESTS
    )


def test_active_documentation_has_no_obsolete_interpreter_claim() -> None:
    offenders = {
        path.relative_to(_REPOSITORY).as_posix(): _OBSOLETE_INTERPRETER.findall(
            path.read_text(encoding="utf-8")
        )
        for path in _ACTIVE_DOCS
        if _OBSOLETE_INTERPRETER.search(path.read_text(encoding="utf-8"))
    }
    assert offenders == {}


def test_active_documentation_has_no_broken_relative_link() -> None:
    broken: dict[str, list[str]] = {}

    for document in _ACTIVE_DOCS:
        for raw_target in _INLINE_MARKDOWN_LINK.findall(
            document.read_text(encoding="utf-8")
        ):
            target = raw_target.strip()
            if target.startswith("<") and ">" in target:
                target = target[1 : target.index(">")]
            else:
                target = target.split(maxsplit=1)[0]

            path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not path_text or target.startswith(("#", "/", "mailto:")):
                continue
            if re.match(r"^[a-z][a-z0-9+.-]*://", target, re.IGNORECASE):
                continue

            resolved = (document.parent / path_text).resolve()
            if not resolved.exists():
                relative_document = document.relative_to(_REPOSITORY).as_posix()
                broken.setdefault(relative_document, []).append(target)

    assert broken == {}


def test_active_entry_docs_keep_the_operator_safety_boundary() -> None:
    for path in (
        _REPOSITORY / "README.md",
        _REPOSITORY / "docs" / "README.md",
        _REPOSITORY / "docs" / "DEPLOYMENT.md",
        _REPOSITORY / "docs" / "PAPER_TRADING_SETUP.md",
        _REPOSITORY / "docs" / "V2_ARCHITECTURE.md",
        _REPOSITORY / "docs" / "architecture_migration" / "04-migration-roadmap.md",
    ):
        source = path.read_text(encoding="utf-8")
        assert "Python 3.14" in source
        assert "ACTIVE" in source and "NO-GO" in source


def test_compatibility_entrypoints_do_not_promise_returns_or_live_execution() -> None:
    research_runner = (_REPOSITORY / "scripts" / "run_research_strategy.sh").read_text(
        encoding="utf-8"
    )
    training_runner = (_REPOSITORY / "scripts" / "run_training.sh").read_text(
        encoding="utf-8"
    )
    compose = (_REPOSITORY / "docker-compose.yml").read_text(encoding="utf-8")
    api = (_REPOSITORY / "trading" / "api" / "app.py").read_text(encoding="utf-8")
    swagger = (_REPOSITORY / "trading" / "api" / "swagger.py").read_text(
        encoding="utf-8"
    )
    dashboard = (_REPOSITORY / "scripts" / "native_console_dashboard.py").read_text(
        encoding="utf-8"
    )
    console_dashboard = (_REPOSITORY / "utils" / "console_dashboard.py").read_text(
        encoding="utf-8"
    )
    example_env = (_REPOSITORY / ".env.example").read_text(encoding="utf-8")
    control_api_runner = (_REPOSITORY / "trading" / "scripts" / "run_api.py").read_text(
        encoding="utf-8"
    )
    trade_history_api = (
        _REPOSITORY / "trading" / "utils" / "trade_history_api.py"
    ).read_text(encoding="utf-8")

    assert "14.9%" not in research_runner
    assert "2.02 Sharpe" not in research_runner
    assert "--live" not in research_runner
    assert "14.9% target returns" not in training_runner
    assert "LEVERAGE=100" not in compose
    assert "PROFIT_MODE=aggressive" not in compose
    assert "COOLDOWN_DISABLED=true" not in compose
    assert "POSTGRES_PASSWORD=elvis_password" not in compose
    assert "GF_SECURITY_ADMIN_PASSWORD=admin" not in compose
    assert 'if mode != "paper"' in api
    assert '"enum": ["paper", "live"]' not in swagger
    assert "Status: LIVE TRADING" not in dashboard
    assert "Status: LIVE TRADING" not in console_dashboard
    assert "paper or live" not in example_env
    assert 'os.getenv("API_HOST", "127.0.0.1")' in control_api_runner
    assert 'getenv("TRADE_HISTORY_API_HOST", "127.0.0.1")' in trade_history_api
    assert "TRADE_API_HOST" not in trade_history_api
    for variable in (
        "API_SECRET_KEY",
        "API_USERNAME",
        "API_PASSWORD",
        "API_HOST",
        "API_PORT",
        "API_KEY",
        "TRADE_HISTORY_API_HOST",
        "TRADE_HISTORY_API_PORT",
    ):
        assert f"{variable}=" in example_env
    assert "bot will refuse to start if VAULT_TOKEN is not set" not in example_env


def test_active_training_helpers_have_no_literal_secret_fallbacks() -> None:
    helper_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_REPOSITORY / "scripts").glob("train_*.py")
    )

    assert "trading-bot-token" not in helper_text
    assert "training_password" not in helper_text
    assert "elvis_password" not in helper_text


def test_v2_versioned_manifests_remain_active() -> None:
    names = {
        "bootstrap-stage-v1.example.json",
        "bootstrap-complete-v1.example.json",
        "cutover-preflight-v1.example.json",
        "legacy-snapshot-import-v1.example.json",
        "legacy-snapshot-reconciliation-v1.example.json",
    }
    assert all((_REPOSITORY / "deploy" / "v2" / name).is_file() for name in names)


def test_loaded_configuration_contracts_remain_active() -> None:
    retained = (
        _REPOSITORY / "trading" / "config" / "validation_config.yaml",
        _REPOSITORY / "training" / "config" / "model_config.yaml",
        _REPOSITORY / "trading_config.yaml",
        _REPOSITORY / "config" / "config.py",
    )
    assert all(path.is_file() for path in retained)


def test_active_compatibility_architecture_remains_for_rollback_context() -> None:
    for name in ("architecture.md", "COMPONENTS.md", "ELVIS_SYSTEM_ARCHITECTURE.md"):
        path = _REPOSITORY / "docs" / name
        assert path.is_file()
        assert "compatib" in path.read_text(encoding="utf-8").lower()
