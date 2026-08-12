from unittest.mock import Mock, patch

import pytest

import config as legacy_config
import config.config as canonical_config
import core.bootstrap as bootstrap_module
from core.di import Container


class FakeExecutor:
    initialize_results: list[bool] = []
    instances: list["FakeExecutor"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def initialize(self) -> bool:
        return self.__class__.initialize_results.pop(0)


def resolve_executor(*, initialize_results: list[bool]) -> object:
    local_container = Container()
    bootstrapper = bootstrap_module.ApplicationBootstrapper(mode="paper")
    bootstrapper.logger = Mock()
    FakeExecutor.instances = []
    FakeExecutor.initialize_results = list(initialize_results)

    with (
        patch.object(bootstrap_module, "container", local_container),
        patch(
            "trading.execution.binance_executor.BinanceExecutor",
            FakeExecutor,
        ),
    ):
        local_container.register_singleton("logger", lambda: bootstrapper.logger)
        bootstrapper._register_configurations()
        bootstrapper._register_trading_services()
        return local_container.get("executor")


def test_bootstrap_keeps_legacy_mode_config_and_aliases_canonical_leverage() -> None:
    assert bootstrap_module.TRADING_CONFIG is legacy_config.TRADING_CONFIG
    assert bootstrap_module.canonical_config_module is canonical_config


def test_primary_and_fallback_executor_receive_the_same_configured_leverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        bootstrap_module.canonical_config_module.TRADING_CONFIG,
        "DEFAULT_LEVERAGE",
        7,
    )

    resolved = resolve_executor(initialize_results=[False, True])

    assert resolved is FakeExecutor.instances[1]
    assert len(FakeExecutor.instances) == 2
    assert [
        instance.kwargs["default_leverage"] for instance in FakeExecutor.instances
    ] == [7, 7]
    assert FakeExecutor.instances[0].kwargs["is_testnet"] is True
    assert FakeExecutor.instances[0].kwargs["use_futures"] is False


def test_bootstrap_fails_closed_when_default_leverage_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(
        bootstrap_module.canonical_config_module.TRADING_CONFIG,
        "DEFAULT_LEVERAGE",
        raising=False,
    )

    with pytest.raises(KeyError, match="DEFAULT_LEVERAGE"):
        resolve_executor(initialize_results=[True])

    assert FakeExecutor.instances == []
