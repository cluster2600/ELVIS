"""
Metrics package for the BTC_BOT project.
"""

from core.metrics.metrics_utils import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_calmar_ratio,
    calculate_expectancy,
    calculate_max_drawdown,
    calculate_performance_metrics,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    compute_data_points_per_year,
    plot_drawdown,
    plot_equity_curve,
    plot_returns_distribution,
    write_metrics_to_file,
)
