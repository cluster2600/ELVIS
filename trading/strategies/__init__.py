from .base_strategy import BaseStrategy
from .ensemble_strategy import EnsembleStrategy

# Optional imports - gracefully handle missing dependencies
try:
    from .ema_rsi_strategy import EmaRsiStrategy
except ImportError as e:
    print(f"Warning: EmaRsiStrategy not available due to missing dependency: {e}")
    EmaRsiStrategy = None

try:
    from .grid_strategy import GridStrategy
except ImportError as e:
    print(f"Warning: GridStrategy not available due to missing dependency: {e}")
    GridStrategy = None

try:
    from .mean_reversion_strategy import MeanReversionStrategy
except ImportError as e:
    print(f"Warning: MeanReversionStrategy not available due to missing dependency: {e}")
    MeanReversionStrategy = None

try:
    from .sentiment_strategy import SentimentStrategy
except ImportError as e:
    print(f"Warning: SentimentStrategy not available due to missing dependency: {e}")
    SentimentStrategy = None

try:
    from .technical_strategy import TechnicalStrategy
except ImportError as e:
    print(f"Warning: TechnicalStrategy not available due to missing dependency: {e}")
    TechnicalStrategy = None

try:
    from .trend_following_strategy import TrendFollowingStrategy
except ImportError as e:
    print(f"Warning: TrendFollowingStrategy not available due to missing dependency: {e}")
    TrendFollowingStrategy = None
