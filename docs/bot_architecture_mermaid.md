# Bot Architecture Mermaid Diagrams

## Component Interaction Flowchart

```mermaid
graph TD
    Main["main.py"]
    BinanceExecutor["BinanceExecutor"]
    EnsembleStrategy["EnsembleStrategy"]
    TelegramNotifier["TelegramNotifier"]
    ConsoleDashboard["ConsoleDashboard"]
    PriceFetcher["PriceFetcher"]
    RiskManager["AdvancedRiskManager"]
    TradeHistoryAPI["Trade History API Server"]

    Main --> BinanceExecutor
    Main --> EnsembleStrategy
    Main --> TelegramNotifier
    Main --> ConsoleDashboard
    Main --> PriceFetcher
    Main --> RiskManager
    Main --> TradeHistoryAPI

    EnsembleStrategy --> BinanceExecutor
    EnsembleStrategy --> PriceFetcher
    EnsembleStrategy --> RiskManager
    EnsembleStrategy --> TelegramNotifier

    ConsoleDashboard --> EnsembleStrategy
    ConsoleDashboard --> RiskManager
```

## Class Diagram

```mermaid
classDiagram
    class BaseStrategy {
        <<abstract>>
        +generate_signals()
        +calculate_stop_loss()
        +calculate_take_profit()
    }

    class EnsembleStrategy {
        +generate_signals()
        +calculate_stop_loss()
        +calculate_take_profit()
    }

    BaseStrategy <|-- EnsembleStrategy

    class BinanceExecutor {
        +initialize()
        +get_balance()
        +get_funding_rate()
        +get_order_book()
    }

    class TelegramNotifier {
        +send_message()
    }

    class ConsoleDashboard {
        +run()
        +_draw_frame()
    }

    class PriceFetcher {
        +get_price()
        +get_order_book()
    }

    class AdvancedRiskManager {
        +manage_risk()
    }
