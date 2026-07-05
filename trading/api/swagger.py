"""
Swagger/OpenAPI documentation for ELVIS Trading Bot API
"""

from flask import Flask, jsonify
from flask_swagger_ui import get_swaggerui_blueprint

# Swagger UI configuration
SWAGGER_URL = "/api/docs"
API_URL = "/api/swagger.json"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "ELVIS Trading Bot API",
        "deepLinking": True,
        "displayOperationId": True,
        "defaultModelsExpandDepth": 1,
        "defaultModelExpandDepth": 1,
        "defaultModelRendering": "example",
        "displayRequestDuration": True,
        "docExpansion": "none",
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "supportedSubmitMethods": ["get", "post", "put", "delete", "patch"],
        "validatorUrl": None,
    },
)


def get_swagger_spec():
    """Generate OpenAPI/Swagger specification"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "ELVIS Trading Bot API",
            "version": "1.0.0",
            "description": "REST API for controlling and monitoring the ELVIS algorithmic trading bot",
            "contact": {"name": "ELVIS Support", "email": "support@elvis-trading.com"},
            "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        },
        "servers": [
            {"url": "http://localhost:5000", "description": "Development server"},
            {
                "url": "https://api.elvis-trading.com",
                "description": "Production server",
            },
        ],
        "tags": [
            {"name": "Authentication", "description": "Authentication endpoints"},
            {"name": "Bot Control", "description": "Bot control and status endpoints"},
            {"name": "Trading", "description": "Trading data and operations"},
            {"name": "Market Data", "description": "Market data endpoints"},
            {"name": "Configuration", "description": "Configuration management"},
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health Check",
                    "description": "Check if the API is running and healthy",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/auth/login": {
                "post": {
                    "summary": "Login",
                    "description": "Authenticate and receive JWT token",
                    "tags": ["Authentication"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LoginRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/LoginResponse"
                                    }
                                }
                            },
                        },
                        "401": {"description": "Invalid credentials"},
                    },
                }
            },
            "/api/bot/status": {
                "get": {
                    "summary": "Get Bot Status",
                    "description": "Get current bot status and information",
                    "tags": ["Bot Control"],
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Bot status retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/BotStatus"}
                                }
                            },
                        },
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/api/bot/start": {
                "post": {
                    "summary": "Start Bot",
                    "description": "Start the trading bot",
                    "tags": ["Bot Control"],
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/StartBotRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Bot started successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/StartBotResponse"
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bot already running"},
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/api/bot/stop": {
                "post": {
                    "summary": "Stop Bot",
                    "description": "Stop the trading bot",
                    "tags": ["Bot Control"],
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {"description": "Bot stopped successfully"},
                        "400": {"description": "Bot not running"},
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/api/account/balance": {
                "get": {
                    "summary": "Get Account Balance",
                    "description": "Get current account balance",
                    "tags": ["Trading"],
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Balance retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AccountBalance"
                                    }
                                }
                            },
                        },
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/api/positions": {
                "get": {
                    "summary": "Get Open Positions",
                    "description": "Get all open trading positions",
                    "tags": ["Trading"],
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Positions retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/Position"
                                        },
                                    }
                                }
                            },
                        },
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/api/trades/history": {
                "get": {
                    "summary": "Get Trade History",
                    "description": "Get historical trades",
                    "tags": ["Trading"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Number of trades to return",
                            "schema": {"type": "integer", "default": 50},
                        },
                        {
                            "name": "start_date",
                            "in": "query",
                            "description": "Start date filter",
                            "schema": {"type": "string", "format": "date-time"},
                        },
                        {
                            "name": "end_date",
                            "in": "query",
                            "description": "End date filter",
                            "schema": {"type": "string", "format": "date-time"},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Trade history retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/TradeHistoryResponse"
                                    }
                                }
                            },
                        },
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/api/performance/stats": {
                "get": {
                    "summary": "Get Performance Statistics",
                    "description": "Get trading performance statistics",
                    "tags": ["Trading"],
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Statistics retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PerformanceStats"
                                    }
                                }
                            },
                        },
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/api/market/price/{symbol}": {
                "get": {
                    "summary": "Get Market Price",
                    "description": "Get current market price for a symbol",
                    "tags": ["Market Data"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {
                            "name": "symbol",
                            "in": "path",
                            "required": True,
                            "description": "Trading symbol (e.g., BTCUSDT)",
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Price retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/MarketPrice"
                                    }
                                }
                            },
                        },
                        "404": {"description": "Symbol not found"},
                    },
                }
            },
            "/api/market/indicators/{symbol}": {
                "get": {
                    "summary": "Get Market Indicators",
                    "description": "Get technical indicators for a symbol",
                    "tags": ["Market Data"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {
                            "name": "symbol",
                            "in": "path",
                            "required": True,
                            "description": "Trading symbol",
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Indicators retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/MarketIndicators"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/config": {
                "get": {
                    "summary": "Get Configuration",
                    "description": "Get current bot configuration",
                    "tags": ["Configuration"],
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Configuration retrieved",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/BotConfiguration"
                                    }
                                }
                            },
                        }
                    },
                },
                "put": {
                    "summary": "Update Configuration",
                    "description": "Update bot configuration",
                    "tags": ["Configuration"],
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/BotConfiguration"
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Configuration updated"},
                        "400": {"description": "Invalid configuration"},
                    },
                },
            },
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                }
            },
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "healthy"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "version": {"type": "string", "example": "1.0.0"},
                    },
                },
                "LoginRequest": {
                    "type": "object",
                    "required": ["username", "password"],
                    "properties": {
                        "username": {"type": "string", "example": "admin"},
                        "password": {"type": "string", "format": "password"},
                    },
                },
                "LoginResponse": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "string", "description": "JWT token"},
                        "expires_in": {
                            "type": "integer",
                            "description": "Token expiration time in seconds",
                        },
                    },
                },
                "BotStatus": {
                    "type": "object",
                    "properties": {
                        "running": {"type": "boolean"},
                        "mode": {"type": "string", "enum": ["paper", "live"]},
                        "start_time": {"type": "string", "format": "date-time"},
                        "strategy": {"type": "string"},
                        "uptime": {"type": "string"},
                    },
                },
                "StartBotRequest": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["paper", "live"],
                            "default": "paper",
                        },
                        "strategy": {"type": "string", "default": "ensemble"},
                    },
                },
                "StartBotResponse": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "mode": {"type": "string"},
                        "strategy": {"type": "string"},
                        "start_time": {"type": "string", "format": "date-time"},
                    },
                },
                "AccountBalance": {
                    "type": "object",
                    "properties": {
                        "total_balance": {"type": "number", "example": 10000},
                        "available_balance": {"type": "number", "example": 8500},
                        "in_position": {"type": "number", "example": 1500},
                        "currency": {"type": "string", "example": "USDT"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "Position": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "example": "BTCUSDT"},
                        "side": {"type": "string", "enum": ["LONG", "SHORT"]},
                        "quantity": {"type": "number", "example": 0.05},
                        "entry_price": {"type": "number", "example": 50000},
                        "current_price": {"type": "number", "example": 51000},
                        "pnl": {"type": "number", "example": 50},
                        "pnl_percentage": {"type": "number", "example": 2.0},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "Trade": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "symbol": {"type": "string"},
                        "side": {"type": "string"},
                        "price": {"type": "number"},
                        "quantity": {"type": "number"},
                        "pnl": {"type": "number"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "TradeHistoryResponse": {
                    "type": "object",
                    "properties": {
                        "trades": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/Trade"},
                        },
                        "count": {"type": "integer"},
                        "total": {"type": "integer"},
                    },
                },
                "PerformanceStats": {
                    "type": "object",
                    "properties": {
                        "total_trades": {"type": "integer"},
                        "winning_trades": {"type": "integer"},
                        "losing_trades": {"type": "integer"},
                        "win_rate": {"type": "number"},
                        "total_pnl": {"type": "number"},
                        "average_pnl": {"type": "number"},
                        "best_trade": {"type": "number"},
                        "worst_trade": {"type": "number"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "MarketPrice": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "price": {"type": "number"},
                        "source": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "MarketIndicators": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "rsi": {"type": "number"},
                        "macd": {"type": "number"},
                        "signal": {"type": "number"},
                        "sma_20": {"type": "number"},
                        "sma_50": {"type": "number"},
                        "ema_12": {"type": "number"},
                        "ema_26": {"type": "number"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "BotConfiguration": {
                    "type": "object",
                    "properties": {
                        "trading": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "max_position_size": {"type": "number"},
                                "risk_per_trade": {"type": "number"},
                                "stop_loss_pct": {"type": "number"},
                                "take_profit_pct": {"type": "number"},
                            },
                        },
                        "strategy": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "models": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "weights": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                            },
                        },
                        "risk_management": {
                            "type": "object",
                            "properties": {
                                "max_daily_trades": {"type": "integer"},
                                "max_drawdown": {"type": "number"},
                                "position_sizing": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }
