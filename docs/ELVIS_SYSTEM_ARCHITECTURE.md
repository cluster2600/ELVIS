# ELVIS System Architecture - Container Relationships & Port Mapping

## 🏗️ Complete System Architecture

```mermaid
graph TB
    subgraph "External Services"
        EXT1[Binance API<br/>WebSocket & REST<br/>Futures Testnet]
        EXT2[GitHub Repository<br/>Source Code & Config]
    end
    
    subgraph "Host System (macOS)"
        HOST1[ELVIS Trading Bot<br/>Native Python Process<br/>PID: Dynamic]
        HOST2[Log Files<br/>elvis_enhanced_metrics.log]
        HOST3[Configuration Files<br/>prometheus.yml, docker-compose.yml]
    end
    
    subgraph "Docker Network: elvis-network"
        subgraph "Database Layer"
            DB1[PostgreSQL Container<br/>elvis-postgres<br/>Host 5433 → 5432<br/>Volume: postgres-data]
            DB2[Redis Container<br/>elvis-redis<br/>Host 6380 → 6379<br/>Volume: redis-data]
        end
        
        subgraph "Monitoring Core (profile: observability)"
            MON1[Prometheus Container<br/>elvis-prometheus<br/>Port: 9090<br/>Volume: prometheus-data]
            MON2[Grafana Container<br/>elvis-grafana<br/>Port: 3001→3000<br/>Volume: grafana-data]
        end
        
        subgraph "Logging Stack (profile: observability)"
            LOG1[Loki Container<br/>elvis-loki<br/>Port: 3100<br/>Volume: loki-data]
            LOG2[Promtail Container<br/>elvis-promtail<br/>No exposed ports<br/>Log shipping agent]
        end
    end
    
    subgraph "User Access Points"
        USER1[Web Browser<br/>Grafana Dashboard<br/>http://localhost:3001]
        USER2[API Client<br/>Prometheus Query<br/>http://localhost:9090]
        USER3[Trading API<br/>ELVIS Endpoints<br/>http://localhost:5050]
    end
    
    %% External Connections
    HOST1 -.->|WebSocket/HTTPS| EXT1
    EXT2 -.->|Git Clone/Pull| HOST3
    
    %% Host to Container Connections
    HOST1 -->|Database Queries<br/>Host port: 5433| DB1
    HOST1 -->|Cache Operations<br/>Host port: 6380| DB2
    HOST1 -->|Metrics Export<br/>Port: 5050/metrics| MON1
    HOST2 -->|Log File Mount<br/>Read-only| LOG2
    HOST3 -->|Config Mount<br/>Read-only| MON1
    HOST3 -->|Config Mount<br/>Read-only| MON2
    
    %% Inter-Container Communications
    MON1 -->|PromQL Queries<br/>HTTP| MON2
    LOG1 -->|LogQL Queries<br/>HTTP| MON2
    LOG2 -->|Log Shipping<br/>HTTP Push| LOG1
    
    %% User Access
    USER1 -->|HTTPS<br/>Port: 3001| MON2
    USER2 -->|HTTP API<br/>Port: 9090| MON1
    USER3 -->|HTTP API<br/>Port: 5050| HOST1
    
    %% Data Flow Annotations
    MON1 -.->|Scrapes every 10s<br/>host.docker.internal:5050| HOST1
    LOG2 -.->|Tails continuously<br/>/var/log/elvis/elvis.log| HOST2
    
    classDef hostProcess fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef container fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef database fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef monitoring fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef logging fill:#fcf4ff,stroke:#6a1b9a,stroke-width:2px
    classDef external fill:#fafafa,stroke:#424242,stroke-width:2px
    classDef user fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    
    class HOST1,HOST2,HOST3 hostProcess
    class DB1,DB2 database
    class MON1,MON2 monitoring
    class LOG1,LOG2 logging
    class EXT1,EXT2 external
    class USER1,USER2,USER3 user
```

> **Monitoring is not a standalone service.** The Prometheus/Grafana/Loki/Promtail
> containers above are optional and only start with the `observability` Compose
> profile (`docker compose --profile observability up`). The bot's actual
> monitoring code lives in three places:
> - **`trading/utils/trade_history_api.py`** — the Flask trade-history API on port
>   `5050`; exposes `GET /metrics` (Prometheus format, via `prometheus_client` /
>   `prometheus_flask_exporter`) and `GET /health`. This is the target Prometheus
>   scrapes at `host.docker.internal:5050`.
> - **`utils/api_connection_tester.py`** — `APIConnectionTester` runs per-service
>   health checks (Binance spot/futures/testnet, PostgreSQL, Redis, Vault,
>   Telegram, Prometheus) and rolls them up via `get_overall_health()`.
> - **`core/metrics/performance_monitor.py`** — `PerformanceMonitor` tracks
>   trading performance (rolling Sharpe, drawdown).

## 🔌 Detailed Port Mapping & Network Configuration

```mermaid
graph LR
    subgraph "Host Ports (localhost)"
        H1[5433<br/>PostgreSQL]
        H2[6380<br/>Redis]
        H3[9090<br/>Prometheus]
        H4[3001<br/>Grafana]
        H5[3100<br/>Loki]
        H6[5050<br/>ELVIS API]
    end
    
    subgraph "Container Internal Ports"
        C1[5432<br/>postgres:5432]
        C2[6379<br/>redis:6379]
        C3[9090<br/>prometheus:9090]
        C4[3000<br/>grafana:3000]
        C5[3100<br/>loki:3100]
        C6[9080<br/>promtail:9080]
    end
    
    subgraph "Internal Container Network"
        N1[elvis-network<br/>Bridge Driver<br/>Subnet: Auto-assigned]
    end
    
    H1 -.-> C1
    H2 -.-> C2
    H3 -.-> C3
    H4 -.-> C4
    H5 -.-> C5
    
    C1 --> N1
    C2 --> N1
    C3 --> N1
    C4 --> N1
    C5 --> N1
    C6 --> N1
    
    classDef hostPort fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef containerPort fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef network fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class H1,H2,H3,H4,H5,H6 hostPort
    class C1,C2,C3,C4,C5,C6 containerPort
    class N1 network
```

## 📊 Data Flow & Communication Patterns

```mermaid
sequenceDiagram
    participant User as 👤 User Browser
    participant Grafana as 📊 Grafana
    participant Prometheus as 📈 Prometheus
    participant ELVIS as 🤖 ELVIS Bot
    participant Loki as 📝 Loki
    participant Promtail as 🚚 Promtail
    participant DB as 🗄️ PostgreSQL
    
    Note over User,DB: Dashboard Access Flow
    User->>Grafana: HTTP Request (port 3001)
    Grafana->>Prometheus: PromQL Query (port 9090)
    Prometheus->>ELVIS: Scrape /metrics (port 5050)
    ELVIS->>DB: Query trading data (host port 5433)
    DB-->>ELVIS: Trade/position data
    ELVIS-->>Prometheus: Metrics response
    Prometheus-->>Grafana: Query results
    Grafana-->>User: Dashboard data
    
    Note over User,DB: Log Access Flow
    ELVIS->>Promtail: Write to log file
    Promtail->>Loki: Ship logs (port 3100)
    User->>Grafana: View logs panel
    Grafana->>Loki: LogQL Query
    Loki-->>Grafana: Log entries
    Grafana-->>User: Log display
    
    Note over User,DB: Real-time Updates
    loop Every 10 seconds
        Prometheus->>ELVIS: Scrape metrics
        ELVIS->>DB: Fresh data query
    end
    
    loop Every 2-5 seconds
        User->>Grafana: Dashboard refresh
    end
```

## 🏗️ Container Dependencies & Startup Order

```mermaid
graph TD
    START([Docker Compose Start]) --> INFRA[Infrastructure Layer]
    
    subgraph INFRA[Infrastructure Containers]
        A[PostgreSQL<br/>Health Check: pg_isready<br/>Timeout: 30s]
        B[Redis<br/>Basic startup<br/>No health check]
    end
    
    INFRA --> MONITOR[Monitoring Layer]
    
    subgraph MONITOR[Monitoring Containers]
        C[Prometheus<br/>Depends: None<br/>Config: observability/prometheus.yml]
        D[Loki<br/>Depends: None<br/>Config: loki/config.yml]
    end
    
    MONITOR --> VISUAL[Visualization Layer]
    
    subgraph VISUAL[Visualization & Collection]
        E[Grafana<br/>Depends: Loki<br/>Data sources: Auto-provision]
        F[Promtail<br/>Depends: Loki<br/>Config: promtail/config.yml]
    end
    
    VISUAL --> APP[Application Layer]
    
    subgraph APP[Application Services]
        G[ELVIS Bot<br/>Depends: PostgreSQL, Redis<br/>Health Check: /health]
    end
    
    APP --> READY[🚀 System Ready]
    
    classDef infrastructure fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef monitoring fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef visualization fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef application fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef ready fill:#e0f2f1,stroke:#00695c,stroke-width:3px
    
    class A,B infrastructure
    class C,D monitoring
    class E,F visualization
    class G application
    class READY ready
```

## 🔒 Network Security & Access Control

```mermaid
graph TB
    subgraph "Public Internet"
        INT[Internet Access<br/>Binance API Only]
    end
    
    subgraph "Host Network (macOS)"
        HOST[Host Interface<br/>127.0.0.1 & 10.x.x.x]
    end
    
    subgraph "Docker Bridge Network"
        subgraph "elvis-network (Isolated)"
            NET[Internal Container DNS<br/>Service Discovery<br/>172.x.x.x subnet]
        end
    end
    
    subgraph "Access Control Matrix"
        PUB[Published Ports (bound to host)<br/>🔓 3001: Grafana UI<br/>🔓 9090: Prometheus API<br/>🔓 5050: ELVIS API<br/>🔓 3100: Loki API<br/>🔓 5433→5432: PostgreSQL<br/>🔓 6380→6379: Redis]
        INT_CONT[Container-to-Container<br/>🔒 Service DNS on elvis-network<br/>🔒 postgres:5432, redis:6379]
    end
    
    INT -.->|HTTPS/WSS| HOST
    HOST --> PUB
    HOST -.->|Port Mapping| NET
    NET --> INT_CONT
    
    classDef public fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    classDef internal fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef network fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    
    class INT,PUB public
    class INT_CONT internal
    class HOST,NET network
```

## 📦 Volume Management & Data Persistence

```mermaid
graph TB
    subgraph "Host File System"
        HOST_CONFIG[Configuration Files<br/>./prometheus.yml<br/>./docker-compose.yml<br/>./grafana/provisioning/]
        HOST_LOGS[Log Files<br/>./elvis_enhanced_metrics.log<br/>./logs/]
        HOST_DATA[Application Data<br/>./data/<br/>./models/]
    end
    
    subgraph "Docker Volumes (Persistent)"
        VOL_PG[postgres-data<br/>Database storage<br/>~500MB]
        VOL_REDIS[redis-data<br/>Cache persistence<br/>~50MB]
        VOL_PROM[prometheus-data<br/>Metrics storage<br/>~1GB]
        VOL_GRAF[grafana-data<br/>Dashboard configs<br/>~100MB]
        VOL_LOKI[loki-data<br/>Log storage<br/>~200MB]
    end
    
    subgraph "Container Mount Points"
        MOUNT_CONFIG[/etc/prometheus/<br/>/etc/grafana/provisioning/<br/>/etc/loki/]
        MOUNT_DATA[/var/lib/postgresql/data<br/>/data<br/>/prometheus<br/>/var/lib/grafana<br/>/loki]
        MOUNT_LOGS[/var/log/elvis/<br/>Read-only mounts]
    end
    
    HOST_CONFIG -.->|Bind Mount<br/>Read-only| MOUNT_CONFIG
    HOST_LOGS -.->|Bind Mount<br/>Read-only| MOUNT_LOGS
    HOST_DATA -.->|Bind Mount<br/>Read-write| MOUNT_DATA
    
    VOL_PG --> MOUNT_DATA
    VOL_REDIS --> MOUNT_DATA
    VOL_PROM --> MOUNT_DATA
    VOL_GRAF --> MOUNT_DATA
    VOL_LOKI --> MOUNT_DATA
    
    classDef hostData fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef dockerVol fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef mountPoint fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    
    class HOST_CONFIG,HOST_LOGS,HOST_DATA hostData
    class VOL_PG,VOL_REDIS,VOL_PROM,VOL_GRAF,VOL_LOKI dockerVol
    class MOUNT_CONFIG,MOUNT_DATA,MOUNT_LOGS mountPoint
```

## 🚀 Service Health & Monitoring

```mermaid
graph TB
    subgraph "Health Check Matrix"
        subgraph "Database Health"
            H1[PostgreSQL<br/>✅ pg_isready -U elvis_user<br/>⏱️ 10s interval, 5s timeout]
            H2[Redis<br/>✅ Built-in health<br/>⏱️ No custom check]
        end
        
        subgraph "Application Health"
            H3[ELVIS Bot<br/>✅ HTTP /health endpoint<br/>⏱️ 30s interval, 10s timeout<br/>🔄 3 retries, 60s start period]
        end
        
        subgraph "Monitoring Health"
            H4[Prometheus<br/>✅ HTTP /-/healthy<br/>⏱️ Self-monitoring]
            H5[Grafana<br/>✅ HTTP /api/health<br/>⏱️ Built-in health]
            H6[Loki<br/>✅ HTTP /ready<br/>⏱️ Readiness probe]
        end
    end
    
    subgraph "Dependency Chain"
        H1 --> H3
        H2 --> H3
        H4 --> H5
        H6 --> H5
    end
    
    subgraph "Alert Conditions"
        ALERT1[🚨 Database Unavailable<br/>PostgreSQL connection failed]
        ALERT2[🚨 Trading System Down<br/>ELVIS /health returning 500]
        ALERT3[🚨 Metrics Collection Failed<br/>Prometheus scrape errors]
        ALERT4[🚨 Dashboard Unreachable<br/>Grafana HTTP errors]
    end
    
    H1 -.->|Health Fail| ALERT1
    H3 -.->|Health Fail| ALERT2
    H4 -.->|Scrape Fail| ALERT3
    H5 -.->|HTTP Error| ALERT4
    
    classDef database fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef application fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef monitoring fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef alert fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    
    class H1,H2 database
    class H3 application
    class H4,H5,H6 monitoring
    class ALERT1,ALERT2,ALERT3,ALERT4 alert
```

---

**Architecture Version**: 2.0  
**Last Updated**: August 4, 2025  
**Container Runtime**: Docker 24.x with Compose v2  
**Network Driver**: Bridge (default)  
**Volume Driver**: Local filesystem