# Prometheus and Grafana Integration for ELVIS

This document provides detailed instructions for setting up and configuring Prometheus and Grafana to monitor the ELVIS trading system. It includes configurations for both services, dashboard setup, and troubleshooting guidance.

## Table of Contents
- [Prometheus Configuration](#prometheus-configuration)
- [Grafana Configuration](#grafana-configuration)
- [Dashboard Provisioning](#dashboard-provisioning)
- [Available Dashboards](#available-dashboards)
- [Setup Instructions](#setup-instructions)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Prometheus Configuration

The ELVIS system uses Prometheus to collect and store metrics. Below is the actual Prometheus configuration file included in the repository:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  - job_name: 'elvis'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scheme: 'http'
```

### Configuration Details:

- **global**: Contains global configuration settings
  - `scrape_interval: 15s`: Prometheus will collect metrics every 15 seconds

- **scrape_configs**: Contains the list of targets to scrape metrics from
  - **prometheus job**: Scrapes Prometheus itself at localhost:9090
  - **elvis job**: Scrapes metrics from the ELVIS trading system
    - `targets`: The ELVIS system exposes metrics at localhost:8000
    - `metrics_path`: Metrics are available at the /metrics endpoint
    - `scheme`: Uses HTTP protocol to connect to the ELVIS system

### Customization:

To adapt this configuration to your environment, you may need to:

1. Adjust the `targets` if your ELVIS system or Prometheus runs on different hosts/ports
2. Modify the `scrape_interval` based on your monitoring requirements
3. Add authentication if your metrics endpoints are protected

## Grafana Configuration

### Data Sources Configuration

Grafana needs to know where to find metrics. Below is the actual `datasources.yml` file included in the repository:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:8000
    basicAuth: false
```

### Configuration Details:

- `name`: The name of the data source as it appears in Grafana
- `type`: Specifies that this is a Prometheus data source
- `access`: Proxy mode means Grafana will access Prometheus on behalf of the browser
- `url`: The URL where Prometheus can be reached
- `basicAuth`: Authentication is not enabled

### Customization:

You may need to:

1. Change the `url` if Prometheus is running on a different host or port
2. Enable `basicAuth` and provide credentials if your Prometheus instance requires authentication

## Dashboard Provisioning

Grafana can automatically load dashboards from files. Below is the actual `dashboards.yml` file included in the repository:

```yaml
apiVersion: 1
providers:
  - name: 'ELVIS Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /Users/maxime/BTC_BOT/BTC_BOT/grafana/dashboards
```

### Configuration Details:

- `name`: Provider name that will appear in Grafana
- `orgId`: The organization ID in Grafana (default is 1)
- `folder`: Folder where dashboards will be stored (empty means root)
- `type`: Provider type, 'file' means dashboards are loaded from the filesystem
- `disableDeletion`: If false, dashboards deleted from the file system will be deleted from Grafana
- `updateIntervalSeconds`: How often to check for dashboard changes
- `options.path`: Location of dashboard JSON files

### Important Path Adjustment:

⚠️ **Note:** You **must** adjust the path in `options.path` to match your local installation. For example:
- Linux: `/path/to/your/ELVIS/grafana/dashboards`
- macOS: `/Users/username/ELVIS/grafana/dashboards`
- Windows: `C:\\path\\to\\your\\ELVIS\\grafana\\dashboards`

## Available Dashboards

The ELVIS repository includes several pre-configured Grafana dashboards in the `grafana/dashboards/` directory:

1. **elvis_master_trading_dashboard.json**: Complete view of trading metrics, portfolio performance, and system status
2. **elvis_advanced_trading_dashboard.json**: Detailed trading analytics with advanced metrics
3. **elvis_advanced_trading_dashboard_v2.json**: Updated version with additional visualizations
4. **elvis_dashboard_fixed_panels.json**: Standard layout with fixed panel positions
5. **elvis_full_prometheus_dashboard.json**: Focused on Prometheus metrics for system monitoring
6. **elvis-trading.json**: Core trading metrics dashboard

These dashboard files are automatically loaded by Grafana if the dashboard provisioning is correctly configured.

### Importing Dashboards Manually:

If you prefer to import dashboards manually:

1. Open Grafana in your browser
2. Go to Dashboards > Import
3. Either upload the JSON file or paste its contents
4. Select the appropriate data source
5. Click Import

## Setup Instructions

### Prerequisites:
- Prometheus installed ([Download Prometheus](https://prometheus.io/download/))
- Grafana installed ([Download Grafana](https://grafana.com/grafana/download))
- ELVIS trading system running with metrics enabled

### Installation Steps:

1. **Configure Prometheus:**
   - Copy the provided `prometheus.yml` to your Prometheus configuration directory
   - Adjust targets as needed for your environment

2. **Configure Grafana:**
   - Create a directory for provisioning: `mkdir -p /etc/grafana/provisioning/datasources /etc/grafana/provisioning/dashboards`
   - Copy `datasources.yml` to `/etc/grafana/provisioning/datasources/`
   - Copy `dashboards.yml` to `/etc/grafana/provisioning/dashboards/`
   - **Important:** Update the path in `dashboards.yml` to point to your dashboards directory
   - Copy the dashboard JSON files to the directory specified in `dashboards.yml`

3. **Start the Services:**
   - Start Prometheus: `prometheus --config.file=prometheus.yml`
   - Start Grafana: `grafana-server`

4. **Access Grafana:**
   - Open your browser and navigate to `http://localhost:3000` (default Grafana port)
   - Default login: admin/admin

## Troubleshooting

### Common Issues:

1. **No Data in Grafana**
   - Verify Prometheus is running and collecting metrics
   - Check that the data source URL in Grafana is correct
   - Verify ELVIS is exposing metrics at the expected endpoint

2. **Dashboards Not Loading**
   - Check the path in `dashboards.yml` is correct for your environment
   - Verify dashboard JSON files exist in the specified path
   - Check Grafana server logs for provisioning errors

3. **Connection Refused Errors**
   - Verify services are running on the expected ports
   - Check firewall settings allow connections to these ports

### Prometheus Logs:

Check Prometheus logs for scraping errors:
```
prometheus --config.file=prometheus.yml --log.level=debug
```

## Best Practices

### Performance:

1. Adjust `scrape_interval` based on your needs - shorter intervals give better resolution but increased storage requirements
2. Consider retention periods in Prometheus based on your disk space constraints
3. Use dashboard variables to dynamically filter data and improve performance

### Security:

1. Use basic authentication or other security measures in production environments
2. Run Prometheus and Grafana behind a reverse proxy with HTTPS
3. Use restrictive file permissions for configuration files containing credentials

### Maintenance:

1. Regularly update Prometheus and Grafana to their latest versions
2. Back up dashboard JSON files and configurations
3. Monitor the monitoring system with alerts for Prometheus or Grafana downtime

---

This documentation reflects the actual configuration files provided in the ELVIS repository. Adjust paths and URLs according to your specific installation environment.