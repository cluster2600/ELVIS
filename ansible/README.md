# ELVIS Trading Bot - Ansible Deployment

This directory contains Ansible playbooks and configuration files to automate the installation and setup of the ELVIS Trading Bot and all its dependencies.

## Quick Start

1. **Install Ansible** (if not already installed):
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ansible
   
   # CentOS/RHEL
   sudo yum install epel-release && sudo yum install ansible
   
   # macOS
   brew install ansible
   
   # Via pip
   pip install ansible
   ```

2. **Run the setup**:
   ```bash
   cd ansible
   chmod +x run_setup.sh
   ./run_setup.sh
   ```

## What Gets Installed

The Ansible playbook automatically installs and configures:

### System Dependencies
- **Python 3.14** - Core runtime environment
- **Build tools** - gcc, make, build-essential
- **Development libraries** - SSL, FFI, HDF5, ATLAS for scientific computing
- **Git** - Version control
- **TA-Lib** - Technical Analysis Library (compiled from source on Linux)

### Database & Caching
- **PostgreSQL** - Database server
- **Redis** - Caching and message broker

### Container Platform
- **Docker** - Container runtime
- **Docker Compose** - Multi-container orchestration

### Web Technologies
- **Node.js 18** - installed for tooling and any future web UI (the bot itself
  uses the curses console dashboard + Grafana, not a bundled web app)

### Python Environment
- **Virtual environment** - Isolated Python environment
- **All Python dependencies** - From requirements.txt, requirements_coreml.txt, requirements_ydf.txt

### Services & Monitoring
- **Systemd service** - Auto-start ELVIS bot
- **Prometheus** - Metrics collection
- **Grafana** - Monitoring dashboards
- **Redis monitoring** - Performance tracking

## File Structure

```
ansible/
├── playbook.yml              # Main Ansible playbook
├── inventory.yml             # Host inventory and variables
├── ansible.cfg               # Ansible configuration
├── requirements.yml          # Ansible Galaxy dependencies
├── run_setup.sh             # Setup script
├── templates/
│   └── elvis-bot.service.j2  # Systemd service template
└── README.md                # This file
```

## Usage Examples

### Basic Installation (localhost)
```bash
./run_setup.sh
```

### Test Connection Only
```bash
./run_setup.sh --test
```

### Dry Run (Check Mode)
```bash
./run_setup.sh --check
```

### Skip Ansible Galaxy Installation
```bash
./run_setup.sh --skip-galaxy
```

### Install on Staging Environment
```bash
./run_setup.sh staging
```

### Install on Production Environment
```bash
./run_setup.sh production
```

## Environment Configuration

### Development (localhost)
- Default environment
- Debug mode enabled
- Local installation

### Staging
- Remote servers
- SSL enabled
- Reduced logging

### Production
- Multiple servers supported
- SSL enforced
- Monitoring enabled
- Backup enabled
- Minimal logging

## Customization

### Environment Variables
Edit `inventory.yml` to customize:
- Python version
- Node.js version
- Project directories
- Port configurations
- Security settings

### Custom Variables
Create environment-specific variable files:
```bash
mkdir -p vars
cat > vars/production.yml << EOF
---
debug_mode: false
log_level: "ERROR"
enable_ssl: true
backup_retention_days: 30
EOF
```

### Host Configuration
Update `inventory.yml` to add your servers:
```yaml
production:
  hosts:
    prod-server-1:
      ansible_host: "your-server-ip"
      ansible_user: "deploy"
      ansible_ssh_private_key_file: "~/.ssh/id_rsa"
```

## Platform-Specific Notes

### Ubuntu/Debian
- Uses APT package manager
- Installs from official repositories
- TA-Lib compiled from source

### CentOS/RHEL
- Uses YUM/DNF package manager
- May require EPEL repository
- PostgreSQL initialization included

### macOS
- Uses Homebrew package manager
- Pre-compiled TA-Lib available
- Docker Desktop installation

## Post-Installation Steps

After running the Ansible playbook:

1. **Configure Environment Variables**:
   ```bash
   cd /path/to/elvis-trading-bot
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Test Installation**:
   ```bash
   source venv/bin/activate
   python -m pytest tests/
   ```

3. **Start Services**:
   ```bash
   # Start ELVIS bot
   sudo systemctl start elvis-bot
   sudo systemctl enable elvis-bot
   
   # Or run manually
   ./run_elvis.sh
   ```

4. **Access Web Interfaces**:
   - Grafana: http://localhost:3001 (admin/admin)
   - Trade History API health: http://localhost:5050/health
   - Prometheus: http://localhost:9090

## Troubleshooting

### Common Issues

1. **Permission Denied**:
   ```bash
   chmod +x run_setup.sh
   ```

2. **Ansible Not Found**:
   ```bash
   pip install ansible
   # or
   sudo apt install ansible
   ```

3. **TA-Lib Compilation Errors**:
   ```bash
   # Install build dependencies
   sudo apt install build-essential
   ```

4. **Docker Permission Issues**:
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

5. **Python Virtual Environment Issues**:
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3 -m venv venv
   ```

### Logs and Debugging

- Ansible logs: `./ansible.log`
- Systemd service logs: `journalctl -u elvis-bot`
- Application logs: `/path/to/project/logs/`

## Security Considerations

The playbook includes several security measures:

- **Service isolation** - Systemd security settings
- **File permissions** - Restricted access to sensitive files
- **Firewall rules** - Only necessary ports opened
- **User separation** - Services run as non-root user
- **Environment variables** - Secure credential storage

## Contributing

To modify the Ansible setup:

1. Edit `playbook.yml` for main installation logic
2. Update `inventory.yml` for host and variable configuration
3. Modify `templates/` for service configurations
4. Test changes with `--check` mode first

## Support

For issues related to:
- **Ansible setup**: Check this README and troubleshooting section
- **ELVIS bot**: See main project documentation
- **Dependencies**: Consult individual package documentation

## License

This Ansible configuration is part of the ELVIS Trading Bot project and follows the same license terms.
