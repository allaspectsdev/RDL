#!/bin/bash
# =============================================================================
# DataLens Deployment Script for Linux Server
#
# Usage:
#   ./scripts/deploy.sh                  # Full Docker deployment
#   ./scripts/deploy.sh --no-docker      # Systemd deployment (no Docker)
#   ./scripts/deploy.sh --ssl DOMAIN     # Set up SSL after initial deploy
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
USE_DOCKER=true
SETUP_SSL=false
DOMAIN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-docker) USE_DOCKER=false; shift ;;
        --ssl)       SETUP_SSL=true; DOMAIN="$2"; shift 2 ;;
        *)           error "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# SSL Setup (standalone mode)
# ---------------------------------------------------------------------------
if $SETUP_SSL; then
    if [ -z "$DOMAIN" ]; then
        error "Domain required: ./scripts/deploy.sh --ssl yourdomain.com"
    fi
    log "Setting up SSL for $DOMAIN ..."

    # Replace domain placeholder in nginx config
    sed -i "s/YOUR_DOMAIN.com/$DOMAIN/g" nginx/datalens.conf
    log "Updated nginx config with domain: $DOMAIN"

    if $USE_DOCKER; then
        # Get initial certificate via Docker
        docker compose run --rm certbot certonly \
            --webroot --webroot-path=/var/www/certbot \
            --email admin@$DOMAIN \
            --agree-tos --no-eff-email \
            -d $DOMAIN -d www.$DOMAIN

        log "SSL certificate obtained. Enabling HTTPS in nginx config..."

        # Enable HTTPS block and HTTP redirect in nginx config
        # Uncomment the HTTPS server block and HTTP redirect
        sed -i 's/^# \(.*listen 443\)/\1/' nginx/datalens.conf
        sed -i 's/^# \(.*ssl_\)/\1/' nginx/datalens.conf
        sed -i 's/^# \(.*add_header Strict\)/\1/' nginx/datalens.conf

        docker compose restart nginx
        log "SSL enabled! Site available at https://$DOMAIN"
    else
        # Certbot standalone
        sudo certbot certonly --webroot -w /var/www/certbot \
            --email admin@$DOMAIN \
            --agree-tos --no-eff-email \
            -d $DOMAIN -d www.$DOMAIN

        sudo systemctl restart nginx
        log "SSL enabled! Site available at https://$DOMAIN"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# System Prerequisites
# ---------------------------------------------------------------------------
log "Checking prerequisites..."

if $USE_DOCKER; then
    # Docker deployment
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker $USER
        warn "Added $USER to docker group. You may need to log out/in."
    fi

    if ! docker compose version &> /dev/null; then
        log "Installing Docker Compose plugin..."
        sudo apt-get update && sudo apt-get install -y docker-compose-plugin
    fi

    log "Docker is ready."
else
    # Systemd deployment — install system packages
    log "Installing system packages..."
    sudo apt-get update
    sudo apt-get install -y \
        python3 python3-pip python3-venv \
        nginx certbot python3-certbot-nginx \
        curl git

    log "System packages installed."
fi

# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------
if command -v ufw &> /dev/null; then
    log "Configuring firewall..."
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw --force enable
    log "Firewall configured (ports 80, 443 open)."
fi

# ---------------------------------------------------------------------------
# Docker Deployment
# ---------------------------------------------------------------------------
if $USE_DOCKER; then
    log "Building and starting Docker containers..."

    docker compose build --no-cache
    docker compose up -d

    log "Waiting for app to start..."
    sleep 5

    if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        log "DataLens is running!"
    else
        warn "App may still be starting. Check: docker compose logs datalens"
    fi

    echo ""
    log "========================================="
    log " DataLens deployed successfully!"
    log " HTTP:  http://$(hostname -I | awk '{print $1}')"
    log ""
    log " Next steps:"
    log "   1. Point your domain DNS to this server IP"
    log "   2. Edit nginx/datalens.conf — replace YOUR_DOMAIN.com"
    log "   3. Run: ./scripts/deploy.sh --ssl yourdomain.com"
    log "========================================="
    exit 0
fi

# ---------------------------------------------------------------------------
# Systemd Deployment (no Docker)
# ---------------------------------------------------------------------------
APP_DIR="/opt/datalens"
VENV_DIR="$APP_DIR/venv"

log "Setting up application at $APP_DIR ..."

# Copy application files
sudo mkdir -p "$APP_DIR"
sudo cp -r . "$APP_DIR/"
sudo chown -R www-data:www-data "$APP_DIR"

# Create virtual environment
log "Creating Python virtual environment..."
sudo -u www-data python3 -m venv "$VENV_DIR"
sudo -u www-data "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u www-data "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

log "Python dependencies installed."

# Create systemd service
log "Creating systemd service..."
sudo tee /etc/systemd/system/datalens.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=DataLens Visual Data Analysis Tool
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/datalens
Environment="PATH=/opt/datalens/venv/bin:/usr/bin:/bin"
ExecStart=/opt/datalens/venv/bin/streamlit run app.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=datalens

# Security hardening
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/opt/datalens

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Configure nginx
log "Configuring nginx..."
sudo cp "$APP_DIR/nginx/datalens.conf" /etc/nginx/sites-available/datalens.conf

# Adjust upstream for non-Docker (localhost instead of container name)
sudo sed -i 's/server datalens:8501/server 127.0.0.1:8501/' /etc/nginx/sites-available/datalens.conf

sudo ln -sf /etc/nginx/sites-available/datalens.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
sudo nginx -t || error "Nginx configuration test failed!"

# Start services
log "Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable datalens
sudo systemctl start datalens
sudo systemctl restart nginx

log "Waiting for app to start..."
sleep 5

if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    log "DataLens is running!"
else
    warn "App may still be starting. Check: sudo journalctl -u datalens -f"
fi

echo ""
log "========================================="
log " DataLens deployed successfully!"
log " HTTP:  http://$(hostname -I | awk '{print $1}')"
log ""
log " Next steps:"
log "   1. Point your domain DNS to this server IP"
log "   2. Edit /etc/nginx/sites-available/datalens.conf"
log "      Replace YOUR_DOMAIN.com with your domain"
log "   3. Run: ./scripts/deploy.sh --ssl yourdomain.com"
log ""
log " Management commands:"
log "   sudo systemctl status datalens"
log "   sudo systemctl restart datalens"
log "   sudo journalctl -u datalens -f"
log "========================================="
