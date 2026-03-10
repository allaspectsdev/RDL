#!/bin/bash
# =============================================================================
# Ryan's Data Lab (RDL) Update Script — pull latest code and restart
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[UPDATE]${NC} $1"; }

APP_DIR="/opt/rdl"

# Detect deployment type
if docker compose version &> /dev/null && docker ps --format '{{.Names}}' | grep -q rdl-app; then
    log "Docker deployment detected."

    log "Pulling latest code..."
    git pull origin main

    log "Rebuilding container..."
    docker compose build --no-cache
    docker compose up -d

    log "Updated! Checking health..."
    sleep 5
    if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        log "Ryan's Data Lab is healthy."
    fi

elif systemctl is-active --quiet rdl; then
    log "Systemd deployment detected."

    log "Pulling latest code..."
    cd "$APP_DIR"
    sudo -u www-data git pull origin main

    log "Updating dependencies..."
    sudo -u www-data "$APP_DIR/venv/bin/pip" install -r requirements.txt

    log "Restarting service..."
    sudo systemctl restart rdl

    sleep 3
    if systemctl is-active --quiet rdl; then
        log "Ryan's Data Lab restarted successfully."
    fi
else
    echo "No running Ryan's Data Lab deployment found."
    exit 1
fi
