#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting Kaleidoscope AI System deployment...${NC}"

# Install system dependencies
sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-pip \
    docker.io \
    docker-compose \
    nginx \
    certbot \
    python3-certbot-nginx \
    awscli \
    jq \
    git

# Install Python dependencies
pip3 install -r requirements.txt

# Start deployment
python3 - << 'EOF'
import asyncio
from automation import AutomationSystem

async def main():
    automation = AutomationSystem()
    
    print("🚀 Setting up infrastructure...")
    await automation.setup_infrastructure()
    
    print("📦 Deploying application...")
    await automation.deploy_application()
    
    print("🌐 Configuring domain...")
    await automation.configure_domain()
    
    print("📊 Setting up monitoring...")
    await automation.setup_monitoring_stack()

asyncio.run(main())
EOF

# Verify deployment
function check_endpoint() {
    local url=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${BLUE}Checking endpoint: $url${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|301\|302"; then
            echo -e "${GREEN}✓ Endpoint $url is up${NC}"
            return 0
        fi
        echo -e "${RED}Attempt $attempt/$max_attempts: Endpoint not ready${NC}"
        ((attempt++))
        sleep 10
    done
    
    echo -e "${RED}❌ Failed to verify endpoint: $url${NC}"
    return 1
}

# Check all endpoints
endpoints=(
    "https://artificialthinker.com"
    "https://artificialthinker.com/api/health"
    "http://localhost:9090"  # Prometheus
    "http://localhost:3000"  # Grafana
)

for endpoint in "${endpoints[@]}"; do
    check_endpoint "$endpoint"
done

# Setup automatic updates
cat > /etc/cron.daily/kaleidoscope-update << 'EOF'
#!/bin/bash
cd /opt/kaleidoscope
git pull
docker-compose pull
docker-compose up -d
EOF

chmod +x /etc/cron.daily/kaleidoscope-update

# Setup log rotation
cat > /etc/logrotate.d/kaleidoscope << 'EOF'
/var/log/kaleidoscope/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        systemctl restart kaleidoscope
    endscript
}
EOF

# Setup systemd service
cat > /etc/systemd/system/kaleidoscope.service << 'EOF'
[Unit]
Description=Kaleidoscope AI System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/kaleidoscope
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec