kaleidoscope_) 2>/dev/null || true
    endscript
}
EOF

# Configure systemd service
cat > /etc/systemd/system/kaleidoscope.service << EOF
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
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Setup monitoring and maintenance scripts
mkdir -p /opt/kaleidoscope/scripts

cat > /opt/kaleidoscope/scripts/health_check.sh << 'EOF'
#!/bin/bash

# Health check endpoints
ENDPOINTS=(
    "http://localhost:8000/health"
    "http://localhost:9090/-/healthy"
    "http://localhost:3000/api/health"
    "http://localhost:6379/ping"
)

# Check each endpoint
for endpoint in "${ENDPOINTS[@]}"; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint")
    if [ "$response" != "200" ]; then
        echo "Service $endpoint is down!"
        docker-compose restart $(echo "$endpoint" | cut -d'/' -f3)
    fi
done
EOF

cat > /opt/kaleidoscope/scripts/backup.sh << 'EOF'
#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/kaleidoscope/backups/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

# Backup Redis data
docker exec kaleidoscope_redis_1 redis-cli SAVE
docker cp kaleidoscope_redis_1:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"

# Backup Prometheus data
tar -czf "$BACKUP_DIR/prometheus_data.tar.gz" /opt/kaleidoscope/data/prometheus

# Backup Grafana data
tar -czf "$BACKUP_DIR/grafana_data.tar.gz" /opt/kaleidoscope/data/grafana

# Backup to S3
aws s3 sync "$BACKUP_DIR" "s3://kaleidoscope-backups/$TIMESTAMP/"

# Cleanup old backups (keep last 7 days)
find /opt/kaleidoscope/backups -type d -mtime +7 -exec rm -rf {} +
EOF

cat > /opt/kaleidoscope/scripts/resource_monitor.sh << 'EOF'
#!/bin/bash

# Monitor system resources
MEMORY_THRESHOLD=90
CPU_THRESHOLD=80
DISK_THRESHOLD=90

# Check memory usage
memory_usage=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
if (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
    docker system prune -f
fi

# Check CPU usage
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d. -f1)
if [ "$cpu_usage" -gt "$CPU_THRESHOLD" ]; then
    systemctl restart kaleidoscope
fi

# Check disk usage
disk_usage=$(df / | tail -1 | awk '{print $5}' | cut -d% -f1)
if [ "$disk_usage" -gt "$DISK_THRESHOLD" ]; then
    /opt/kaleidoscope/scripts/cleanup.sh
fi
EOF

cat > /opt/kaleidoscope/scripts/cleanup.sh << 'EOF'
#!/bin/bash

# Cleanup old logs
find /var/log/kaleidoscope -type f -name "*.log.*" -mtime +30 -delete

# Cleanup Docker
docker system prune -af --volumes

# Cleanup old Redis keys
redis-cli SCAN 0 COUNT 1000 | while read -r key; do
    ttl=$(redis-cli TTL "$key")
    if [ "$ttl" -eq -1 ]; then
        redis-cli EXPIRE "$key" 86400  # Set 24h expiry
    fi
done

# Cleanup temporary files
find /tmp -type f -atime +7 -delete
EOF

cat > /opt/kaleidoscope/scripts/auto_scale.sh << 'EOF'
#!/bin/bash

# Get current resource usage
CPU_USAGE=$(docker stats --no-stream --format "{{.CPUPerc}}" kaleidoscope_chatbot_1 | cut -d'%' -f1)
MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" kaleidoscope_chatbot_1 | cut -d'%' -f1)

# Scale up if necessary
if (( $(echo "$CPU_USAGE > 80" | bc -l) )) || (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    CURRENT_REPLICAS=$(docker service ls --format "{{.Replicas}}" kaleidoscope_chatbot)
    NEW_REPLICAS=$((CURRENT_REPLICAS + 1))
    docker service scale kaleidoscope_chatbot=$NEW_REPLICAS
fi

# Scale down if usage is low
if (( $(echo "$CPU_USAGE < 20" | bc -l) )) && (( $(echo "$MEMORY_USAGE < 20" | bc -l) )); then
    CURRENT_REPLICAS=$(docker service ls --format "{{.Replicas}}" kaleidoscope_chatbot)
    if [ "$CURRENT_REPLICAS" -gt 1 ]; then
        NEW_REPLICAS=$((CURRENT_REPLICAS - 1))
        docker service scale kaleidoscope_chatbot=$NEW_REPLICAS
    fi
fi
EOF

# Make scripts executable
chmod +x /opt/kaleidoscope/scripts/*.sh

# Setup cron jobs
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/kaleidoscope/scripts/health_check.sh") | crontab -
(crontab -l 2>/dev/null; echo "0 1 * * * /opt/kaleidoscope/scripts/backup.sh") | crontab -
(crontab -l 2>/dev/null; echo "*/15 * * * * /opt/kaleidoscope/scripts/resource_monitor.sh") | crontab -
(crontab -l 2>/dev/null; echo "*/10 * * * * /opt/kaleidoscope/scripts/auto_scale.sh") | crontab -

# Start services
systemctl daemon-reload
systemctl enable kaleidoscope
systemctl start kaleidoscope

# Setup log forwarding to CloudWatch
cat > /etc/awslogs/awslogs.conf << EOF
[general]
state_file = /var/awslogs/state/agent-state

[/var/log/kaleidoscope]
datetime_format = %Y-%m-%d %H:%M:%S
file = /var/log/kaleidoscope/*.log
buffer_duration = 5000
log_stream_name = {instance_id}
initial_position = start_of_file
log_group_name = /kaleidoscope/production
EOF

# Start logging service
systemctl enable awslogsd
systemctl start awslogsd

echo "✅ Deployment complete! System is running at https://$DOMAIN"