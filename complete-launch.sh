# Complete systemd service configuration
cat > /etc/systemd/system/kaleidoscope.service << 'EOF'
[Unit]
Description=Kaleidoscope AI System
After=docker.service network-online.target
Requires=docker.service network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/kaleidoscope
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=AWS_ACCESS_KEY_ID=AKIA4WJPWX757RLAGXU7
Environment=AWS_SECRET_ACCESS_KEY=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
Environment=AWS_DEFAULT_REGION=us-east-2

ExecStartPre=/usr/local/bin/docker-compose pull
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
ExecReload=/usr/local/bin/docker-compose up -d --force-recreate

TimeoutStartSec=0
TimeoutStopSec=180
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Configure system limits
cat > /etc/security/limits.d/kaleidoscope.conf << 'EOF'
*       soft    nofile  65535
*       hard    nofile  65535
*       soft    nproc   65535
*       hard    nproc   65535
root    soft    nofile  65535
root    hard    nofile  65535
root    soft    nproc   65535
root    hard    nproc   65535
EOF

# Configure sysctl
cat > /etc/sysctl.d/99-kaleidoscope.conf << 'EOF'
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_tw_reuse = 1
net.ipv4.ip_local_port_range = 1024 65535
vm.swappiness = 10
vm.dirty_ratio = 60
vm.dirty_background_ratio = 2
fs.file-max = 2097152
net.ipv4.tcp_mtu_probing = 1
EOF

# Apply sysctl settings
sysctl --system

# Setup monitoring
cat > /etc/prometheus/rules/kaleidoscope.yml << 'EOF'
groups:
- name: kaleidoscope
  rules:
  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      description: Memory usage is above 85% for 5 minutes
  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      description: CPU usage is above 85% for 5 minutes
  - alert: DiskSpaceLow
    expr: node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100 < 10
    for: 5m
    labels:
      severity: warning
    annotations:
      description: Disk space is below 10%
EOF

# Setup automatic backup
cat > /opt/kaleidoscope/scripts/backup.sh << 'EOF'
#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/kaleidoscope/backups/$TIMESTAMP"
S3_BUCKET="kaleidoscope-backups"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup Docker volumes
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$BACKUP_DIR":/backup \
  alpine sh -c "docker volume ls -q | xargs -I {} docker run --rm -v {}:/volume -v /backup:/backup alpine tar -czf /backup/{}.tar.gz -C /volume ."

# Backup configurations
tar -czf "$BACKUP_DIR/configs.tar.gz" /opt/kaleidoscope/configs

# Backup database
docker-compose exec -T postgres pg_dump -U postgres kaleidoscope > "$BACKUP_DIR/database.sql"

# Upload to S3
aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$TIMESTAMP/"

# Cleanup old backups (keep last 7 days)
find /opt/kaleidoscope/backups -type d -mtime +7 -exec rm -rf {} +

# Cleanup old S3 backups
aws s3 ls "s3://$S3_BUCKET/" | while read -r line; do
    createDate=$(echo "$line" | awk '{print $1" "$2}')
    createDate=$(date -d "$createDate" +%s)
    olderThan=$(date -d "7 days ago" +%s)
    if [ $createDate -lt $olderThan ]; then
        fileName=$(echo "$line" | awk '{print $4}')
        if [ "$fileName" != "" ]; then
            aws s3 rm "s3://$S3_BUCKET/$fileName"
        fi
    fi
done
EOF

chmod +x /opt/kaleidoscope/scripts/backup.sh

# Setup automatic SSL renewal
cat > /opt/kaleidoscope/scripts/renew-ssl.sh << 'EOF'
#!/bin/bash
set -e

certbot renew --quiet --pre-hook "systemctl stop nginx" --post-hook "systemctl start nginx"

# Update AWS ACM
CERT_PATH="/etc/letsencrypt/live/artificialthinker.com"
aws acm import-certificate \
    --certificate-arn arn:aws:acm:us-east-2:872515289083:certificate/your-cert-arn \
    --certificate fileb://$CERT_PATH/cert.pem \
    --private-key fileb://$CERT_PATH/privkey.pem \
    --certificate-chain fileb://$CERT_PATH/chain.pem

# Reload ELB listeners
aws elbv2 describe-load-balancers --query 'LoadBalancers[*].LoadBalancerArn' --output text | while read -r lb; do
    aws elbv2 describe-listeners --load-balancer-arn "$lb" --query 'Listeners[*].ListenerArn' --output text | while read -r listener; do
        aws elbv2 modify-listener --listener-arn "$listener" --certificates CertificateArn=arn:aws:acm:us-east-2:872515289083:certificate/your-cert-arn
    done
done
EOF

chmod +x /opt/kaleidoscope/scripts/renew-ssl.sh

# Setup cron jobs
(crontab -l 2>/dev/null || true; echo "0 2 * * * /opt/kaleidoscope/scripts/backup.sh") | crontab -
(crontab -l 2>/dev/null; echo "0 3 * * * /opt/kaleidoscope/scripts/renew-ssl.sh") | crontab -

# Start services
systemctl daemon-reload
systemctl enable kaleidoscope
systemctl start kaleidoscope

# Final health check
docker-compose ps
curl -s localhost:8000/health | jq .
echo "✅ Kaleidoscope AI System deployment complete"