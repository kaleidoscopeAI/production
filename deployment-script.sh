#!/bin/bash
set -e

export AWS_ACCESS_KEY_ID="AKIA4WJPWX757RLAGXU7"
export AWS_SECRET_ACCESS_KEY="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"
export AWS_DEFAULT_REGION="us-east-2"
export GODADDY_OTE_KEY="3mM44Ywf1677ab_Hx1wV7Hy6JjiXcPeGiRQzL"
export GODADDY_OTE_SECRET="DrHG2dfy7N1Qmeo1z23cbE"
export GODADDY_PROD_KEY="h1ULrdTohqt8_EKjfAyyafQFKFVbZ5CdKzb"
export GODADDY_PROD_SECRET="DPke8SKwuYk8EUsv6JAxkP"
export DOMAIN="artificialthinker.com"

# Install required tools
apt-get update && apt-get install -y \
    docker.io \
    docker-compose \
    certbot \
    python3-certbot-nginx \
    nginx \
    jq \
    curl

# Setup SSL certificates
certbot certonly --standalone \
    -d $DOMAIN \
    -d www.$DOMAIN \
    --non-interactive \
    --agree-tos \
    --email jmgraham1000@gmail.com \
    --http-01-port=80

# Create necessary directories
mkdir -p ssl data/prometheus data/grafana data/loki

# Copy SSL certificates
cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem ssl/
cp /etc/letsencrypt/live/$DOMAIN/privkey.pem ssl/
cp /etc/letsencrypt/live/$DOMAIN/chain.pem ssl/

# Setup Docker Swarm
docker swarm init

# Create Docker secrets
echo "$AWS_ACCESS_KEY_ID" | docker secret create aws_access_key_id -
echo "$AWS_SECRET_ACCESS_KEY" | docker secret create aws_secret_access_key -
echo "$GODADDY_PROD_KEY" | docker secret create godaddy_key -
echo "$GODADDY_PROD_SECRET" | docker secret create godaddy_secret -

# Create required networks
docker network create --driver overlay proxy
docker network create --driver overlay monitoring

# Pull required images
docker-compose pull

# Deploy stack
docker stack deploy -c docker-compose.yml -c docker-compose.override.yml kaleidoscope

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Configure Grafana
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="admin"

# Create Prometheus data source
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"name":"Prometheus","type":"prometheus","url":"http://prometheus:9090","access":"proxy"}' \
    $GRAFANA_URL/api/datasources

# Create Loki data source
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"name":"Loki","type":"loki","url":"http://loki:3100","access":"proxy"}' \
    $GRAFANA_URL/api/datasources

# Import dashboards
for dashboard in grafana/dashboards/*.json; do
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @$dashboard \
        $GRAFANA_URL/api/dashboards/db
done

# Setup cron for certificate renewal
echo "0 0 1 * * certbot renew --quiet" | crontab -

# Setup monitoring for certificate expiry
cat > /etc/prometheus/rules/ssl.yml << EOF
groups:
- name: ssl
  rules:
  - alert: SSLCertificateExpiringSoon
    expr: probe_ssl_earliest_cert_expiry - time() < 86400 * 30
    for: 1h
    labels:
      severity: warning
    annotations:
      description: SSL certificate for {{ \$labels.instance }} expires in less than 30 days
EOF

# Configure log rotation
cat > /etc/logrotate.d/kaleidoscope << EOF
/var/log/kaleidoscope/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 640 root root
    postrotate
        docker kill -s SIGUSR1 \$(docker ps -q --filter name=