# deploy_all.sh
#!/bin/bash
set -e

export AWS_ACCESS_KEY_ID="AKIA4WJPWX757RLAGXU7"
export AWS_SECRET_ACCESS_KEY="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"
export AWS_DEFAULT_REGION="us-east-2"
export DOMAIN="artificialthinker.com"
export ECR_REPO="872515289083.dkr.ecr.us-east-2.amazonaws.com"

# Install required tools
apt-get update && apt-get install -y \
    docker.io \
    docker-compose \
    nginx \
    certbot \
    python3-certbot-nginx \
    aws-cli \
    jq \
    terraform

# Generate SSL certificates
certbot certonly --standalone -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email jmgraham1000@gmail.com

# Configure AWS
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

# Create ECR repositories
for repo in supernode kaleidoscope mirror chatbot frontend nginx; do
    aws ecr create-repository --repository-name kaleidoscope-$repo --image-scanning-configuration scanOnPush=true
done

# Build and push Docker images
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO

for service in supernode kaleidoscope mirror chatbot frontend; do
    docker build -t $ECR_REPO/kaleidoscope-$service:latest -f docker/$service/Dockerfile .
    docker push $ECR_REPO/kaleidoscope-$service:latest
done

# Deploy infrastructure
cd terraform
terraform init
terraform apply -auto-approve

# Get load balancer DNS
ALB_DNS=$(terraform output -raw alb_dns_name)

# Update DNS records
ZONE_ID=$(aws route53 list-hosted-zones --query 'HostedZones[0].Id' --output text)

aws route53 change-resource-record-sets \
    --hosted-zone-id $ZONE_ID \
    --change-batch '{
        "Changes": [{
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "'$DOMAIN'",
                "Type": "A",
                "AliasTarget": {
                    "HostedZoneId": "Z35SXDOTRQ7X7K",
                    "DNSName": "'$ALB_DNS'",
                    "EvaluateTargetHealth": true
                }
            }
        }]
    }'

# Deploy ECS services
for service in supernode kaleidoscope mirror chatbot frontend; do
    aws ecs update-service \
        --cluster kaleidoscope \
        --service $service \
        --force-new-deployment
done

# Setup reverse proxy
cat > /etc/nginx/sites-available/artificialthinker.com << EOF
server {
    listen 80;
    server_name artificialthinker.com www.artificialthinker.com;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name artificialthinker.com www.artificialthinker.com;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;

    location / {
        proxy_pass http://$ALB_DNS;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

ln -sf /etc/nginx/sites-available/artificialthinker.com /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx

# Setup monitoring
cat > /etc/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kaleidoscope'
    ec2_sd_configs:
      - region: us-east-2
        port: 9100
    relabel_configs:
      - source_labels: [__meta_ec2_tag_Environment]
        regex: production
        action: keep
EOF

# Setup continuous deployment
cat > .github/workflows/deploy.yml << EOF
name: Deploy Kaleidoscope

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: \${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-2
      
      - name: Build and push
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
          for service in supernode kaleidoscope mirror chatbot frontend; do
            docker build -t $ECR_REPO/kaleidoscope-\$service:latest -f docker/\$service/Dockerfile .
            docker push $ECR_REPO/kaleidoscope-\$service:latest
          done
      
      - name: Deploy
        run: |
          for service in supernode kaleidoscope mirror chatbot frontend; do
            aws ecs update-service --cluster kaleidoscope --service \$service --force-new-deployment
          done
EOF

# Setup automatic backups
cat > /etc/cron.daily/backup-kaleidoscope << EOF
#!/bin/bash

TIMESTAMP=\$(date +%Y%m%d-%H%M%S)
BACKUP_BUCKET="kaleidoscope-backups"

# Backup ECS task definitions
aws ecs list-task-definitions | \
    jq -r '.taskDefinitionArns[]' | \
    xargs -I {} aws ecs describe-task-definition --task-definition {} | \
    aws s3 cp - s3://\$BACKUP_BUCKET/ecs/\$TIMESTAMP/

# Backup EFS data
aws efs describe-file-systems | \
    jq -r '.FileSystems[].FileSystemId' | \
    xargs -I {} aws backup start-backup-job \
        --backup-vault-name kaleidoscope \
        --resource-arn arn:aws:elasticfilesystem:us-east-2:872515289083:file-system/{} \
        --iam-role-arn arn:aws:iam::872515289083:role/backup-role

# Backup RDS snapshots
aws rds describe-db-instances | \
    jq -r '.DBInstances[].DBInstanceIdentifier' | \
    xargs -I {} aws rds create-db-snapshot \
        --db-instance-identifier {} \
        --db-snapshot-identifier {}-\$TIMESTAMP
EOF

chmod +x /etc/cron.daily/backup-kaleidoscope

# Setup monitoring dashboard
aws cloudwatch put-dashboard --dashboard-name KaleidoscopeMetrics --dashboard-body '{
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/ECS", "CPUUtilization", "ClusterName", "kaleidoscope"],
                    [".", "MemoryUtilization", ".", "."]
                ],
                "period": 300,
                "stat": "Average",
                "region": "us-east-2",
                "title": "ECS Metrics"
            }
        }
    ]
}'

echo "✅ Deployment complete! System is now live at https://$DOMAIN"