#!/bin/bash
set -e

export AWS_ACCESS_KEY_ID="AKIA4WJPWX757RLAGXU7"
export AWS_SECRET_ACCESS_KEY="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"
export AWS_DEFAULT_REGION="us-east-2"
export AWS_ACCOUNT_ID="872515289083"
export GODADDY_OTE_KEY="3mM44Ywf1677ab_Hx1wV7Hy6JjiXcPeGiRQzL"
export GODADDY_OTE_SECRET="DrHG2dfy7N1Qmeo1z23cbE"
export GODADDY_PROD_KEY="h1ULrdTohqt8_EKjfAyyafQFKFVbZ5CdKzb"
export GODADDY_PROD_SECRET="DPke8SKwuYk8EUsv6JAxkP"

# Secure credentials in AWS Secrets Manager
aws secretsmanager create-secret \
    --name kaleidoscope/credentials \
    --secret-string "{\"godaddy\": {\"ote\": {\"key\": \"$GODADDY_OTE_KEY\", \"secret\": \"$GODADDY_OTE_SECRET\"}, \"production\": {\"key\": \"$GODADDY_PROD_KEY\", \"secret\": \"$GODADDY_PROD_SECRET\"}}}"

# Deploy infrastructure and configure domains
python3 - << EOF
import asyncio
from domain_manager import DomainManager, Environment

async def deploy():
    manager = DomainManager()
    await manager.initialize()
    
    try:
        print("Deploying OTE environment...")
        await manager.deploy_environment(Environment.OTE, "kaleidoscope-ote-alb.us-east-2.elb.amazonaws.com")
        
        print("Deploying Production environment...")
        await manager.deploy_environment(Environment.PRODUCTION, "kaleidoscope-prod-alb.us-east-2.elb.amazonaws.com")
    finally:
        await manager.close()

asyncio.run(deploy())
EOF

# Deploy monitoring
cat > /etc/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kaleidoscope-ote'
    static_configs:
      - targets: ['ote.artificialthinker.com:9090']
    
  - job_name: 'kaleidoscope-production'
    static_configs:
      - targets: ['artificialthinker.com:9090']
EOF

# Create CloudWatch dashboards
aws cloudwatch put-dashboard \
    --dashboard-name KaleidoscopeMetrics \
    --dashboard-body file://monitoring/dashboard.json

# Setup automated failover
aws route53 create-health-check \
    --caller-reference $(date +%s) \
    --health-check-config "{\"Type\":\"HTTPS\",\"FullyQualifiedDomainName\":\"artificialthinker.com\",\"RequestInterval\":30,\"FailureThreshold\":3,\"MeasureLatency\":true,\"Enabled\":true}"

echo "✅ Environment setup complete"