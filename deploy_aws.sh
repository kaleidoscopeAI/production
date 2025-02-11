#!/bin/bash

echo "🚀 Deploying Kaleidoscope AI System to AWS..."

# Authenticate Docker with AWS
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build & Push AI Docker Image
docker build -t kaleidoscope-ai -f docker/Dockerfile.ai .
docker tag kaleidoscope-ai:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/kaleidoscope-ai:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/kaleidoscope-ai:latest

# Build & Push Chatbot Docker Image
docker build -t kaleidoscope-chatbot -f docker/Dockerfile.chatbot .
docker tag kaleidoscope-chatbot:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/kaleidoscope-chatbot:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/kaleidoscope-chatbot:latest

# Deploy with Terraform
cd infrastructure/terraform/
terraform init
terraform apply -auto-approve

echo "✅ Deployment Complete! AI System is now live on AWS Fargate."

