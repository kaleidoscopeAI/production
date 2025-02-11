terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# VPC and Networking
resource "aws_vpc" "kaleidoscope_vpc" {
  cidr_block = "10.0.0.0/16"
  
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "kaleidoscope-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.kaleidoscope_vpc.id
  
  tags = {
    Name = "kaleidoscope-igw"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "kaleidoscope" {
  name = "kaleidoscope-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "kaleidoscope" {
  cluster_name = aws_ecs_cluster.kaleidoscope.name
  
  capacity_providers = ["FARGATE"]
  
  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = "FARGATE"
  }
}

# Task Definitions
resource "aws_ecs_task_definition" "membrane" {
  family                   = "kaleidoscope-membrane"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = 1024
  memory                  = 2048
  execution_role_arn      = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn
  
  container_definitions = jsonencode([
    {
      name      = "membrane"
      image     = "${aws_ecr_repository.kaleidoscope.repository_url}:latest"
      essential = true
      
      environment = [
        {
          name  = "AWS_DEFAULT_REGION"
          value = "us-east-1"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/kaleidoscope"
          awslogs-region        = "us-east-1"
          awslogs-stream-prefix = "membrane"
        }
      }
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
    }
  ])
}

# IAM Roles
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "kaleidoscope-task-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# DynamoDB Tables
resource "aws_dynamodb_table" "insights" {
  name           = "kaleidoscope-insights"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "insight_id"
  range_key      = "timestamp"
  
  attribute {
    name = "insight_id"
    type = "S"
  }
  
  attribute {
    name = "timestamp"
    type = "N"
  }
  
  ttl {
    attribute_name = "expiry"
  }
  
  global_secondary_index {
    name               = "timestamp-index"
    hash_key           = "timestamp"
    projection_type    = "ALL"
    write_capacity     = 10
    read_capacity      = 10
  }
}

# S3 Bucket for Insights
resource "aws_s3_bucket" "insights" {
  bucket = "kaleidoscope-insights"
  
  versioning {
    enabled = true
  }
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# SQS Queues
resource "aws_sqs_queue" "insights_queue" {
  name                      = "kaleidoscope-insights-queue"
  delay_seconds             = 0
  max_message_size         = 262144
  message_retention_seconds = 86400
  receive_wait_time_seconds = 10
  visibility_timeout_seconds = 30
  
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.insights_dlq.arn
    maxReceiveCount     = 3
  })
}

resource "aws_sqs_queue" "insights_dlq" {
  name = "kaleidoscope-insights-dlq"
}

# CloudWatch Monitoring
resource "aws_cloudwatch_dashboard" "kaleidoscope" {
  dashboard_name = "kaleidoscope-monitoring"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", aws_ecs_cluster.kaleidoscope.name],
            [".", "MemoryUtilization", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = "us-east-1"
          title  = "ECS Cluster Performance"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/DynamoDB", "ConsumedReadCapacityUnits", "TableName", aws_dynamodb_table.insights.name],
            [".", "ConsumedWriteCapacityUnits", ".", "."]
          ]
          period = 300
          stat   = "Sum"
          region = "us-east-1"
          title  = "DynamoDB Consumption"
        }
      }
    ]
  })
}

# Lambda Functions
resource "aws_lambda_function" "insight_processor" {
  filename         = "insight_processor.zip"
  function_name    = "kaleidoscope-insight-processor"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 30
  memory_size     = 1024
  
  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.insights.name
      SQS_QUEUE_URL = aws_sqs_queue.insights_queue.url
    }
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "kaleidoscope" {
  name = "kaleidoscope-api"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_resource" "insights" {
  rest_api_id = aws_api_gateway_rest_api.kaleidoscope.id
  parent_id   = aws_api_gateway_rest_api.kaleidoscope.root_resource_id
  path_part   = "insights"
}

resource "aws_api_gateway_method" "post_insight" {
  rest_api_id   = aws_api_gateway_rest_api.kaleidoscope.id
  resource_id   = aws_api_gateway_resource.insights.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.kaleidoscope.id
  resource_id = aws_api_gateway_resource.insights.id
  http_method = aws_api_gateway_method.post_insight.http_method
  
  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.insight_processor.invoke_arn
}

# Outputs
output "api_gateway_url" {
  value = "${aws_api_gateway_rest_api.kaleidoscope.execution_arn}/prod/insights"
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.kaleidoscope.name
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.insights.name
}

output "sqs_queue_url" {
  value = aws_sqs_queue.insights_queue.url
}