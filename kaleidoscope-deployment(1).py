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
    attribute_name = "exp