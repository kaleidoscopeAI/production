provider "aws" {
  region = "us-east-1"
}

resource "aws_ecr_repository" "ai_repo" {
  name = "kaleidoscope-ai"
}

resource "aws_ecr_repository" "chatbot_repo" {
  name = "kaleidoscope-chatbot"
}

resource "aws_ecs_cluster" "kaleidoscope_cluster" {
  name = "KaleidoscopeCluster"
}

resource "aws_ecs_task_definition" "ai_task" {
  family                   = "ai_engine"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  network_mode             = "awsvpc"
  container_definitions = jsonencode([
    {
      name      = "ai_engine"
      image     = "${aws_ecr_repository.ai_repo.repository_url}:latest"
      cpu       = 512
      memory    = 1024
      essential = true
      portMappings = [{ containerPort = 5000, hostPort = 5000 }]
    }
  ])
}

resource "aws_ecs_service" "ai_service" {
  name            = "ai_service"
  cluster         = aws_ecs_cluster.kaleidoscope_cluster.id
  task_definition = aws_ecs_task_definition.ai_task.arn
  launch_type     = "FARGATE"
}

resource "aws_ecs_task_definition" "chatbot_task" {
  family                   = "chatbot_service"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  network_mode             = "awsvpc"
  container_definitions = jsonencode([
    {
      name      = "chatbot_service"
      image     = "${aws_ecr_repository.chatbot_repo.repository_url}:latest"
      cpu       = 512
      memory    = 1024
      essential = true
      portMappings = [{ containerPort = 8000, hostPort = 8000 }]
    }
  ])
}

resource "aws_ecs_service" "chatbot_service" {
  name            = "chatbot_service"
  cluster         = aws_ecs_cluster.kaleidoscope_cluster.id
  task_definition = aws_ecs_task_definition.chatbot_task.arn
  launch_type     = "FARGATE"
}

