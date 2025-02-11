provider "aws" {
  region = "us-east-2"
}

# OpenSearch Cluster
resource "aws_opensearch_domain" "monitoring" {
  domain_name    = "kaleidoscope-monitoring"
  engine_version = "OpenSearch_2.5"

  cluster_config {
    instance_type            = "r6g.large.search"
    instance_count          = 3
    zone_awareness_enabled  = true
    dedicated_master_enabled = true
    dedicated_master_count  = 3
    dedicated_master_type   = "r6g.large.search"
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 100
  }

  encrypt_at_rest {
    enabled = true
  }

  node_to_node_encryption {
    enabled = true
  }

  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }
}

# ML Pipeline Infrastructure
resource "aws_sagemaker_domain" "ml_pipeline" {
  domain_name = "kaleidoscope-ml"
  auth_mode   = "IAM"
  vpc_id      = aws_vpc.main.id
  subnet_ids  = [aws_subnet.private_1.id, aws_subnet.private_2.id]

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution_role.arn
  }
}

resource "aws_sagemaker_model" "kaleidoscope" {
  name               = "kaleidoscope-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image = "${aws_ecr_repository.ml_models.repository_url}:latest"
  }
}

# Distributed Tracing
resource "aws_xray_sampling_rule" "kaleidoscope" {
  rule_name      = "kaleidoscope-tracing"
  priority       = 1
  reservoir_size = 1
  fixed_rate     = 0.05
  host           = "*"
  http_method    = "*"
  service_name   = "*"
  service_type   = "*"
  url_path       = "*"
  version        = 1
}

# Security Enhancements
resource "aws_wafv2_web_acl" "main" {
  name        = "kaleidoscope-waf"
  description = "WAF for Kaleidoscope services"
  scope       = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "RateLimit"
    priority = 1

    override_action {
      none {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "RateLimitRule"
      sampled_requests_enabled  = true
    }
  }

  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "AWSManagedRulesCommonRuleSetMetric"
      sampled_requests_enabled  = true
    }
  }
}

# GuardDuty
resource "aws_guardduty_detector" "main" {
  enable = true

  datasources {
    s3_logs {
      enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
    malware_protection {
      scan_ec2_instance_with_findings {
        ebs_volumes {
          enable = true
        }
      }
    }
  }
}

# Enhanced Monitoring
resource "aws_prometheus_workspace" "monitoring" {
  alias = "kaleidoscope-prometheus"
}

resource "aws_prometheus_rule_group_namespace" "monitoring" {
  name         = "kaleidoscope-rules"
  workspace_id = aws_prometheus_workspace.monitoring.id

  data = <<EOF
groups:
  - name: kaleidoscope
    rules:
      - record: job:node_memory_utilization:avg
        expr: avg(node_memory_utilization) by (job)
      - alert: HighMemoryUsage
        expr: job:node_memory_utilization:avg > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High memory usage detected
EOF
}

# Chaos Engineering Infrastructure
resource "aws_sns_topic" "chaos_notifications" {
  name = "chaos-notifications"
}

resource "aws_iam_role" "chaos_execution" {
  name = "chaos-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Enhanced Networking
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.region}.s3"
}

resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.region}.dynamodb"
}

# Backup Infrastructure
resource "aws_backup_vault" "main" {
  name = "kaleidoscope-backup-vault"
  kms_key_arn = aws_kms_key.backup.arn
}

resource "aws_backup_plan" "main" {
  name = "kaleidoscope-backup-plan"

  rule {
    rule_name         = "daily_backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 5 ? * * *)"

    lifecycle {
      delete_after = 30
    }
  }
}

# Service Mesh
resource "aws_appmesh_mesh" "main" {
  name = "kaleidoscope-mesh"

  spec {
    egress_filter {
      type = "ALLOW_ALL"
    }
  }
}

resource "aws_appmesh_virtual_node" "services" {
  for_each = toset(["supernode", "kaleidoscope", "mirror", "chatbot"])

  name      = each.key
  mesh_name = aws_appmesh_mesh.main.id

  spec {
    listener {
      port_mapping {
        port     = 8080
        protocol = "http"
      }
    }

    service_discovery {
      aws_cloud_map {
        namespace_name = aws_service_discovery_private_dns_namespace.main.name
        service_name   = each.key
      }
    }
  }
}