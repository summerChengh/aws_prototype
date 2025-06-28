provider "aws" {
  region = var.aws_region
}

# S3 Buckets
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-data-lake-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-data-lake"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "data_lake_versioning" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake_encryption" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM Role for Glue
resource "aws_iam_role" "glue_role" {
  name = "${var.project_name}-glue-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_policy" "glue_s3_access" {
  name        = "${var.project_name}-glue-s3-access-${var.environment}"
  description = "Policy for Glue to access S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "glue_s3_access" {
  role       = aws_iam_role.glue_role.name
  policy_arn = aws_iam_policy.glue_s3_access.arn
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_policy" "lambda_s3_access" {
  name        = "${var.project_name}-lambda-s3-access-${var.environment}"
  description = "Policy for Lambda to access S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_s3_access" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_s3_access.arn
}

# Lambda Function for Data Fetching
resource "aws_lambda_function" "data_fetcher" {
  function_name = "${var.project_name}-data-fetcher-${var.environment}"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.handler"
  runtime       = "python3.9"
  timeout       = 300
  memory_size   = 256

  # This is a placeholder - in a real project, you would use a deployment package
  filename      = "${path.module}/lambda_placeholder.zip"
  
  environment {
    variables = {
      S3_BUCKET = aws_s3_bucket.data_lake.bucket
      ENVIRONMENT = var.environment
    }
  }

  tags = {
    Name        = "${var.project_name}-data-fetcher"
    Environment = var.environment
    Project     = var.project_name
  }
}

# EventBridge Rule for scheduled execution
resource "aws_cloudwatch_event_rule" "data_fetch_schedule" {
  name                = "${var.project_name}-data-fetch-schedule-${var.environment}"
  description         = "Schedule for fetching data from APIs"
  schedule_expression = "rate(6 hours)"
}

resource "aws_cloudwatch_event_target" "fetch_data_lambda" {
  rule      = aws_cloudwatch_event_rule.data_fetch_schedule.name
  target_id = "FetchDataLambda"
  arn       = aws_lambda_function.data_fetcher.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.data_fetcher.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.data_fetch_schedule.arn
}

# Glue Database
resource "aws_glue_catalog_database" "air_quality_db" {
  name = "${var.project_name}_db_${var.environment}"
}

# Glue Crawler for OpenAQ data
resource "aws_glue_crawler" "openaq_crawler" {
  name          = "${var.project_name}-openaq-crawler-${var.environment}"
  role          = aws_iam_role.glue_role.arn
  database_name = aws_glue_catalog_database.air_quality_db.name

  s3_target {
    path = "s3://${aws_s3_bucket.data_lake.bucket}/raw/openaq/"
  }

  schedule = "cron(0 */12 * * ? *)"
  
  tags = {
    Name        = "${var.project_name}-openaq-crawler"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Glue Crawler for NOAA data
resource "aws_glue_crawler" "noaa_crawler" {
  name          = "${var.project_name}-noaa-crawler-${var.environment}"
  role          = aws_iam_role.glue_role.arn
  database_name = aws_glue_catalog_database.air_quality_db.name

  s3_target {
    path = "s3://${aws_s3_bucket.data_lake.bucket}/raw/noaa/"
  }

  schedule = "cron(0 */12 * * ? *)"
  
  tags = {
    Name        = "${var.project_name}-noaa-crawler"
    Environment = var.environment
    Project     = var.project_name
  }
}

# API Gateway REST API
resource "aws_api_gateway_rest_api" "air_quality_api" {
  name        = "${var.project_name}-api-${var.environment}"
  description = "Air Quality Prediction API"
  
  tags = {
    Name        = "${var.project_name}-api"
    Environment = var.environment
    Project     = var.project_name
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name = "/aws/apigateway/${aws_api_gateway_rest_api.air_quality_api.name}"
  retention_in_days = 30
  
  tags = {
    Name        = "${var.project_name}-api-logs"
    Environment = var.environment
    Project     = var.project_name
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "api_distribution" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "${var.project_name} API Distribution"
  default_root_object = "index.html"
  price_class         = "PriceClass_100"

  origin {
    domain_name = "${aws_api_gateway_rest_api.air_quality_api.id}.execute-api.${var.aws_region}.amazonaws.com"
    origin_id   = "APIGateway"
    origin_path = "/${var.environment}"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "APIGateway"

    forwarded_values {
      query_string = true
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
  
  tags = {
    Name        = "${var.project_name}-cloudfront"
    Environment = var.environment
    Project     = var.project_name
  }
}

# SageMaker Notebook Instance
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.lambda_s3_access.arn  # Reusing the same policy
}

resource "aws_sagemaker_notebook_instance" "air_quality_notebook" {
  name                    = "${var.project_name}-notebook-${var.environment}"
  role_arn                = aws_iam_role.sagemaker_role.arn
  instance_type           = "ml.t3.medium"
  volume_size             = 50
  
  tags = {
    Name        = "${var.project_name}-notebook"
    Environment = var.environment
    Project     = var.project_name
  }
} 