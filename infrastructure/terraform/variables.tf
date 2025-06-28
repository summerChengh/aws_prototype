variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "air-quality-predictor"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "openaq_api_url" {
  description = "Base URL for OpenAQ API"
  type        = string
  default     = "https://api.openaq.org/v2"
}

variable "noaa_api_url" {
  description = "Base URL for NOAA API"
  type        = string
  default     = "https://www.ncei.noaa.gov/access/services/data/v1"
}

variable "data_fetch_interval" {
  description = "Interval for data fetching in hours"
  type        = number
  default     = 6
}

variable "sagemaker_instance_type" {
  description = "Instance type for SageMaker notebook"
  type        = string
  default     = "ml.t3.medium"
}

variable "glue_job_timeout" {
  description = "Timeout for Glue jobs in minutes"
  type        = number
  default     = 60
}

variable "api_throttling_rate_limit" {
  description = "API Gateway throttling rate limit"
  type        = number
  default     = 100
}

variable "api_throttling_burst_limit" {
  description = "API Gateway throttling burst limit"
  type        = number
  default     = 50
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default     = {
    ManagedBy = "Terraform"
    Project   = "AirQualityPredictor"
  }
} 