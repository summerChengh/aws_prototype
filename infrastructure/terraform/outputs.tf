output "data_lake_bucket" {
  description = "Name of the S3 bucket used as data lake"
  value       = aws_s3_bucket.data_lake.bucket
}

output "data_lake_arn" {
  description = "ARN of the S3 bucket used as data lake"
  value       = aws_s3_bucket.data_lake.arn
}

output "glue_database_name" {
  description = "Name of the Glue catalog database"
  value       = aws_glue_catalog_database.air_quality_db.name
}

output "lambda_function_name" {
  description = "Name of the Lambda function for data fetching"
  value       = aws_lambda_function.data_fetcher.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function for data fetching"
  value       = aws_lambda_function.data_fetcher.arn
}

output "api_gateway_id" {
  description = "ID of the API Gateway REST API"
  value       = aws_api_gateway_rest_api.air_quality_api.id
}

output "api_gateway_url" {
  description = "URL of the API Gateway REST API"
  value       = "${aws_api_gateway_rest_api.air_quality_api.id}.execute-api.${var.aws_region}.amazonaws.com/${var.environment}"
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.api_distribution.domain_name
}

output "sagemaker_notebook_url" {
  description = "URL of the SageMaker notebook instance"
  value       = "https://${var.aws_region}.console.aws.amazon.com/sagemaker/home?region=${var.aws_region}#/notebook-instances/openNotebook/${aws_sagemaker_notebook_instance.air_quality_notebook.name}?view=classic"
}

output "glue_role_arn" {
  description = "ARN of the IAM role for Glue"
  value       = aws_iam_role.glue_role.arn
}

output "lambda_role_arn" {
  description = "ARN of the IAM role for Lambda"
  value       = aws_iam_role.lambda_role.arn
}

output "sagemaker_role_arn" {
  description = "ARN of the IAM role for SageMaker"
  value       = aws_iam_role.sagemaker_role.arn
} 