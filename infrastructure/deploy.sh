#!/bin/bash

# AWS Air Quality Predictor - Infrastructure Deployment Script
# This script deploys the AWS infrastructure using Terraform

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
ENV="dev"
REGION="us-west-2"
ACTION="plan"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -e|--environment)
      ENV="$2"
      shift
      shift
      ;;
    -r|--region)
      REGION="$2"
      shift
      shift
      ;;
    -a|--action)
      ACTION="$2"
      shift
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [-e|--environment <dev|staging|prod>] [-r|--region <aws-region>] [-a|--action <plan|apply|destroy>]"
      exit 1
      ;;
  esac
done

echo -e "${GREEN}=== AWS Air Quality Predictor - Infrastructure Deployment ===${NC}"
echo -e "${YELLOW}Environment: ${ENV}${NC}"
echo -e "${YELLOW}AWS Region: ${REGION}${NC}"
echo -e "${YELLOW}Action: ${ACTION}${NC}"

# Create a placeholder Lambda zip file if it doesn't exist
if [ ! -f "terraform/lambda_placeholder.zip" ]; then
  echo -e "${YELLOW}Creating placeholder Lambda zip file...${NC}"
  mkdir -p tmp
  echo "# Placeholder Lambda function" > tmp/index.py
  echo "def handler(event, context):" >> tmp/index.py
  echo "    return {'statusCode': 200, 'body': 'Placeholder function'}" >> tmp/index.py
  cd tmp
  zip -q ../terraform/lambda_placeholder.zip index.py
  cd ..
  rm -rf tmp
fi

# Navigate to Terraform directory
cd terraform

# Initialize Terraform
echo -e "${GREEN}Initializing Terraform...${NC}"
terraform init

# Create terraform.tfvars file
cat > terraform.tfvars << EOF
aws_region = "${REGION}"
environment = "${ENV}"
project_name = "air-quality-predictor"
EOF

# Run Terraform
case $ACTION in
  plan)
    echo -e "${GREEN}Creating Terraform plan...${NC}"
    terraform plan -var-file=terraform.tfvars -out=tfplan
    ;;
  apply)
    echo -e "${GREEN}Applying Terraform configuration...${NC}"
    terraform apply -var-file=terraform.tfvars -auto-approve
    ;;
  destroy)
    echo -e "${RED}WARNING: This will destroy all resources. Are you sure? (y/N)${NC}"
    read -r confirm
    if [[ $confirm == "y" || $confirm == "Y" ]]; then
      echo -e "${GREEN}Destroying infrastructure...${NC}"
      terraform destroy -var-file=terraform.tfvars -auto-approve
    else
      echo -e "${YELLOW}Destroy cancelled.${NC}"
      exit 0
    fi
    ;;
  *)
    echo -e "${RED}Invalid action: ${ACTION}${NC}"
    echo "Valid actions: plan, apply, destroy"
    exit 1
    ;;
esac

# Output results
if [ "$ACTION" == "apply" ]; then
  echo -e "${GREEN}=== Deployment Complete ===${NC}"
  echo -e "${YELLOW}Infrastructure outputs:${NC}"
  terraform output
  
  # Save outputs to a file
  echo -e "${GREEN}Saving outputs to ../outputs-${ENV}.json${NC}"
  terraform output -json > "../outputs-${ENV}.json"
fi

echo -e "${GREEN}=== Done ===${NC}" 