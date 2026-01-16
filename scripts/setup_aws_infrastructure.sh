#!/bin/bash
# =============================================================================
# AWS Infrastructure Setup Script for SBA Loan ML System
# =============================================================================
# This script sets up the AWS infrastructure needed for ECS deployment.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Permissions to create IAM roles, CloudWatch log groups
#   - ECR repositories already created: sba-api, sba-streamlit
#   - ECS cluster already created: sba-cluster
#
# Usage:
#   bash scripts/setup_aws_infrastructure.sh
# =============================================================================

set -e  # Exit on error

# Configuration
REGION="ap-south-2"
CLUSTER_NAME="sba-cluster"
API_LOG_GROUP="/ecs/sba-loan-api"
UI_LOG_GROUP="/ecs/sba-loan-ui"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "AWS Infrastructure Setup for SBA Loan ML System"
echo "Region: $REGION"
echo "================================================================================"

# Get AWS Account ID
echo -e "\n${YELLOW}Getting AWS Account ID...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}❌ Failed to get AWS Account ID. Check your AWS credentials.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ AWS Account ID: $AWS_ACCOUNT_ID${NC}"

# =============================================================================
# Step 1: Verify ECR repositories exist
# =============================================================================
echo -e "\n${YELLOW}[1/5] Verifying ECR repositories...${NC}"

for repo in "sba-api" "sba-streamlit"; do
    if aws ecr describe-repositories --repository-names "$repo" --region "$REGION" &>/dev/null; then
        echo -e "${GREEN}✓ ECR repository exists: $repo${NC}"
    else
        echo -e "${RED}❌ ECR repository not found: $repo${NC}"
        echo "Please create it manually or with:"
        echo "  aws ecr create-repository --repository-name $repo --region $REGION"
        exit 1
    fi
done

# =============================================================================
# Step 2: Verify ECS cluster exists
# =============================================================================
echo -e "\n${YELLOW}[2/5] Verifying ECS cluster...${NC}"

if aws ecs describe-clusters --clusters "$CLUSTER_NAME" --region "$REGION" | grep -q "ACTIVE"; then
    echo -e "${GREEN}✓ ECS cluster exists: $CLUSTER_NAME${NC}"
else
    echo -e "${RED}❌ ECS cluster not found: $CLUSTER_NAME${NC}"
    echo "Please create it manually or with:"
    echo "  aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION"
    exit 1
fi

# =============================================================================
# Step 3: Create CloudWatch Log Groups
# =============================================================================
echo -e "\n${YELLOW}[3/5] Creating CloudWatch log groups...${NC}"

for log_group in "$API_LOG_GROUP" "$UI_LOG_GROUP"; do
    if aws logs describe-log-groups --log-group-name-prefix "$log_group" --region "$REGION" | grep -q "$log_group"; then
        echo -e "${YELLOW}⏭  Log group already exists: $log_group${NC}"
    else
        aws logs create-log-group --log-group-name "$log_group" --region "$REGION"
        echo -e "${GREEN}✓ Created log group: $log_group${NC}"
    fi
done

# =============================================================================
# Step 4: Create IAM Task Execution Role (if not exists)
# =============================================================================
echo -e "\n${YELLOW}[4/5] Creating IAM task execution role...${NC}"

EXECUTION_ROLE_NAME="ecsTaskExecutionRole"

if aws iam get-role --role-name "$EXECUTION_ROLE_NAME" &>/dev/null; then
    echo -e "${YELLOW}⏭  IAM role already exists: $EXECUTION_ROLE_NAME${NC}"
else
    # Create the role
    aws iam create-role \
        --role-name "$EXECUTION_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' &>/dev/null

    # Attach the AWS managed policy
    aws iam attach-role-policy \
        --role-name "$EXECUTION_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"

    echo -e "${GREEN}✓ Created IAM role: $EXECUTION_ROLE_NAME${NC}"
fi

# =============================================================================
# Step 5: Create IAM Task Role for S3 Access
# =============================================================================
echo -e "\n${YELLOW}[5/5] Creating IAM task role for S3 access...${NC}"

TASK_ROLE_NAME="sba-loan-api-task-role"

if aws iam get-role --role-name "$TASK_ROLE_NAME" &>/dev/null; then
    echo -e "${YELLOW}⏭  IAM role already exists: $TASK_ROLE_NAME${NC}"
else
    # Create the role with trust policy
    aws iam create-role \
        --role-name "$TASK_ROLE_NAME" \
        --assume-role-policy-document file://infrastructure/iam-trust-policy.json \
        &>/dev/null

    echo -e "${GREEN}✓ Created IAM role: $TASK_ROLE_NAME${NC}"
fi

# Create and attach S3 access policy
POLICY_NAME="sba-loan-s3-access-policy"
POLICY_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:policy/$POLICY_NAME"

if aws iam get-policy --policy-arn "$POLICY_ARN" &>/dev/null; then
    echo -e "${YELLOW}⏭  IAM policy already exists: $POLICY_NAME${NC}"
else
    aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document file://infrastructure/iam-task-role-policy.json \
        &>/dev/null

    echo -e "${GREEN}✓ Created IAM policy: $POLICY_NAME${NC}"
fi

# Attach policy to role
if aws iam list-attached-role-policies --role-name "$TASK_ROLE_NAME" | grep -q "$POLICY_NAME"; then
    echo -e "${YELLOW}⏭  Policy already attached to role${NC}"
else
    aws iam attach-role-policy \
        --role-name "$TASK_ROLE_NAME" \
        --policy-arn "$POLICY_ARN"

    echo -e "${GREEN}✓ Attached policy to role${NC}"
fi

# =============================================================================
# Summary
# =============================================================================
echo -e "\n================================================================================"
echo -e "${GREEN}✓ Infrastructure Setup Complete!${NC}"
echo "================================================================================"
echo ""
echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "Region: $REGION"
echo ""
echo "Resources created/verified:"
echo "  ✓ ECR repositories: sba-api, sba-streamlit"
echo "  ✓ ECS cluster: $CLUSTER_NAME"
echo "  ✓ CloudWatch log groups: $API_LOG_GROUP, $UI_LOG_GROUP"
echo "  ✓ IAM execution role: $EXECUTION_ROLE_NAME"
echo "  ✓ IAM task role: $TASK_ROLE_NAME (with S3 access)"
echo ""
echo "Next steps:"
echo "  1. Update task definitions with your account ID:"
echo "     sed -i '' 's/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g' infrastructure/ecs-task-*.json"
echo ""
echo "  2. Build and push Docker images:"
echo "     bash scripts/push_to_ecr.sh"
echo ""
echo "  3. Register task definitions:"
echo "     aws ecs register-task-definition --cli-input-json file://infrastructure/ecs-task-api.json --region $REGION"
echo "     aws ecs register-task-definition --cli-input-json file://infrastructure/ecs-task-ui.json --region $REGION"
echo ""
echo "  4. Create ECS services (via AWS Console or CLI)"
echo ""
echo "  5. Set up Application Load Balancer and target groups"
echo ""
echo "  6. Configure GitHub Actions secrets:"
echo "     - AWS_ACCESS_KEY_ID"
echo "     - AWS_SECRET_ACCESS_KEY"
echo "     - AWS_ACCOUNT_ID"
echo ""
echo "================================================================================"
