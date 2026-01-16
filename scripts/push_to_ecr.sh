#!/bin/bash
# =============================================================================
# Docker Image Build and Push Script for AWS ECR
# =============================================================================
# This script builds Docker images and pushes them to AWS ECR repositories.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed and running
#   - ECR repositories created: sba-api, sba-streamlit
#
# Usage:
#   bash scripts/push_to_ecr.sh
# =============================================================================

set -e  # Exit on error

# Configuration
REGION="ap-south-2"
API_REPO="sba-api"
UI_REPO="sba-streamlit"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Docker Image Build and Push to ECR"
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

# ECR Registry URL
ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# =============================================================================
# Step 1: Login to ECR
# =============================================================================
echo -e "\n${YELLOW}[1/5] Logging in to ECR...${NC}"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
echo -e "${GREEN}✓ Logged in to ECR${NC}"

# =============================================================================
# Step 2: Build API Image
# =============================================================================
echo -e "\n${YELLOW}[2/5] Building API Docker image...${NC}"
docker build -f Dockerfile.api -t "$API_REPO:latest" .
echo -e "${GREEN}✓ Built API image${NC}"

# =============================================================================
# Step 3: Build UI Image
# =============================================================================
echo -e "\n${YELLOW}[3/5] Building UI Docker image...${NC}"
docker build -f Dockerfile.streamlit -t "$UI_REPO:latest" .
echo -e "${GREEN}✓ Built UI image${NC}"

# =============================================================================
# Step 4: Tag and Push API Image
# =============================================================================
echo -e "\n${YELLOW}[4/5] Tagging and pushing API image...${NC}"
docker tag "$API_REPO:latest" "$ECR_REGISTRY/$API_REPO:latest"
docker tag "$API_REPO:latest" "$ECR_REGISTRY/$API_REPO:$(git rev-parse --short HEAD 2>/dev/null || echo 'manual')"
docker push "$ECR_REGISTRY/$API_REPO:latest"
docker push "$ECR_REGISTRY/$API_REPO:$(git rev-parse --short HEAD 2>/dev/null || echo 'manual')"
echo -e "${GREEN}✓ Pushed API image${NC}"

# =============================================================================
# Step 5: Tag and Push UI Image
# =============================================================================
echo -e "\n${YELLOW}[5/5] Tagging and pushing UI image...${NC}"
docker tag "$UI_REPO:latest" "$ECR_REGISTRY/$UI_REPO:latest"
docker tag "$UI_REPO:latest" "$ECR_REGISTRY/$UI_REPO:$(git rev-parse --short HEAD 2>/dev/null || echo 'manual')"
docker push "$ECR_REGISTRY/$UI_REPO:latest"
docker push "$ECR_REGISTRY/$UI_REPO:$(git rev-parse --short HEAD 2>/dev/null || echo 'manual')"
echo -e "${GREEN}✓ Pushed UI image${NC}"

# =============================================================================
# Summary
# =============================================================================
echo -e "\n================================================================================"
echo -e "${GREEN}✓ Docker Images Pushed to ECR Successfully!${NC}"
echo "================================================================================"
echo ""
echo "Images pushed:"
echo "  API: $ECR_REGISTRY/$API_REPO:latest"
echo "  UI:  $ECR_REGISTRY/$UI_REPO:latest"
echo ""
echo "Next steps:"
echo "  1. Register ECS task definitions (if not done already)"
echo "  2. Create or update ECS services"
echo "  3. Deploy to ECS cluster"
echo ""
echo "================================================================================"
