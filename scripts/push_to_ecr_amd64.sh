#!/bin/bash
# Build for AMD64 (x86_64) architecture for AWS Fargate

set -e

REGION="ap-south-2"
API_REPO="sba-api"
UI_REPO="sba-streamlit"

echo "Building Docker images for AMD64 architecture..."

# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Build API image for AMD64
echo "Building API image for linux/amd64..."
docker buildx build --platform linux/amd64 -f Dockerfile.api -t "$API_REPO:latest" --load .

# Build UI image for AMD64
echo "Building UI image for linux/amd64..."
docker buildx build --platform linux/amd64 -f Dockerfile.streamlit -t "$UI_REPO:latest" --load .

# Tag and push API
echo "Pushing API image..."
docker tag "$API_REPO:latest" "$ECR_REGISTRY/$API_REPO:latest"
docker push "$ECR_REGISTRY/$API_REPO:latest"

# Tag and push UI
echo "Pushing UI image..."
docker tag "$UI_REPO:latest" "$ECR_REGISTRY/$UI_REPO:latest"
docker push "$ECR_REGISTRY/$UI_REPO:latest"

echo "✓ Images pushed successfully for AMD64 architecture!"
