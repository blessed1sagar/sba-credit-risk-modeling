# AWS ECS Deployment Guide

Complete guide for deploying the SBA Loan ML System to AWS ECS Fargate.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [AWS Infrastructure Setup](#aws-infrastructure-setup)
4. [Docker Image Build & Push](#docker-image-build--push)
5. [ECS Service Configuration](#ecs-service-configuration)
6. [GitHub Actions CI/CD](#github-actions-cicd)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Prerequisites

### Required Tools
- AWS CLI configured with credentials
- Docker installed and running
- Git for version control
- Python 3.10+

### AWS Resources Already Created
✅ ECR repositories: `sba-api`, `sba-streamlit`
✅ ECS cluster: `sba-cluster`
✅ S3 bucket: `sba-credit-risk-artifacts-sagar` (ap-south-2)

### AWS Permissions Required
- ECR: Push/pull images
- ECS: Create/update services and task definitions
- IAM: Create roles and policies
- CloudWatch: Create log groups
- S3: Read access to artifacts bucket

---

## Local Setup

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd ml-eng-lr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your AWS credentials
# Required variables:
#   - AWS_REGION=ap-south-2
#   - AWS_ACCESS_KEY_ID=<your-key>
#   - AWS_SECRET_ACCESS_KEY=<your-secret>
#   - S3_BUCKET_NAME=sba-credit-risk-artifacts-sagar
#   - SYNC_FROM_S3=false  (for local dev)
```

### 3. Run Pipeline Locally (Optional)

```bash
# Run full pipeline (preprocessing + training)
python run_pipeline.py

# Upload artifacts to S3
python scripts/push_artifacts.py
```

### 4. Test Locally with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access services:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Streamlit UI: http://localhost:8501

# Stop services
docker-compose down
```

---

## AWS Infrastructure Setup

### Step 1: Run Infrastructure Setup Script

This script creates IAM roles, CloudWatch log groups, and verifies existing resources.

```bash
# Make script executable (if not already)
chmod +x scripts/setup_aws_infrastructure.sh

# Run setup
bash scripts/setup_aws_infrastructure.sh
```

**What it creates:**
- ✅ CloudWatch log groups: `/ecs/sba-loan-api`, `/ecs/sba-loan-ui`
- ✅ IAM execution role: `ecsTaskExecutionRole`
- ✅ IAM task role: `sba-loan-api-task-role` (with S3 access)
- ✅ IAM policy: `sba-loan-s3-access-policy`

### Step 2: Configure Task Definitions

**Important Note on API_URL Configuration:**

The UI task definition uses a placeholder `${ALB_DNS}/api` for the API_URL environment variable. This placeholder is automatically replaced during deployment:

- **GitHub Actions Deployment**: The workflow automatically fetches the ALB DNS and replaces the placeholder
- **Local/Manual Deployment**: Use the helper script: `bash scripts/update_ui_task_def.sh`

**DO NOT manually hardcode the ALB DNS** in the task definition file. The placeholder ensures the configuration stays dynamic and works across different environments.

---

## Docker Image Build & Push

### Option 1: Using the Helper Script (Recommended)

```bash
# Make script executable (if not already)
chmod +x scripts/push_to_ecr.sh

# Build and push both images
bash scripts/push_to_ecr.sh
```

### Option 2: Manual Build and Push

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.ap-south-2.amazonaws.com"

# Login to ECR
aws ecr get-login-password --region ap-south-2 | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

# Build and push API image
docker build -f Dockerfile.api -t sba-api:latest .
docker tag sba-api:latest $ECR_REGISTRY/sba-api:latest
docker push $ECR_REGISTRY/sba-api:latest

# Build and push UI image
docker build -f Dockerfile.streamlit -t sba-streamlit:latest .
docker tag sba-streamlit:latest $ECR_REGISTRY/sba-streamlit:latest
docker push $ECR_REGISTRY/sba-streamlit:latest
```

---

## ECS Service Configuration

### Step 1: Register Task Definitions

```bash
# Register API task definition
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs-task-api.json \
  --region ap-south-2

# Register UI task definition
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs-task-ui.json \
  --region ap-south-2
```

### Step 2: Create ECS Services

#### Option A: Using AWS Console (Recommended for first-time setup)

1. **Navigate to ECS Console** → Clusters → `sba-cluster`

2. **Create API Service:**
   - Click "Create" under Services
   - Launch type: Fargate
   - Task definition: `sba-loan-api:latest`
   - Service name: `sba-loan-api-service`
   - Number of tasks: 1
   - VPC: Select your VPC
   - Subnets: Select at least 2 subnets
   - Security group: Allow inbound on port 8000
   - Load balancer: Create new ALB or use existing
     - Target group: Create new (HTTP, port 8000)
     - Health check path: `/health`
   - Service discovery: Enable (optional)
   - Create service

3. **Create UI Service:**
   - Same steps as API service
   - Task definition: `sba-loan-ui:latest`
   - Service name: `sba-loan-ui-service`
   - Port: 8501
   - Health check path: `/_stcore/health`

#### Option B: Using AWS CLI

```bash
# Create API service (replace placeholders with your values)
aws ecs create-service \
  --cluster sba-cluster \
  --service-name sba-loan-api-service \
  --task-definition sba-loan-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:ap-south-2:xxx:targetgroup/xxx,containerName=sba-loan-api,containerPort=8000" \
  --region ap-south-2

# Create UI service
aws ecs create-service \
  --cluster sba-cluster \
  --service-name sba-loan-ui-service \
  --task-definition sba-loan-ui \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:ap-south-2:xxx:targetgroup/xxx,containerName=sba-loan-ui,containerPort=8501" \
  --region ap-south-2
```

### Step 3: Update UI Task Definition with ALB DNS (Manual Deployment Only)

**For Local/Manual Deployment:**

After creating the ALB (see next section), you need to update the UI task definition with the actual ALB DNS:

```bash
# Run the helper script
bash scripts/update_ui_task_def.sh
```

This script will:
1. Fetch the ALB DNS automatically
2. Replace the `${ALB_DNS}` placeholder with the actual DNS
3. Register the new task definition
4. Clean up temporary files

**For GitHub Actions Deployment:**

The CI/CD pipeline handles this automatically - no manual intervention needed!

### Step 4: Configure Application Load Balancer (ALB)

1. **Create ALB** (if not exists):
   - Type: Application Load Balancer
   - Scheme: Internet-facing
   - Listeners: HTTP (80) and/or HTTPS (443)
   - Availability Zones: Select at least 2

2. **Create Target Groups:**
   - API target group:
     - Protocol: HTTP, Port: 8000
     - Target type: IP
     - Health check: `/health`
     - Success codes: 200
   - UI target group:
     - Protocol: HTTP, Port: 8501
     - Target type: IP
     - Health check: `/_stcore/health`
     - Success codes: 200

3. **Configure Listener Rules:**
   - Path `/api/*` → API target group
   - Path `/*` → UI target group (default)

**Important:** After ALB creation, the ALB DNS will be automatically used:
- **GitHub Actions**: Workflow fetches ALB DNS and updates task definition on every deployment
- **Manual**: Run `bash scripts/update_ui_task_def.sh` to update the UI task definition

---

## GitHub Actions CI/CD

### Step 1: Configure GitHub Secrets

Navigate to your repository → Settings → Secrets and variables → Actions

Add the following secrets:

```
AWS_ACCESS_KEY_ID        = <your-aws-access-key>
AWS_SECRET_ACCESS_KEY    = <your-aws-secret-key>
AWS_ACCOUNT_ID           = <your-12-digit-account-id>
```

### Step 2: Workflow Configuration

Two workflows are configured:

#### 1. Test Workflow (`.github/workflows/test.yml`)
- **Triggers:** Push to `dev`/`main`, Pull requests
- **Actions:**
  - Run Python tests with pytest
  - Lint code with flake8
  - Test Docker builds
  - Upload coverage to Codecov

#### 2. Deploy Workflow (`.github/workflows/deploy.yml`)
- **Triggers:** Push to `main`, Manual dispatch
- **Actions:**
  - Build Docker images (multi-arch AMD64 for Fargate)
  - Push to ECR
  - Fetch ALB DNS and replace placeholder in UI task definition
  - Update ECS task definitions with new image URIs
  - Deploy to ECS services
  - Wait for service stability

### Step 3: Trigger Deployment

```bash
# Commit and push to main branch
git add .
git commit -m "Deploy to production"
git push origin main

# Or manually trigger via GitHub UI:
# Actions → Deploy to AWS ECS → Run workflow
```

### Step 4: Monitor Deployment

```bash
# Watch deployment progress
aws ecs describe-services \
  --cluster sba-cluster \
  --services sba-loan-api-service sba-loan-ui-service \
  --region ap-south-2

# Check task status
aws ecs list-tasks \
  --cluster sba-cluster \
  --service-name sba-loan-api-service \
  --region ap-south-2
```

---

## Monitoring & Troubleshooting

### CloudWatch Logs

```bash
# View API logs
aws logs tail /ecs/sba-loan-api --follow --region ap-south-2

# View UI logs
aws logs tail /ecs/sba-loan-ui --follow --region ap-south-2

# Filter for errors
aws logs filter-log-events \
  --log-group-name /ecs/sba-loan-api \
  --filter-pattern "ERROR" \
  --region ap-south-2
```

### ECS Service Health

```bash
# Check service status
aws ecs describe-services \
  --cluster sba-cluster \
  --services sba-loan-api-service \
  --region ap-south-2 \
  --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'

# Check task health
aws ecs describe-tasks \
  --cluster sba-cluster \
  --tasks $(aws ecs list-tasks --cluster sba-cluster --service-name sba-loan-api-service --query 'taskArns[0]' --output text --region ap-south-2) \
  --region ap-south-2
```

### Common Issues

#### 1. Task Fails to Start
**Symptom:** Tasks transition from PENDING → STOPPED

**Check:**
```bash
# View stopped task reason
aws ecs describe-tasks \
  --cluster sba-cluster \
  --tasks <task-id> \
  --region ap-south-2 \
  --query 'tasks[0].stoppedReason'
```

**Common causes:**
- IAM role missing or incorrect
- Security group blocking port access
- Image pull failure (check ECR permissions)
- Insufficient resources (CPU/memory)

#### 2. Health Check Failures
**Symptom:** Tasks continuously restart

**Check:**
```bash
# View CloudWatch logs for health check errors
aws logs tail /ecs/sba-loan-api --follow --region ap-south-2 | grep health
```

**Common causes:**
- Application not listening on correct port
- Health endpoint returning non-200 status
- Startup time exceeds health check start period
- Dependencies not available (e.g., S3 artifacts)

#### 3. S3 Artifact Download Fails
**Symptom:** API fails to start, logs show "artifacts missing"

**Check:**
```bash
# Verify artifacts exist in S3
aws s3 ls s3://sba-credit-risk-artifacts-sagar/models/ --region ap-south-2

# Test IAM role S3 access
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::<account-id>:role/sba-loan-api-task-role \
  --action-names s3:GetObject s3:ListBucket \
  --resource-arns "arn:aws:s3:::sba-credit-risk-artifacts-sagar/*"
```

**Solutions:**
- Upload artifacts: `python scripts/push_artifacts.py`
- Verify IAM role has S3 read permissions
- Check S3 bucket policy

#### 4. GitHub Actions Deploy Fails
**Symptom:** Workflow fails at deploy step

**Check:**
- GitHub secrets configured correctly
- ECS services created and running
- Task definitions registered
- IAM user has ECS deployment permissions

---

## Cost Optimization

### Estimated Monthly Costs (24/7 operation in ap-south-2)

| Resource | Configuration | Estimated Cost |
|----------|--------------|----------------|
| Fargate API | 1 vCPU, 2GB RAM | ~$35/month |
| Fargate UI | 0.5 vCPU, 1GB RAM | ~$18/month |
| S3 Storage | < 1GB | < $1/month |
| CloudWatch Logs | ~1GB/month | ~$1/month |
| ALB | 1 ALB | ~$20/month |
| **Total** | | **~$75/month** |

### Cost Reduction Strategies

1. **Auto-scaling:** Scale down during off-peak hours
2. **Spot instances:** Use Fargate Spot (not recommended for production)
3. **Log retention:** Set CloudWatch log retention to 7-30 days
4. **Development environment:** Use Docker Compose locally instead of ECS

---

## Next Steps

1. ✅ Set up custom domain with Route 53
2. ✅ Enable HTTPS with ACM certificate
3. ✅ Configure auto-scaling policies
4. ✅ Set up CloudWatch alarms for monitoring
5. ✅ Implement blue/green deployments
6. ✅ Add WAF for security
7. ✅ Enable container insights for detailed metrics

---

## Support

For issues or questions:
1. Check CloudWatch logs
2. Review AWS ECS console for task/service status
3. Verify security groups and networking
4. Test locally with Docker Compose first

---

**Last Updated:** 2026-01-08
**Version:** 1.0.0
