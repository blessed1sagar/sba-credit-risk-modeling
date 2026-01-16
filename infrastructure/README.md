# Infrastructure Configuration Files

This directory contains AWS ECS task definition files for deploying the SBA Loan ML application.

## Files

### Task Definitions

- **`ecs-task-api.json`**: API service task definition (FastAPI backend)
  - CPU: 1024 (1 vCPU)
  - Memory: 2048 MB (2 GB)
  - Port: 8000
  - S3 sync enabled for model artifacts

- **`ecs-task-ui.json`**: UI service task definition (Streamlit frontend)
  - CPU: 512 (0.5 vCPU)
  - Memory: 1024 MB (1 GB)
  - Port: 8501
  - **Important**: Uses `${ALB_DNS}` placeholder for API_URL (see below)

## Dynamic ALB DNS Configuration

### The `${ALB_DNS}` Placeholder

The UI task definition uses a **placeholder** `${ALB_DNS}/api` for the `API_URL` environment variable instead of a hardcoded value.

**Why?**
- Keeps configuration version-controlled and environment-agnostic
- Eliminates manual updates when ALB DNS changes
- Works seamlessly with automated deployments
- Prevents accidental commits of environment-specific URLs

### How It Works

#### GitHub Actions Deployment (Automated)

When you push to the `main` branch, the deployment workflow automatically:

1. Checks out the repository
2. Fetches the ALB DNS from AWS:
   ```bash
   aws elbv2 describe-load-balancers \
     --query 'LoadBalancers[?LoadBalancerName==`sba-loan-alb`].DNSName' \
     --output text
   ```
3. Replaces the placeholder in the task definition:
   ```bash
   sed -i "s|\${ALB_DNS}|http://$ALB_DNS|g" infrastructure/ecs-task-ui.json
   ```
4. Registers the updated task definition with ECS
5. Deploys the new task definition to the UI service

**No manual intervention required!**

#### Local/Manual Deployment

For local development or manual deployments, use the helper script:

```bash
bash scripts/update_ui_task_def.sh
```

This script:
- Fetches the current ALB DNS from AWS
- Creates a temporary task definition with the placeholder replaced
- Registers the new task definition
- Cleans up temporary files automatically

### Important Notes

⚠️ **DO NOT manually edit** `ecs-task-ui.json` to hardcode the ALB DNS
✅ **ALWAYS keep** the `${ALB_DNS}/api` placeholder in version control
✅ **USE the helper script** for local testing
✅ **LET GitHub Actions** handle production deployments

### Example

**Version Control (ecs-task-ui.json):**
```json
{
  "name": "API_URL",
  "value": "${ALB_DNS}/api"
}
```

**After Deployment (in ECS):**
```json
{
  "name": "API_URL",
  "value": "http://sba-loan-alb-123456789.ap-south-2.elb.amazonaws.com/api"
}
```

## Troubleshooting

### UI Cannot Connect to API

**Symptom**: UI loads but predictions fail

**Possible Causes**:
1. ALB DNS placeholder not replaced
2. ALB listener rules not configured correctly
3. Target groups not healthy

**Solution**:
```bash
# Check current UI task definition
aws ecs describe-task-definition \
  --task-definition sba-loan-ui \
  --query 'taskDefinition.containerDefinitions[0].environment'

# If you see ${ALB_DNS}, run the helper script
bash scripts/update_ui_task_def.sh

# Update the service to use new task definition
aws ecs update-service \
  --cluster sba-cluster \
  --service sba-loan-ui-service \
  --force-new-deployment
```

### Helper Script Fails

**Symptom**: `Could not find ALB with name 'sba-loan-alb'`

**Solution**:
1. Verify ALB exists:
   ```bash
   aws elbv2 describe-load-balancers --region ap-south-2
   ```
2. If ALB has a different name, update `ALB_NAME` in `scripts/update_ui_task_def.sh`

## Additional Resources

- [AWS ECS Task Definitions Documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html)
- [Application Load Balancer Guide](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)
- [Main Deployment Guide](../DEPLOYMENT.md)
