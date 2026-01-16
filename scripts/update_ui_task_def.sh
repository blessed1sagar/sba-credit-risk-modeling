#!/bin/bash
# =============================================================================
# Helper Script: Update UI Task Definition with ALB DNS
# =============================================================================
# This script fetches the ALB DNS dynamically and replaces the ${ALB_DNS}
# placeholder in the UI task definition file for local development/testing.
#
# Usage:
#   bash scripts/update_ui_task_def.sh
#
# What it does:
#   1. Fetches ALB DNS from AWS
#   2. Creates a temporary task definition with ALB DNS replaced
#   3. Registers the new task definition
#   4. Cleans up temporary file
# =============================================================================

set -e

REGION="ap-south-2"
ALB_NAME="sba-loan-alb"
TASK_DEF_FILE="infrastructure/ecs-task-ui.json"
TEMP_TASK_DEF="infrastructure/ecs-task-ui-local.json"

echo "========================================="
echo "UI Task Definition Update Helper"
echo "========================================="
echo ""

# Step 1: Fetch ALB DNS
echo "[1/4] Fetching ALB DNS name..."
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --region "$REGION" \
  --query "LoadBalancers[?LoadBalancerName==\`$ALB_NAME\`].DNSName" \
  --output text)

if [ -z "$ALB_DNS" ]; then
  echo "❌ Error: Could not find ALB with name '$ALB_NAME' in region '$REGION'"
  echo "   Please ensure the ALB exists or update ALB_NAME in this script."
  exit 1
fi

echo "✓ ALB DNS: $ALB_DNS"
echo ""

# Step 2: Create temporary task definition with placeholder replaced
echo "[2/4] Creating temporary task definition..."
cp "$TASK_DEF_FILE" "$TEMP_TASK_DEF"
sed -i.bak "s|\${ALB_DNS}|http://$ALB_DNS|g" "$TEMP_TASK_DEF"
rm -f "$TEMP_TASK_DEF.bak"  # Remove backup file created by sed on macOS
echo "✓ Replaced \${ALB_DNS} with http://$ALB_DNS"
echo ""

# Step 3: Register task definition
echo "[3/4] Registering task definition..."
TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json file://"$TEMP_TASK_DEF" \
  --region "$REGION" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

echo "✓ Task definition registered: $TASK_DEF_ARN"
echo ""

# Step 4: Clean up temporary file
echo "[4/4] Cleaning up..."
rm -f "$TEMP_TASK_DEF"
echo "✓ Removed temporary file"
echo ""

echo "========================================="
echo "✓ Update Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Update your ECS service to use the new task definition:"
echo "     aws ecs update-service \\"
echo "       --cluster sba-cluster \\"
echo "       --service sba-loan-ui-service \\"
echo "       --task-definition sba-loan-ui \\"
echo "       --region $REGION"
echo ""
echo "  2. Or let the service pick it up automatically on next deployment"
echo ""
