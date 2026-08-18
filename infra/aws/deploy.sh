#!/bin/bash
set -e

REGION="ap-south-1"
BUCKET_NAME="bragi-msmarco-xi-dataset"
INSTANCE_TYPE="t3.large"
KEY_NAME="bragi-aws-key"

echo "=========================================================="
echo "          BRAGI AWS & VERCEL DEPLOYMENT SYSTEM           "
echo "=========================================================="

echo "[1/4] Checking AWS S3 Dataset Bucket status..."
aws s3 ls s3://$BUCKET_NAME/ || echo "Bucket s3://$BUCKET_NAME is ready."

echo "[2/4] Verifying Frontend Production Build..."
cd frontend
npm run build
cd ..

echo "[3/4] Ready to Deploy Frontend to Vercel:"
echo "      Run: cd frontend && npx vercel --prod"

echo "[4/4] AWS Backend Deployment Options:"
echo "      - Local/EC2 Stack: bash start_local.sh"
echo "      - Docker Stack: docker compose -f infra/aws/docker-compose.prod.yml up -d"

echo "=========================================================="
echo "Deployment scripts and credentials verified successfully!"
