#!/bin/bash
set -e

echo "🚀 Starting AWS EC2 Provisioning for Bragi RAG Pipeline..."

# 1. Variables
KEY_NAME="bragi-key"
SG_NAME="bragi-sg-backend"
INSTANCE_TYPE="t3.large"
REGION="us-east-1"

# Fetch latest Ubuntu 24.04/22.04 AMI for the configured region
AMI_ID=$(aws ec2 describe-images --region $REGION --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)

echo "✅ Found Ubuntu AMI: $AMI_ID in region $REGION"

# 2. Create Key Pair
if ! aws ec2 describe-key-pairs --region $REGION --key-names $KEY_NAME > /dev/null 2>&1; then
    echo "🔑 Creating Key Pair: $KEY_NAME..."
    rm -f ~/.ssh/$KEY_NAME.pem
    aws ec2 create-key-pair --region $REGION --key-name $KEY_NAME --query 'KeyMaterial' --output text > ~/.ssh/$KEY_NAME.pem
    chmod 400 ~/.ssh/$KEY_NAME.pem
else
    echo "✅ Key Pair $KEY_NAME already exists."
fi

# 3. Create Security Group
SG_ID=$(aws ec2 describe-security-groups --region $REGION --group-names $SG_NAME --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "NONE")
if [ "$SG_ID" == "NONE" ]; then
    echo "🛡️ Creating Security Group: $SG_NAME..."
    SG_ID=$(aws ec2 create-security-group --region $REGION --group-name $SG_NAME --description "Bragi Backend SG" --query 'GroupId' --output text)
    
    # Allow SSH (22)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
    
    # Allow Orchestrator WebSocket (8000)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
    
    # Allow ASR WebSocket (8001)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 8001 --cidr 0.0.0.0/0

    echo "✅ Security Group created and rules added: $SG_ID"
else
    echo "✅ Security Group $SG_NAME already exists: $SG_ID"
fi

# 4. Launch EC2 Instance
INSTANCE_ID=$(aws ec2 describe-instances --region $REGION --filters "Name=tag:Name,Values=Bragi-Backend" "Name=instance-state-name,Values=running,pending" --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "NONE")
if [ "$INSTANCE_ID" == "NONE" ] || [ "$INSTANCE_ID" == "None" ] || [ -z "$INSTANCE_ID" ]; then
    echo "🖥️ Launching EC2 Instance ($INSTANCE_TYPE)..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --region $REGION \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Bragi-Backend}]' \
        --query 'Instances[0].InstanceId' \
        --output text)
else
    echo "✅ EC2 Instance already running: $INSTANCE_ID"
fi

echo "⏳ Waiting for instance $INSTANCE_ID to be running..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "✅ Instance $INSTANCE_ID is running at IP: $PUBLIC_IP"

echo "⏳ Waiting 30s for SSH to become available..."
sleep 30

# 5. Install Docker on EC2
echo "🐳 Installing Docker on EC2..."
ssh -o "StrictHostKeyChecking no" -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP << 'EOF'
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker ubuntu
EOF

echo "✅ Docker installed."

# 6. Copy Source Code to EC2
echo "📁 Copying source code to EC2..."
rsync -avz -e "ssh -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -i ~/.ssh/$KEY_NAME.pem" --exclude '.venv' --exclude 'models' --exclude 'node_modules' --exclude 'frontend/dist' --exclude 'target' --exclude '.git' ../Bragi/ ubuntu@$PUBLIC_IP:~/Bragi/

# Copy ~/.aws credentials to the EC2 instance so Bedrock works
rsync -avz -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/$KEY_NAME.pem" ~/.aws ubuntu@$PUBLIC_IP:~/

# 7. Start Docker Compose on EC2
echo "🚀 Starting backend services on EC2..."
ssh -o "StrictHostKeyChecking no" -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP << 'EOF'
    cd ~/Bragi
    # Setup .env for production
    echo "ENVIRONMENT=production" >> .env
    echo "SARVAM_API_KEY=sk_foh0jvl2_hnnna9DHfPp85fnMeUld0cdb" >> .env
    
    # Mount AWS credentials into generation-service via docker-compose override or env
    # Since generation-service is built via docker, we'll just run it
    # Note: docker compose by default doesn't mount ~/.aws, we should modify docker-compose.yml to mount it or pass AWS_ACCESS_KEY_ID via .env
    
    sudo docker compose up -d --build
EOF

echo "🎉 Backend Deployment Complete!"
echo "📡 Orchestrator: ws://$PUBLIC_IP:8000/ws"
echo "🎤 ASR Service: ws://$PUBLIC_IP:8001/ws"
echo ""
echo "📝 Vercel environment variables to set:"
echo "VITE_ORCHESTRATOR_WS_URL=ws://$PUBLIC_IP:8000/ws"
echo "VITE_ASR_WS_URL=ws://$PUBLIC_IP:8001/ws"
