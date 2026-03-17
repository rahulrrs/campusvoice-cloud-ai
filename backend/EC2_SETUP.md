# Deployment Guide

This project supports mixed deployment modes:

- Local frontend + local backend
- S3/CloudFront frontend + local backend
- Local frontend + EC2 backend
- S3/CloudFront frontend + EC2 backend

Across those modes, Cognito, PostgreSQL, attachments, and model artifacts can stay on AWS as long as the backend can reach them.

Common combinations:

- Local backend + AWS Cognito + RDS + S3 attachments + S3 model sync
- EC2 backend + AWS Cognito + RDS + S3 attachments + S3 model sync
- Local backend + local Postgres + local model files + optional S3 attachments
- EC2 backend image from ECR + AWS Cognito + RDS + S3 attachments + optional S3 model sync

## 1) Required backend environment

### Cognito
- `COGNITO_USER_POOL_ID`: Cognito user pool ID
- `COGNITO_APP_CLIENT_ID`: Cognito app client ID

### RDS PostgreSQL
- `RDS_HOST`: RDS instance or cluster writer endpoint
- `RDS_PORT`: usually `5432`
- `RDS_DATABASE`: database name, typically `postgres`
- `RDS_USER`: database username
- `RDS_PASSWORD`: database password

### S3
- `ATTACHMENTS_BUCKET`: bucket name for complaint uploads

### Other
- `AWS_REGION`: same region as Cognito/S3/RDS
- `CORS_ALLOW_ORIGIN`: CloudFront frontend URL, for example `https://app.example.com`
- `PRESIGNED_URL_EXPIRES_SECONDS`: presigned upload/download URL TTL
- `ADMIN_EMAILS`: comma-separated admin emails

### Optional model sync from S3
- `MODEL_SYNC_ON_STARTUP=true` to download model artifacts before Uvicorn starts
- `MODEL_SYNC_MODE=missing` to only download missing model folders, or `always` to force re-download
- `MODEL_S3_BUCKET`: bucket containing your model artifacts
- `MODEL_S3_PREFIX`: prefix inside the bucket, for example `models`
- `MODEL_REQUIRED_DIRS`: comma-separated model folders to sync into `backend/outputs`

Example S3 layout:

```text
s3://your-model-bucket/models/distilbert_cfpb_mlm/...
s3://your-model-bucket/models/edu_classifier_multitask/...
```

## 2) Prepare the database

Run the schema in [sql/schema.sql](/d:/Dev/programming/online_complaint_system/backend/sql/schema.sql).

For `psql`:

```sql
\i backend/sql/schema.sql
```

Or paste the file into pgAdmin or any PostgreSQL client connected to RDS.

## 3) Run the backend locally with Docker

From the repository root:

```bash
cd backend
cp .env.example .env
# fill real values
docker compose up --build
```

API base URL becomes `http://localhost`.

Docker uses [requirements.cpu.txt](/d:/Dev/programming/online_complaint_system/backend/requirements.cpu.txt), so EC2 gets CPU-only PyTorch by default.

## 4) Run the frontend locally

```bash
cd frontend
cp .env.example .env
```

Set:

- `VITE_API_BASE_URL=http://localhost:8000`
- `VITE_AWS_REGION=<region>`
- `VITE_AWS_USER_POOL_ID=<pool_id>`
- `VITE_AWS_USER_POOL_CLIENT_ID=<app_client_id>`

Then run:

```bash
npm install
```

Then start Vite:

```bash
npm run dev
```

If the frontend will be hosted on S3 or CloudFront while your backend runs locally, keep `VITE_API_BASE_URL` pointed at your public local tunnel or your LAN-reachable backend URL instead of `localhost`.

## 5) Deploy the backend to EC2

1. Launch an Ubuntu EC2 instance in the same VPC/subnets that can reach RDS.
2. Attach an IAM role that can access the attachments bucket:
   - `s3:GetObject`
   - `s3:PutObject`
   If you also store model artifacts in S3, include `s3:GetObject` for the model bucket or prefix too.
3. Open EC2 security group ports:
   - `80` from the internet
   - `443` from the internet
   - `22` only from your admin IP
4. SSH into EC2 and install Docker:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

5. Clone the repo:

```bash
git clone <your-repo-url>
cd online_complaint_system/backend
cp .env.example .env
# fill real values
```

6. Build and start the backend stack:

```bash
docker compose up -d --build
```

The Docker build uses [requirements.cpu.txt](/d:/Dev/programming/online_complaint_system/backend/requirements.cpu.txt), so it installs CPU-only PyTorch for EC2.
If `MODEL_SYNC_ON_STARTUP=true`, the container entrypoint downloads the configured model folders from S3 into `backend/outputs` before starting the API.

7. Install the systemd unit from [deploy/campusvoice-api.service](/d:/Dev/programming/online_complaint_system/backend/deploy/campusvoice-api.service):

```bash
sudo cp deploy/campusvoice-api.service /etc/systemd/system/campusvoice-api.service
sudo systemctl daemon-reload
sudo systemctl enable campusvoice-api
sudo systemctl restart campusvoice-api
sudo systemctl status campusvoice-api
```

8. Point your domain DNS to the EC2 public IP or load balancer.
9. If you need HTTPS, terminate TLS with an AWS load balancer, CloudFront, or extend the Docker stack with certificates.

The Nginx container proxies traffic to the Dockerized Uvicorn process over the internal Docker network.

## 5A) Use the same backend image locally and on EC2 with ECR

This is possible with the current repo layout:

- S3 is used at runtime by the backend through `ATTACHMENTS_BUCKET`
- ECR is only used to store and distribute the Docker image
- EC2 can pull that image from ECR and still read/write S3 attachments at runtime

Local Docker with S3 attachments:

```bash
cd backend
docker compose up --build
```

That uses your local `.env`, including:

- `ATTACHMENTS_BUCKET`
- `AWS_REGION`
- AWS credentials or an AWS profile available to Docker

Build and push to ECR:

```bash
aws ecr create-repository --repository-name campusvoice-api
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-1.amazonaws.com
cd backend
docker build -t campusvoice-api:latest .
docker tag campusvoice-api:latest <account>.dkr.ecr.ap-south-1.amazonaws.com/campusvoice-api:latest
docker push <account>.dkr.ecr.ap-south-1.amazonaws.com/campusvoice-api:latest
```

Run on EC2 from ECR:

1. Set `API_IMAGE=<account>.dkr.ecr.ap-south-1.amazonaws.com/campusvoice-api:latest` in the EC2 shell environment or an env file used by Compose.
2. Keep the same backend `.env` style values for:
   - `ATTACHMENTS_BUCKET`
   - `AWS_REGION`
   - `RDS_*`
   - `COGNITO_*`
3. Attach an EC2 IAM role with:
   - `AmazonEC2ContainerRegistryReadOnly` or equivalent ECR pull permissions
   - `s3:GetObject`
   - `s3:PutObject`
   - optional model-bucket read access if using S3 model sync
4. Start the stack with:

```bash
docker compose pull api
docker compose up -d
```

For systemd on EC2, use [deploy/campusvoice-api-ecr.service](/d:/Dev/programming/online_complaint_system/backend/deploy/campusvoice-api-ecr.service) instead of the build-oriented service.

## 6) Host the frontend on S3 + CloudFront

1. In `frontend/.env`, set:
   - `VITE_API_BASE_URL=https://api.example.com`
   - `VITE_AWS_REGION=<region>`
   - `VITE_AWS_USER_POOL_ID=<pool_id>`
   - `VITE_AWS_USER_POOL_CLIENT_ID=<app_client_id>`
2. Build the frontend:

```bash
cd frontend
npm install
npm run build
```

3. Create an S3 bucket for static hosting content.
4. Upload the contents of `frontend/dist`.
5. Create a CloudFront distribution with the S3 bucket as origin.
6. If you use a custom domain, attach an ACM certificate in `us-east-1` and map the domain in CloudFront.

## 7) Point the frontend to the EC2 backend

Once the backend is live behind Nginx, set:

- `VITE_API_BASE_URL=https://api.example.com`

Then rebuild and redeploy the frontend to S3.
