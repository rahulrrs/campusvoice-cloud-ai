# EC2 Setup Guide

## 1) Values you must fill

### Cognito
- `COGNITO_USER_POOL_ID`: from Cognito User Pool (example: `us-east-1_abc123`)
- `COGNITO_APP_CLIENT_ID`: from Cognito App clients

### RDS PostgreSQL
- `RDS_HOST`: writer endpoint from RDS/Aurora
- `RDS_PORT`: usually `5432`
- `RDS_DATABASE`: database name (example: `complaints`)
- `RDS_USER`: db username
- `RDS_PASSWORD`: db password

### S3
- `ATTACHMENTS_BUCKET`: bucket name for uploaded attachments

### Other
- `AWS_REGION`: same region as Cognito/S3/RDS
- `CORS_ALLOW_ORIGIN`: frontend origin (for local dev use `http://localhost:8080`)
- `PRESIGNED_URL_EXPIRES_SECONDS`: URL expiry in seconds, default `900`

## 2) Database schema

Run:

```sql
\i backend/sql/schema.sql
```

Or copy/paste `backend/sql/schema.sql` in pgAdmin/RDS Query Editor.

## 3) Run backend locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# fill .env values
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API base URL becomes `http://localhost:8000`.

## 4) Run frontend locally

```bash
cd frontend
cp .env.example .env
# set VITE_AWS_API_BASE_URL=http://localhost:8000
# set Cognito values to match backend
npm install
npm run dev
```

Frontend runs on `http://localhost:8080` (or Vite-selected port).

## 5) Deploy backend to EC2

1. Launch EC2 (Ubuntu recommended), attach IAM role with:
   - `s3:PutObject`, `s3:GetObject` on your bucket
2. Open security groups:
   - inbound `80/443` (public)
   - inbound `8000` only if testing directly (optional)
   - outbound allowed to RDS/Cognito/S3
3. SSH into EC2 and deploy:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv nginx
git clone <your-repo-url>
cd online_complaint_system/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill real values
```

4. Create systemd service `/etc/systemd/system/campusvoice-api.service`:

```ini
[Unit]
Description=CampusVoice FastAPI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/online_complaint_system/backend
EnvironmentFile=/home/ubuntu/online_complaint_system/backend/.env
ExecStart=/home/ubuntu/online_complaint_system/backend/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

5. Start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable campusvoice-api
sudo systemctl start campusvoice-api
sudo systemctl status campusvoice-api
```

6. Put Nginx in front (domain + TLS with Certbot recommended), proxy to `127.0.0.1:8000`.

## 6) Frontend in AWS

You can host frontend on:
- S3 + CloudFront (recommended), or
- EC2/Nginx

Set frontend env:
- `VITE_AWS_API_BASE_URL=https://<your-api-domain>`
- `VITE_AWS_REGION=<region>`
- `VITE_AWS_USER_POOL_ID=<pool_id>`
- `VITE_AWS_USER_POOL_CLIENT_ID=<app_client_id>`
