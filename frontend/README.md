# Frontend Deployment

This frontend is a Vite React app intended to be deployed as static assets on S3 behind CloudFront.

## Local development

```sh
npm install
npm run dev
```

Create `frontend/.env` from `frontend/.env.example` and set:

```dotenv
VITE_AWS_REGION=ap-south-1
VITE_AWS_USER_POOL_ID=ap-south-1_example
VITE_AWS_USER_POOL_CLIENT_ID=exampleclientid
VITE_AWS_API_BASE_URL=http://localhost
```

For the backend in local development:

```sh
cd ../backend
pip install -r requirements.gpu.txt
```

If your machine does not need GPU PyTorch, use `requirements.cpu.txt` instead.

## Frontend modes

Use the mode-specific env files in this folder instead of editing one `.env` repeatedly.

- `npm run dev:local-direct`
  Uses [frontend/.env.local-direct](/d:/Dev/programming/online_complaint_system/frontend/.env.local-direct) for local frontend + direct local backend on `http://localhost:8000`
- `npm run dev:local-docker`
  Uses [frontend/.env.local-docker](/d:/Dev/programming/online_complaint_system/frontend/.env.local-docker) for local frontend + Docker Compose backend on `http://localhost`
- `npm run build:local-direct`
  Builds the frontend with [frontend/.env.local-direct](/d:/Dev/programming/online_complaint_system/frontend/.env.local-direct)
- `npm run build:local-docker`
  Builds the frontend with [frontend/.env.local-docker](/d:/Dev/programming/online_complaint_system/frontend/.env.local-docker)
- `npm run build:cloudfront-tunnel`
  Uses [frontend/.env.cloudfront-tunnel](/d:/Dev/programming/online_complaint_system/frontend/.env.cloudfront-tunnel) for CloudFront frontend + publicly tunneled local backend
- `npm run build:production`
  Uses [frontend/.env.production](/d:/Dev/programming/online_complaint_system/frontend/.env.production) for CloudFront frontend + public production backend such as EC2

Important:
`frontend/.env.local-direct` is only for local frontend development and local API testing. If you deploy to S3 or CloudFront, build with `build:production` or `build:cloudfront-tunnel` so the compiled bundle picks up the correct public API URL.

Mode mapping for your 5 scenarios:

- `1. local normal backend + local frontend`: `npm run dev:local-direct`
- `2. local normal backend + CloudFront frontend`: edit [frontend/.env.cloudfront-tunnel](/d:/Dev/programming/online_complaint_system/frontend/.env.cloudfront-tunnel) with your tunnel URL, then run `npm run build:cloudfront-tunnel`
- `3. dockerized local backend + local frontend`: `npm run dev:local-docker`
- `4. dockerized local backend + CloudFront frontend`: edit [frontend/.env.cloudfront-tunnel](/d:/Dev/programming/online_complaint_system/frontend/.env.cloudfront-tunnel) with your tunnel URL, then run `npm run build:cloudfront-tunnel`
- `5. EC2 dockerized backend + CloudFront frontend`: edit [frontend/.env.production](/d:/Dev/programming/online_complaint_system/frontend/.env.production) with your public backend URL, then run `npm run build:production`

## Matching backend commands

Use these backend commands with the frontend modes above.

### 1. Local normal backend + local frontend

Backend:

```powershell
cd d:\Dev\programming\online_complaint_system\backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd d:\Dev\programming\online_complaint_system\frontend
npm.cmd run dev:local-direct
```

### 2. Local normal backend + CloudFront frontend

Backend:

```powershell
cd d:\Dev\programming\online_complaint_system\backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Expose the backend with a tunnel such as:

```powershell
ngrok http 8000
```

Put that public URL into [frontend/.env.cloudfront-tunnel](/d:/Dev/programming/online_complaint_system/frontend/.env.cloudfront-tunnel), then build:

```powershell
cd d:\Dev\programming\online_complaint_system\frontend
npm.cmd run build:cloudfront-tunnel
```

### 3. Dockerized local backend + local frontend

Backend:

```powershell
cd d:\Dev\programming\online_complaint_system\backend
docker compose up --build
```

Frontend:

```powershell
cd d:\Dev\programming\online_complaint_system\frontend
npm.cmd run dev:local-docker
```

### 4. Dockerized local backend + CloudFront frontend

Backend:

```powershell
cd d:\Dev\programming\online_complaint_system\backend
docker compose up --build
```

Expose the backend with a tunnel such as:

```powershell
ngrok http 80
```

Put that public URL into [frontend/.env.cloudfront-tunnel](/d:/Dev/programming/online_complaint_system/frontend/.env.cloudfront-tunnel), then build:

```powershell
cd d:\Dev\programming\online_complaint_system\frontend
npm.cmd run build:cloudfront-tunnel
```

### 5. EC2 dockerized backend + CloudFront frontend

On the EC2 machine:

```bash
cd /path/to/online_complaint_system/backend
docker compose up -d --build
```

Then set the EC2 or production API URL in [frontend/.env.production](/d:/Dev/programming/online_complaint_system/frontend/.env.production), and build:

```powershell
cd d:\Dev\programming\online_complaint_system\frontend
npm.cmd run build:production
```

## Production build

Set the backend URL to your EC2 Nginx endpoint, then build:

```dotenv
VITE_AWS_API_BASE_URL=https://api.example.com
```

```sh
npm run build
```

Upload the `dist` directory to an S3 bucket and serve it through CloudFront.

## Recommended S3 deploy

Use the deploy helper instead of manually copying files. It rebuilds the app, syncs `dist` to S3 with `--delete`, uploads `index.html` with no-cache headers, and can invalidate CloudFront in the same step.

PowerShell example for production:

```powershell
cd d:\Dev\programming\online_complaint_system\frontend
$env:FRONTEND_S3_BUCKET="your-frontend-bucket"
$env:FRONTEND_CLOUDFRONT_DISTRIBUTION_ID="E1234567890"
npm.cmd run deploy:s3:production
```

PowerShell example for a CloudFront frontend talking to a local tunneled backend:

```powershell
cd d:\Dev\programming\online_complaint_system\frontend
$env:FRONTEND_S3_BUCKET="your-frontend-bucket"
$env:FRONTEND_CLOUDFRONT_DISTRIBUTION_ID="E1234567890"
npm.cmd run deploy:s3:cloudfront-tunnel
```

Optional environment variables:

- `FRONTEND_S3_PREFIX` to upload into a bucket prefix instead of the bucket root
- `FRONTEND_INVALIDATION_PATHS` to override the default CloudFront invalidation path list, for example `/*`
- `AWS_PROFILE` and `AWS_REGION` if you want the deploy script to target a specific AWS CLI profile or region

If you already built locally and only want to upload the existing `dist` output, run:

```powershell
node .\scripts\deploy-s3.mjs --skip-build --bucket your-frontend-bucket
```

## Production architecture

- Frontend static files: S3
- CDN and TLS: CloudFront
- API origin: Dockerized FastAPI on EC2 behind Nginx
- Auth: Cognito
- File uploads: S3 presigned URLs generated by the backend
