# CampusVoice Cloud AI

CampusVoice is a complaint management system for educational institutions. The repository contains a React frontend, a FastAPI backend, PostgreSQL schema, AWS integrations, and ML-assisted complaint routing.

This README is written for someone cloning the repository from GitHub and setting up every local file that is intentionally excluded from Git.

## What Is In GitHub

The repository should contain:

- frontend source code
- backend source code
- deployment files
- SQL schema
- `.env.example` files
- dependency files such as `package.json`, `package-lock.json`, and Python requirements files

## What Is Excluded From GitHub

The following are intentionally excluded and must be created or provided after cloning:

- `backend/.env`
- `frontend/.env`
- `backend/outputs/` model folders
- local datasets and generated feedback files under `backend/data`
- virtual environments such as `.venv/` or `backend/env/`
- `frontend/node_modules/`
- `frontend/dist/`
- private keys such as `*.pem`
- the entire `paper/` folder

## Project Structure

Main folders:

```text
online_complaint_system/
|-- backend/
|   |-- api/
|   |-- data/
|   |-- deploy/
|   |-- outputs/
|   |-- scripts/
|   |-- sql/
|   |-- src/
|   |-- .env.example
|   |-- compose.yaml
|   |-- Dockerfile
|   `-- requirements.cpu.txt
|-- frontend/
|   |-- public/
|   |-- src/
|   |-- .env.example
|   `-- package.json
`-- README.md
```

Important local-only paths after setup:

```text
backend/.env
frontend/.env
backend/outputs/general_complaint_model/
backend/outputs/edu_classifier_multitask/
backend/data/
```

## Prerequisites

Install these first:

1. Python 3.12 or compatible
2. Node.js and npm
3. PostgreSQL
4. Git
5. Optional: Docker Desktop
6. Optional: AWS account with Cognito and S3

## Step 1: Clone The Repository

```bash
git clone https://github.com/rahulrrs/campusvoice-cloud-ai.git
cd campusvoice-cloud-ai
```

## Step 2: Create The Backend Environment File

Create this file:

```text
backend/.env
```

You can create it from the example:

```bash
cd backend
cp .env.example .env
```

Add these values to `backend/.env`:

```dotenv
AWS_REGION=ap-south-1

COGNITO_USER_POOL_ID=your_user_pool_id
COGNITO_APP_CLIENT_ID=your_app_client_id
ADMIN_EMAILS=admin@example.com
SUPER_ADMIN_EMAILS=superadmin@example.com

RDS_HOST=localhost
RDS_PORT=5432
RDS_DATABASE=complaints
RDS_USER=postgres
RDS_PASSWORD=postgres

ATTACHMENTS_BUCKET=your-attachments-bucket
PRESIGNED_URL_EXPIRES_SECONDS=900

CORS_ALLOW_ORIGIN=http://localhost:5173,http://127.0.0.1:5173

BACKBONE_MODEL_NAME=distilbert-base-uncased
BACKBONE_MODEL_DIR=outputs/general_complaint_model

CHATBOT_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

MODEL_SYNC_ON_STARTUP=false
MODEL_SYNC_MODE=missing
MODEL_S3_BUCKET=your-model-bucket
MODEL_S3_PREFIX=models
MODEL_REQUIRED_DIRS=general_complaint_model,edu_classifier_multitask
```

What each part is for:

- `COGNITO_*`: required for authentication
- `RDS_*`: required for PostgreSQL connection
- `ATTACHMENTS_BUCKET`: required for S3 complaint file uploads
- `CORS_ALLOW_ORIGIN`: allows the frontend to call the backend
- `BACKBONE_MODEL_*`: controls backbone model loading
- `GEMINI_*`: optional chatbot provider setup
- `MODEL_SYNC_*`: used if models are stored in S3 instead of local disk

## Step 3: Create The Frontend Environment File

Create this file:

```text
frontend/.env
```

You can create it from the example:

```bash
cd frontend
cp .env.example .env
```

Add these values to `frontend/.env`:

```dotenv
VITE_AWS_REGION=ap-south-1
VITE_AWS_USER_POOL_ID=your_user_pool_id
VITE_AWS_USER_POOL_CLIENT_ID=your_app_client_id
VITE_API_BASE_URL=http://localhost:8000
VITE_ADMIN_EMAILS=admin@example.com
VITE_API_TIMEOUT_MS=30000
```

Optional frontend values for Google OAuth:

```dotenv
VITE_COGNITO_OAUTH_DOMAIN=your-domain.auth.ap-south-1.amazoncognito.com
VITE_COGNITO_REDIRECT_SIGN_IN=http://localhost:5173
VITE_COGNITO_REDIRECT_SIGN_OUT=http://localhost:5173
VITE_GOOGLE_OAUTH_ENABLED=false
```

What each part is for:

- `VITE_AWS_*`: Cognito setup for frontend login
- `VITE_API_BASE_URL`: backend API base URL
- `VITE_ADMIN_EMAILS`: admin email list used in frontend behavior
- `VITE_API_TIMEOUT_MS`: request timeout

## Step 4: Create The PostgreSQL Database

Create a PostgreSQL database named `complaints` or update `backend/.env` to match your actual database name.

Then run the schema from [`backend/sql/schema.sql`](/d:/Dev/programming/online_complaint_system/backend/sql/schema.sql).

Example:

```bash
psql -U postgres -d complaints -f backend/sql/schema.sql
```

This creates the required tables for:

- complaints
- complaint updates
- audit log
- admin users

## Step 5: Set Up The Excluded Model Files

The trained model files are not stored in GitHub. You must provide them yourself.

Create these folders locally:

```text
backend/outputs/general_complaint_model/
backend/outputs/edu_classifier_multitask/
```

Recommended contents:

- tokenizer files
- config files
- model weights
- metadata files used by your trained classifier

Best place to store model files:

1. Local development:
   `backend/outputs/...`
2. Shared or production use:
   S3 bucket

Recommended production storage:

- store large model files in S3
- keep GitHub only for source code
- sync models into `backend/outputs` at runtime

Suggested S3 structure:

```text
s3://your-bucket/models/general_complaint_model/...
s3://your-bucket/models/edu_classifier_multitask/...
```

If you use S3, set in `backend/.env`:

```dotenv
MODEL_SYNC_ON_STARTUP=true
MODEL_SYNC_MODE=missing
MODEL_S3_BUCKET=your-model-bucket
MODEL_S3_PREFIX=models
MODEL_REQUIRED_DIRS=general_complaint_model,edu_classifier_multitask
```

## Step 6: Set Up Excluded Data Files

The files under `backend/data` are mostly local-only files for training, evaluation, and feedback collection. They are excluded from GitHub.

Possible local files include:

```text
backend/data/dataset.csv
backend/data/dataset.xlsx
backend/data/dataset_clean.csv
backend/data/dataset_corrected.csv
backend/data/train.csv
backend/data/val.csv
backend/data/test.csv
backend/data/pseudo_feedback.csv
backend/data/frontend_feedback.csv
```

Use them like this:

- `dataset.csv` or `dataset.xlsx`: original raw dataset
- `dataset_clean.csv`: cleaned dataset
- `dataset_corrected.csv`: reviewed or corrected dataset
- `train.csv`, `val.csv`, `test.csv`: train-validation-test splits
- `pseudo_feedback.csv`: pseudo-labeled feedback
- `frontend_feedback.csv`: feedback generated by application usage

Best place to store large datasets:

1. Local disk for development
2. S3 or other private object storage for shared team access

Do not store private or sensitive complaint data in GitHub.

## Step 7: Create Local Python Environment

From the project root or backend folder, create a virtual environment.

Example:

```bash
cd backend
python -m venv .venv
```

Activate it:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install backend dependencies:

CPU version:

```bash
pip install -r requirements.cpu.txt
```

GPU version:

```bash
pip install -r requirements.gpu.txt
```

## Step 8: Install Frontend Dependencies

From the frontend folder:

```bash
cd frontend
npm install
```

This recreates the excluded `frontend/node_modules/` folder locally.

## Step 9: Start The Backend

From the backend folder:

```bash
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The backend will use:

- `backend/.env`
- PostgreSQL
- local models in `backend/outputs` or S3 model sync

Backend default URL:

```text
http://localhost:8000
```

## Step 10: Start The Frontend

From the frontend folder:

```bash
cd frontend
npm run dev
```

Frontend default URL:

```text
http://localhost:5173
```

## Step 11: Docker Setup

If you want to run the backend with Docker instead of a local Python process:

```bash
cd backend
docker compose up --build
```

This uses:

- `backend/.env`
- local `backend/outputs`
- local `backend/data`

## Recommended Storage For Large Files

Use this rule:

- GitHub: source code, config templates, docs
- local disk: local dev models and temporary datasets
- S3: production models, shared large datasets, complaint attachments
- secrets store or `.env`: API keys and credentials

For this project:

- model files should go in `backend/outputs` locally or in S3 for production
- datasets should stay outside GitHub and preferably in local private storage or S3
- `.env` files should always stay local and never be committed

## If You Want To Clean Already Tracked Files

If old excluded files were already tracked in Git before these rules, remove them from tracking with:

```bash
git rm -r --cached backend/outputs
git rm --cached backend/.env frontend/.env
git rm --cached backend/data/dataset.csv
git rm --cached backend/data/dataset.xlsx
git rm --cached backend/data/dataset_clean.csv
git rm --cached backend/data/dataset_corrected.csv
git rm --cached backend/data/train.csv
git rm --cached backend/data/val.csv
git rm --cached backend/data/test.csv
git rm --cached backend/data/pseudo_feedback.csv
git rm -r --cached paper
```

Then commit:

```bash
git commit -m "Remove excluded local files from Git tracking"
```

## Minimum Setup Checklist

To get the app running after cloning:

1. Create `backend/.env`
2. Create `frontend/.env`
3. Create PostgreSQL database and run schema
4. Place model files in `backend/outputs` or configure S3 sync
5. Add any needed local dataset files in `backend/data`
6. Create Python virtual environment
7. Install backend dependencies
8. Install frontend dependencies
9. Start backend
10. Start frontend

If models are missing, prediction features will not work.
If database settings are wrong, complaint storage will not work.
If Cognito settings are missing, authentication will not work.
