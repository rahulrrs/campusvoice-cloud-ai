# CampusVoice Project Documentation

## 1. Project Overview

CampusVoice is an AI-assisted online complaint management platform designed for educational institutions. It allows students to submit complaints, track progress, receive updates, and communicate with administrators. The platform also provides admin and super-admin interfaces for complaint triage, routing, review, analytics, and access management.

This repository contains:

- A React + TypeScript frontend
- A FastAPI + Python backend
- PostgreSQL schema and data structures
- AWS-integrated authentication and storage flows
- ML/NLP utilities for complaint classification, prioritization, duplicate detection, fairness monitoring, and workflow automation
- Docker and EC2-oriented deployment artifacts

The project name used in code and docs is primarily `CampusVoice`, while the repository root is `online_complaint_system`.

## 2. Core Goals

The system is built to solve several operational problems in institutional complaint handling:

- Make complaint submission simple and accessible
- Support anonymous reporting
- Allow evidence-backed complaints using images, documents, and audio
- Keep complaint state visible to students
- Reduce admin triage effort through AI-assisted classification and routing
- Detect duplicate complaints and spam-like patterns
- Preserve auditability of automated and manual decisions
- Monitor fairness, risk, escalation, and workload
- Support cloud-hosted frontend and backend deployment on AWS-compatible infrastructure

## 3. Repository Structure

Top-level structure:

```text
online_complaint_system/
|-- aws/
|-- backend/
|-- frontend/
|-- paper/
|-- README.md
`-- PROJECT_DOCUMENTATION.md
```

### 3.1 `frontend/`

Contains the Vite React application.

Important areas:

- `src/App.tsx`: frontend application root and route composition
- `src/pages/`: screens for students, admins, and super-admins
- `src/integrations/aws/client.ts`: authentication and backend API integration
- `src/contexts/AuthContext.tsx`: auth state provider
- `src/hooks/`: complaint, access-profile, and sync hooks
- `src/offline/`: IndexedDB storage and offline sync behavior
- `src/components/`: UI building blocks and complaint/admin widgets

### 3.2 `backend/`

Contains the FastAPI application, ML logic, scripts, SQL schema, deployment files, and Docker setup.

Important areas:

- `api/main.py`: primary backend application and API surface
- `src/utils/`: shared ML and utility modules
- `sql/schema.sql`: PostgreSQL schema
- `scripts/`: training, evaluation, prediction, data cleaning, and S3 model sync
- `deploy/`: Nginx and systemd service files
- `compose.yaml`: backend Docker Compose stack
- `Dockerfile`: backend container build definition

### 3.3 `aws/`

Present in the repository but currently not used as a full Infrastructure-as-Code directory. The codebase relies on AWS resources through environment variables and deployment docs rather than Terraform/CDK/CloudFormation in this repo.

## 4. Technology Stack

### 4.1 Frontend Stack

- React 18
- TypeScript
- Vite
- React Router
- TanStack React Query
- Tailwind CSS
- shadcn/ui
- Radix UI primitives
- Framer Motion
- AWS Amplify Auth
- IndexedDB via `idb`
- Zod
- Vitest and Testing Library

### 4.2 Backend Stack

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- psycopg2
- boto3
- python-dotenv
- PyJWT

### 4.3 Database Stack

- PostgreSQL
- JSONB for semi-structured complaint analysis data
- UUID primary keys
- Triggers and indexes for update tracking and query efficiency

### 4.4 AI / ML Stack

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- safetensors
- scikit-learn
- pandas
- numpy
- sentence-transformers
- hnswlib
- langdetect
- google-genai for Gemini-backed assistant responses

### 4.5 Cloud / Deployment Stack

- AWS Cognito
- Amazon S3
- Amazon RDS PostgreSQL
- Amazon EC2
- Amazon CloudFront
- Docker
- Docker Compose
- Nginx
- systemd
- Optional Amazon ECR flow

## 5. Functional Modules

The platform is divided into five main functional layers:

1. Student-facing complaint platform
2. Admin operations console
3. Super-admin access management
4. AI automation and decision support
5. Deployment and runtime infrastructure

## 6. Frontend Architecture

### 6.1 Application Model

The frontend is a client-rendered SPA built with Vite and React. Routing is lazy-loaded for performance. Application state is split into:

- Authentication state in `AuthContext`
- Remote API state in React Query
- Local offline queue state in IndexedDB

The root app tree:

- `QueryClientProvider`
- `AuthProvider`
- `BrowserRouter`
- lazily loaded page routes

### 6.2 Main Pages

#### Public / Shared

- `Index`
- `FAQ`
- `Auth`
- `AdminLogin`
- `AdminAccess`
- `NotFound`

#### Student Pages

- `Dashboard`
- `SubmitComplaint`
- `ComplaintDetail`
- `Notifications`

#### Staff Pages

- `Admin`
- `SuperAdmin`

### 6.3 Authentication Model

Authentication uses AWS Cognito through Amplify.

Supported flows:

- Email/password sign-up
- Email verification
- Sign-in
- Password reset
- Optional Google OAuth redirect sign-in

The frontend stores a short-lived cached session in memory and uses the Cognito token for backend authorization.

### 6.4 Role Handling

The frontend does not rely only on hardcoded client-side email checks. Instead, it calls the backend `/me/access` endpoint to determine:

- whether the user is a normal user
- whether they are an admin
- whether they are a super-admin
- whether they have pending admin invites

This allows the backend to remain the source of truth for access control.

### 6.5 Complaint Submission UX

The `SubmitComplaint` page supports:

- title and description entry
- anonymous complaint toggle
- image/document upload
- audio upload
- in-browser voice recording
- online and offline submission behavior

If the browser is online:

- the frontend asks the backend for presigned S3 upload URLs
- uploads attachments directly to S3
- then submits the complaint payload to the backend

If the browser is offline:

- complaint data and queued attachment blobs are stored in IndexedDB
- a later sync process submits them when connectivity returns

### 6.6 Offline Architecture

Offline support is implemented with IndexedDB via `idb`.

Local stores:

- `pendingComplaints`
- `cachedComplaints`

Behavior:

- complaints created offline are stored locally
- once the browser regains connectivity, `useOfflineSync` and related sync hooks replay pending complaints
- attachments are uploaded first, then the complaint payload is posted to the backend

This is one of the stronger architectural features of the frontend.

### 6.7 Student Dashboard

The dashboard provides:

- total complaint count
- pending count
- in-progress count
- resolved count
- search
- category filtering
- status filtering
- complaint cards with unread-update indicators

### 6.8 Complaint Detail Page

The detail view supports:

- complaint metadata display
- status timeline
- attachment access via presigned download URLs
- student/admin conversation thread
- reopen flow for resolved complaints
- delete flow for both online and local-only complaints

### 6.9 Admin Console

The admin console is a high-value operational module. It supports:

- complaint queue browsing
- search and filter by status, department, and review state
- prediction-on-demand
- auto-apply AI classification
- bulk auto-classification
- assignment handling
- internal notes
- public updates to students
- status updates and resolution summaries
- attachment preview loading
- analytics and trend monitoring
- recent audit trail viewing

The admin UI is designed as an operational control panel rather than a passive dashboard.

### 6.10 Super-Admin Console

The super-admin interface supports:

- creating admin invites
- assigning admin or super-admin roles
- activating, suspending, or revoking access
- copying invite links
- reviewing invite state and account history

The access model is invite-based and tied to user email identity.

## 7. Backend Architecture

### 7.1 Application Style

The backend is a single FastAPI application in `backend/api/main.py`. It combines:

- API routing
- auth verification
- PostgreSQL access
- complaint workflow logic
- ML inference orchestration
- chatbot response handling
- admin analytics

This makes it monolithic in deployment style, but feature-rich in behavior.

### 7.2 Major Backend Responsibilities

- Verify Cognito bearer tokens
- Resolve user/admin/super-admin roles
- Persist and retrieve complaints
- Generate S3 presigned upload/download URLs
- Run complaint analysis and routing logic
- Maintain updates and notifications
- Log manual and automated decisions
- Manage admin access records
- Trigger retraining workflows
- Serve analytics and chatbot responses

### 7.3 Authentication And Authorization

The backend validates Cognito JWTs using JWKS fetched from the Cognito issuer URL. It builds a `CurrentUser` model and uses dependency-based access enforcement:

- `get_current_user`
- `require_admin`
- `require_super_admin`

Super-admin identity is partially bootstrap-driven from `SUPER_ADMIN_EMAILS` and can also be persisted in the `admin_users` table.

### 7.4 Backend API Surface

The API includes:

- health checks
- access profile and invite acceptance
- complaint analysis
- complaint CRUD-style operations
- complaint updates
- complaint reopen flow
- file upload/download URL generation
- admin complaint review and update endpoints
- reports export
- analytics
- audit log retrieval
- retraining trigger endpoint
- chatbot endpoint

### 7.5 Complaint Lifecycle

The complaint lifecycle includes both workflow status and automation decision state.

Workflow status values:

- `submitted`
- `pending`
- `in_progress`
- `resolved`
- `rejected`

Automation decision states include:

- `submitted`
- `auto_classified`
- `routed`
- `in_review`
- `escalated`
- `resolved`
- `reopened`
- `quarantined`

This separation is important:

- workflow status tracks operational progress
- decision state tracks automation/triage interpretation

### 7.6 Complaint Creation Flow

When a complaint is created:

1. Backend validates payload basics
2. Complaint text is analyzed using the AI bundle
3. Attachments and evidence metadata are interpreted
4. Submission guard checks for spam, abuse, and strong duplicates
5. Automation logic determines:
   - category
   - priority
   - department
   - status
   - decision state
   - risk score
   - fairness flags
   - human-review requirement
   - escalation level
   - SLA due date
6. Complaint is inserted into PostgreSQL
7. Audit log entry is written
8. Feedback row may be appended for future retraining

### 7.7 Updates And Messaging

Complaint conversations are stored in `complaint_updates`.

Supported update types:

- public student updates
- public admin updates
- internal admin-only notes

The backend also tracks last viewed timestamps for user/admin unread notification behavior.

### 7.8 Notifications

Notifications are generated from complaint activity and grouped on the frontend. The backend provides:

- unread item grouping
- mark single complaint notification as read
- mark all notifications as read

### 7.9 Audit Logging

The backend writes structured audit records for important complaint changes.

Examples include:

- automation intake
- manual route override
- admin case update
- automation refreshed
- bulk automation refresh

Each audit row can store:

- actor type
- actor id
- event type
- previous state
- new state
- reason object
- model version
- rule version

This is a strong compliance and traceability feature.

## 8. Database Design

### 8.1 Main Tables

#### `complaints`

Stores core complaint records and workflow metadata.

Important fields:

- id
- user_id
- title
- description
- category
- priority
- department
- status
- is_anonymous
- attachments
- evidence_types
- analysis
- source_language
- assigned_to
- admin_notes
- submitted/pending/in-progress/resolved timestamps
- decision_state
- risk_score
- routing_confidence
- decision_source
- decision_reason
- fairness_flags
- requires_human_review
- escalation_level
- sla_due_at
- quarantined_reason
- auto_route_version
- reopen_count
- resolution_summary

#### `complaint_updates`

Stores threaded discussion and internal notes.

#### `complaint_audit_log`

Stores structured audit history.

#### `admin_users`

Stores admin invitation, role, and status data.

### 8.2 JSONB Usage

The schema uses JSONB heavily for:

- attachment arrays
- evidence types
- ML analysis payloads
- decision reasoning
- fairness flags
- audit state snapshots

This lets the system keep structured ML metadata without over-normalizing the schema.

### 8.3 Triggers And Indexes

The schema includes:

- `touch_updated_at` trigger function
- update triggers on `complaints` and `admin_users`
- indexes for complaint list ordering
- indexes for update thread ordering
- indexes for audit retrieval
- indexes for admin lookup

## 9. AI / ML System

### 9.1 ML Objectives

The ML layer is not a standalone product. It is embedded into operational workflow automation.

Its main objectives are:

- classify complaint category
- classify complaint priority
- map category to department
- detect duplicates
- estimate sentiment/urgency
- estimate abuse/spam risk
- assist complaint coaching and chatbot responses
- support human-review decisions
- support analytics and explainability

### 9.2 Core Classifier

The main model is a multitask transformer-based classifier.

It predicts:

- category label
- priority label

Training is implemented in `scripts/train_multitask.py`.

Architecture highlights:

- Hugging Face `AutoModel` backbone
- shared pooled transformer representation
- label head
- priority head
- optional metadata projections

### 9.3 Metadata Features

The model is augmented with handcrafted metadata from `complaint_ml.py`.

Feature groups include:

- text length and word count
- urgency keywords
- deadline keywords
- negative sentiment terms
- harassment terms
- academic and infrastructure terms
- repeated-issue terms
- financial-impact terms
- access-block terms
- safety-risk terms
- academic-impact terms
- duration terms
- punctuation and casing ratios
- attachment counts and attachment kind counts
- evidence type count
- attachment context availability
- anonymous flag
- hour-of-day and weekday cyclic features

This is a hybrid NLP design: learned semantic embeddings plus engineered operational features.

### 9.4 Training Strategy

The training pipeline uses:

- weighted sampling for class imbalance
- focal loss
- label smoothing
- cost-sensitive priority loss
- early stopping
- optional pseudo-feedback ingestion
- optional frontend/admin feedback ingestion
- optional reviewed-note weighting

This design shows clear intent to improve real-world class balance and severity prediction reliability.

### 9.5 Duplicate Detection

Duplicate detection uses semantic search:

- preferred: sentence-transformer embeddings
- approximate nearest neighbor indexing via `hnswlib`
- fallback: TF-IDF cosine similarity

This allows the system to identify similar prior complaints and feed:

- submission warnings
- recommendation logic
- chatbot context
- admin duplicate review

### 9.6 Explainability

The system generates explainability payloads containing:

- summary string
- confidence band
- rationale items
- text preview

Rationale signals come from metadata features like urgency terms, deadlines, harassment indicators, and evidence counts.

### 9.7 Risk And Automation Logic

The backend combines ML outputs with rules to compute:

- routing confidence
- risk score
- escalation level
- fairness flags
- human-review requirement
- quarantine decision
- SLA due date

This logic is central to operations and is implemented in `_build_automation_decision`.

### 9.8 Submission Guard

Before storing a complaint, the backend may block or warn on:

- strongly abusive language
- spam-like patterns
- near-certain duplicates

This reduces bad submissions while still allowing legitimate complaints to pass.

### 9.9 Fairness Monitoring

The project explicitly includes fairness evaluation and runtime fairness flags.

Monitoring dimensions include:

- anonymity groups
- language groups
- complaint categories
- repeat submitter group

This is notable because fairness has been considered as an operational signal, not just an offline paper metric.

### 9.10 Analytics And Forecasting

Admin analytics includes:

- urgent count
- abusive/spam count
- duplicate count
- auto-routed count
- human-review count
- escalated count
- quarantined count
- overdue SLA count
- average risk score
- department workload
- assignee workload
- emotion distribution
- explainability rationale summaries
- fairness alerts
- simple next-7-day trend forecast

The forecast is heuristic, based on recent complaint counts, not a full time-series ML model.

## 10. Chatbot / Assistant Layer

The project includes a backend chatbot endpoint used as a complaint assistant rather than a generic open-domain bot.

Capabilities:

- identify user intent
- explain how to submit complaints
- summarize complaint status
- warn about duplicates
- coach users on writing better complaints
- suggest category, priority, and likely department
- assist with evidence guidance

The chatbot uses:

- rule-based intent detection
- complaint-analysis context
- optional Gemini generation if configured
- fallback handcrafted response blending if generation is unavailable

This keeps the system usable even when external model APIs are unavailable.

## 11. AWS Architecture

### 11.1 Services Used

The repository expects the following AWS services:

- Cognito User Pool for auth
- S3 bucket for attachments
- S3 bucket or prefix for model artifacts
- RDS PostgreSQL for persistence
- EC2 for backend hosting
- CloudFront for frontend distribution
- optional ECR for image distribution

### 11.2 Authentication Architecture

- Frontend authenticates against Cognito using Amplify
- Frontend gets a Cognito token
- Token is sent to backend in `Authorization: Bearer ...`
- Backend validates token using Cognito JWKS

### 11.3 Attachment Architecture

- Backend generates presigned S3 URLs
- Frontend uploads directly to S3
- Backend stores only S3 object keys in PostgreSQL
- Download access is also provided via presigned URLs

This is scalable and reduces backend bandwidth load.

### 11.4 Model Artifact Strategy

Model files are intentionally excluded from Git.

They can be:

- kept locally in `backend/outputs`
- synced from S3 on startup

The startup sync is handled by `scripts/sync_models_from_s3.py` and `entrypoint.sh`.

### 11.5 Frontend Hosting Strategy

The intended production strategy is:

- build frontend into static assets
- host assets in S3
- serve via CloudFront
- point frontend API URL to backend origin

### 11.6 Backend Hosting Strategy

The backend is designed to run:

- locally via Uvicorn
- locally via Docker Compose
- on EC2 via Docker Compose
- optionally using ECR-hosted images

Nginx fronts the FastAPI container and exposes port 80.

## 12. Deployment Modes

The codebase supports multiple deployment combinations:

1. local frontend + local backend
2. local frontend + Dockerized local backend
3. CloudFront frontend + tunneled local backend
4. CloudFront frontend + EC2 backend

Mode-specific frontend env files are used instead of constantly rewriting one `.env`.

This is practical for demos, development, and staged deployment.

## 13. Security Model

### 13.1 Strengths

- Cognito-backed token auth
- backend-side token validation
- role-protected admin and super-admin routes
- attachment access restricted by ownership/admin status
- presigned URLs avoid exposing bucket-wide permissions
- invite-based admin access
- audit log for sensitive complaint actions

### 13.2 Risks / Gaps

- AWS infrastructure definitions are not codified in this repo
- secrets depend on external env management
- the backend is large and monolithic, which raises maintenance risk over time
- some fairness and risk logic is heuristic and may need governance review before production at scale

## 14. End-To-End Data Flow

### 14.1 Student Complaint Flow

1. User signs in through Cognito
2. Frontend gets session/token
3. User writes complaint and optionally uploads evidence
4. Frontend requests presigned upload URL(s)
5. Evidence is uploaded directly to S3
6. Complaint payload is sent to backend
7. Backend analyzes text and metadata
8. Complaint is stored in PostgreSQL
9. Audit log entry is created
10. Complaint appears in student dashboard and admin queue

### 14.2 Admin Review Flow

1. Admin signs in
2. Backend confirms admin access
3. Admin opens queue and filters items
4. Admin can request prediction, auto-apply, assign, update, or resolve
5. Backend writes state updates and audit records
6. Student sees updates in complaint detail and notifications

### 14.3 Offline Submission Flow

1. User fills complaint while offline
2. Frontend saves complaint and attachment blobs locally
3. Browser regains connectivity
4. Sync hook uploads attachments
5. Complaint is submitted to backend
6. Local pending copy is deleted

## 15. Operational Strengths

This project has several mature ideas already implemented:

- offline complaint queueing
- anonymous complaint support
- direct S3 upload architecture
- audit logging
- admin invite workflow
- AI-assisted routing
- fairness flagging
- explainability generation
- feedback-driven retraining hooks
- multimodal evidence metadata support

For a student grievance system, that is an unusually broad feature set.

## 16. Limitations And Current Architectural Notes

- The backend is concentrated in a very large single file, which makes long-term maintenance harder
- The `aws/` folder does not yet provide reproducible infrastructure definitions
- Production hardening details like secrets management, observability, WAF, TLS automation, and backup automation are not implemented in repo code
- The ML pipeline depends on local or externally stored model artifacts and datasets that are intentionally not committed
- Some automation and fairness logic is rule-based rather than fully learned or formally calibrated

## 17. Recommended Future Improvements

- split `backend/api/main.py` into routers, services, repositories, and ML modules
- add Infrastructure as Code for Cognito, S3, RDS, EC2, CloudFront, and IAM
- add structured logging and metrics
- add API tests and backend integration tests
- add database migrations tooling
- add stronger moderation/governance review for complaint-risk logic
- add attachment scanning and malware validation
- add admin SLA dashboards and escalation notifications
- move model lifecycle management into a more formal registry/versioning flow

## 18. Conclusion

CampusVoice is a full-stack complaint management system with a notably ambitious operational AI layer. It is not only a frontend form plus backend storage service; it is a workflow-oriented platform that combines complaint intake, role-based access control, conversation threads, analytics, cloud storage, and ML-assisted triage into one system.

From the codebase, the project can be accurately described as:

- a React/TypeScript complaint portal
- a FastAPI/PostgreSQL workflow backend
- an AWS-integrated cloud deployment candidate
- an NLP-assisted complaint routing and review platform

The strongest parts of the implementation are the end-to-end workflow thinking, the offline-first complaint submission behavior, the auditability of admin and automated actions, and the integration of ML into actual case handling rather than using AI as a detached demo feature.
