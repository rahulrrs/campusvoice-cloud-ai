# AWS Complaint API

This backend exposes complaint APIs via API Gateway + Lambda and stores data in Aurora PostgreSQL (Data API).

## Endpoints

- `GET /health`
- `GET /complaints` (JWT protected)
- `POST /complaints` (JWT protected)
- `POST /uploads/presigned-url` (JWT protected)

## Deploy (SAM)

1. Enable Data API on your Aurora PostgreSQL cluster.
2. Run SQL in `sql/schema.sql`.
3. Build and deploy:

```bash
sam build -t template.yaml
sam deploy --guided
```

Required deploy parameters:

- `UserPoolId`
- `UserPoolClientId`
- `RdsClusterArn`
- `RdsSecretArn`
- `RdsDatabaseName`
- `CorsAllowOrigin`

## Frontend env

Use API output URL as `VITE_AWS_API_BASE_URL`.
