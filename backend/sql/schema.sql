CREATE TABLE IF NOT EXISTS complaints (
  id UUID PRIMARY KEY,
  user_id TEXT NOT NULL,
  title VARCHAR(200) NOT NULL,
  description TEXT NOT NULL,
  category VARCHAR(100) NOT NULL,
  priority VARCHAR(20) NOT NULL DEFAULT 'medium',
  department VARCHAR(120),
  status VARCHAR(30) NOT NULL DEFAULT 'submitted',
  attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
  evidence_types JSONB NOT NULL DEFAULT '[]'::jsonb,
  analysis JSONB NOT NULL DEFAULT '{}'::jsonb,
  source_language VARCHAR(40),
  assigned_to TEXT,
  admin_notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_complaints_user_created_at
  ON complaints (user_id, created_at DESC);

CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_touch_complaints_updated_at ON complaints;
CREATE TRIGGER trg_touch_complaints_updated_at
BEFORE UPDATE ON complaints
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'complaint_status') THEN
    CREATE TYPE complaint_status AS ENUM ('submitted', 'pending', 'in_progress', 'resolved', 'rejected');
  END IF;
END $$;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS is_anonymous BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS pending_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS in_progress_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS assigned_to TEXT,
  ADD COLUMN IF NOT EXISTS admin_notes TEXT;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS student_name TEXT,
  ADD COLUMN IF NOT EXISTS student_email TEXT,
  ADD COLUMN IF NOT EXISTS student_phone TEXT,
  ADD COLUMN IF NOT EXISTS student_department TEXT,
  ADD COLUMN IF NOT EXISTS student_registration_number TEXT;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS last_student_update_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_public_admin_update_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_user_viewed_updates_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_admin_viewed_updates_at TIMESTAMPTZ;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS resolution_summary TEXT,
  ADD COLUMN IF NOT EXISTS reopened_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS reopen_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS decision_state VARCHAR(30) NOT NULL DEFAULT 'submitted',
  ADD COLUMN IF NOT EXISTS risk_score NUMERIC NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS routing_confidence NUMERIC NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS decision_source TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS decision_reason JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS fairness_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS requires_human_review BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS escalation_level VARCHAR(20),
  ADD COLUMN IF NOT EXISTS sla_due_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS quarantined_reason TEXT,
  ADD COLUMN IF NOT EXISTS auto_route_version TEXT NOT NULL DEFAULT 'rules-v1';

UPDATE complaints
SET last_student_update_at = COALESCE(last_student_update_at, submitted_at, created_at)
WHERE last_student_update_at IS NULL;

CREATE TABLE IF NOT EXISTS complaint_updates (
  id UUID PRIMARY KEY,
  complaint_id UUID NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  author_role VARCHAR(20) NOT NULL,
  author_id TEXT,
  body TEXT NOT NULL,
  is_internal BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_complaint_updates_complaint_created_at
  ON complaint_updates (complaint_id, created_at ASC);

CREATE TABLE IF NOT EXISTS complaint_audit_log (
  id UUID PRIMARY KEY,
  complaint_id UUID NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  actor_type VARCHAR(20) NOT NULL,
  actor_id TEXT,
  event_type VARCHAR(80) NOT NULL,
  previous_state JSONB NOT NULL DEFAULT '{}'::jsonb,
  new_state JSONB NOT NULL DEFAULT '{}'::jsonb,
  reason JSONB NOT NULL DEFAULT '{}'::jsonb,
  model_version TEXT,
  rule_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_complaint_audit_log_complaint_created_at
  ON complaint_audit_log (complaint_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_complaints_decision_state_created_at
  ON complaints (decision_state, created_at DESC);

CREATE TABLE IF NOT EXISTS admin_users (
  id UUID PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  role VARCHAR(20) NOT NULL DEFAULT 'admin',
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  invite_token TEXT,
  invite_expires_at TIMESTAMPTZ,
  invited_by TEXT,
  accepted_by_user_id TEXT,
  accepted_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_admin_users_email
  ON admin_users (email);

CREATE INDEX IF NOT EXISTS idx_admin_users_status_role
  ON admin_users (status, role);

DROP TRIGGER IF EXISTS trg_touch_admin_users_updated_at ON admin_users;
CREATE TRIGGER trg_touch_admin_users_updated_at
BEFORE UPDATE ON admin_users
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

UPDATE complaints
SET submitted_at = COALESCE(submitted_at, created_at)
WHERE submitted_at IS NULL;

UPDATE complaints
SET pending_at = COALESCE(pending_at, created_at)
WHERE pending_at IS NULL
  AND status::text IN ('pending');

UPDATE complaints
SET in_progress_at = COALESCE(in_progress_at, updated_at, created_at)
WHERE in_progress_at IS NULL
  AND status::text IN ('in-progress', 'in_progress');

UPDATE complaints
SET resolved_at = COALESCE(resolved_at, updated_at, created_at)
WHERE resolved_at IS NULL
  AND status::text IN ('resolved', 'rejected');

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'complaints'
      AND column_name = 'status'
      AND udt_name <> 'complaint_status'
  ) THEN
    ALTER TABLE complaints
      ALTER COLUMN status DROP DEFAULT;

    ALTER TABLE complaints
      ALTER COLUMN status TYPE complaint_status
      USING (
        CASE
          WHEN status = 'in-progress' THEN 'in_progress'::complaint_status
          WHEN status = 'in_progress' THEN 'in_progress'::complaint_status
          WHEN status = 'resolved' THEN 'resolved'::complaint_status
          WHEN status = 'submitted' THEN 'submitted'::complaint_status
          WHEN status = 'rejected' THEN 'rejected'::complaint_status
          ELSE 'pending'::complaint_status
        END
      );
  END IF;
END $$;

ALTER TABLE complaints
  ALTER COLUMN status SET DEFAULT 'submitted';
