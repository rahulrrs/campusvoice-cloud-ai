import { Amplify } from "aws-amplify";
import {
  confirmResetPassword as cognitoConfirmResetPassword,
  confirmSignUp as cognitoConfirmSignUp,
  fetchAuthSession,
  getCurrentUser,
  resetPassword as cognitoResetPassword,
  resendSignUpCode as cognitoResendSignUpCode,
  signIn as cognitoSignIn,
  signInWithRedirect,
  signOut as cognitoSignOut,
  signUp as cognitoSignUp,
} from "aws-amplify/auth";

export interface AuthUser {
  id: string;
  email?: string;
}

export interface AuthSession {
  accessToken: string;
  user: AuthUser;
}

type CachedSessionState = {
  value: AuthSession | null;
  expiresAt: number;
  pending?: Promise<AuthSession | null>;
};

const AWS_REGION = import.meta.env.VITE_AWS_REGION;
const AWS_USER_POOL_ID = import.meta.env.VITE_AWS_USER_POOL_ID;
const AWS_USER_POOL_CLIENT_ID = import.meta.env.VITE_AWS_USER_POOL_CLIENT_ID;
const COGNITO_OAUTH_DOMAIN = import.meta.env.VITE_COGNITO_OAUTH_DOMAIN ?? "";
const COGNITO_REDIRECT_SIGN_IN = import.meta.env.VITE_COGNITO_REDIRECT_SIGN_IN ?? window.location.origin;
const COGNITO_REDIRECT_SIGN_OUT = import.meta.env.VITE_COGNITO_REDIRECT_SIGN_OUT ?? window.location.origin;
const GOOGLE_OAUTH_ENABLED = String(import.meta.env.VITE_GOOGLE_OAUTH_ENABLED ?? "").toLowerCase() === "true";
const RAW_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? import.meta.env.VITE_AWS_API_BASE_URL ?? "";
const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 30000);
const AUTH_CACHE_TTL_MS = 10_000;
const cachedSession: CachedSessionState = {
  value: null,
  expiresAt: 0,
};

const normalizeApiBaseUrl = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return "http://localhost:8000";
  }

  const withoutTrailingSlash = trimmed.replace(/\/+$/, "");
  if (withoutTrailingSlash.startsWith("/")) {
    return `${window.location.origin}${withoutTrailingSlash}`;
  }

  return withoutTrailingSlash;
};

const AWS_API_BASE_URL = normalizeApiBaseUrl(RAW_API_BASE_URL);
const IS_NGROK_TUNNEL = /https:\/\/.*\.ngrok-(free\.app|free\.dev)$/i.test(AWS_API_BASE_URL);

const oauthConfigured =
  GOOGLE_OAUTH_ENABLED &&
  !!COGNITO_OAUTH_DOMAIN.trim() &&
  !!COGNITO_REDIRECT_SIGN_IN.trim() &&
  !!COGNITO_REDIRECT_SIGN_OUT.trim();

if (AWS_REGION && AWS_USER_POOL_ID && AWS_USER_POOL_CLIENT_ID) {
  Amplify.configure({
    Auth: {
      Cognito: {
        userPoolId: AWS_USER_POOL_ID,
        userPoolClientId: AWS_USER_POOL_CLIENT_ID,
        loginWith: {
          email: true,
          ...(oauthConfigured
            ? {
                oauth: {
                  domain: COGNITO_OAUTH_DOMAIN.trim(),
                  scopes: ["email", "openid", "profile"],
                  redirectSignIn: [COGNITO_REDIRECT_SIGN_IN.trim()],
                  redirectSignOut: [COGNITO_REDIRECT_SIGN_OUT.trim()],
                  responseType: "code" as const,
                  providers: ["Google"],
                },
              }
            : {}),
        },
      },
    },
  });
}

const normalizeError = (error: unknown) => {
  if (error instanceof Error) {
    return error;
  }
  if (typeof error === "string") {
    return new Error(error);
  }
  return new Error("Unknown authentication error");
};

const normalizeEmailAddress = (email: string) => email.trim().toLowerCase();

const buildCognitoUsername = (email: string) => {
  const normalized = normalizeEmailAddress(email);
  return `u_${normalized.replace(/[^a-z0-9]/g, "_")}`;
};

const cacheResolvedSession = (session: AuthSession | null) => {
  cachedSession.value = session;
  cachedSession.expiresAt = Date.now() + AUTH_CACHE_TTL_MS;
  cachedSession.pending = undefined;
  return session;
};

const clearCachedSession = () => {
  cachedSession.value = null;
  cachedSession.expiresAt = 0;
  cachedSession.pending = undefined;
};

const getFreshSession = async (): Promise<AuthSession | null> => {
  try {
    const [current, session] = await Promise.all([getCurrentUser(), fetchAuthSession()]);
    const signInDetails = current.signInDetails;
    const tokenEmail =
      typeof session.tokens?.idToken?.payload?.email === "string"
        ? session.tokens.idToken.payload.email
        : undefined;
    const email =
      typeof signInDetails?.loginId === "string" ? signInDetails.loginId : tokenEmail;

    const token =
      session.tokens?.idToken?.toString() ??
      session.tokens?.accessToken?.toString() ??
      null;

    if (!token) {
      return null;
    }

    return {
      accessToken: token,
      user: {
        id: current.userId,
        email,
      },
    };
  } catch {
    return null;
  }
};

const getCachedSession = async (forceRefresh = false): Promise<AuthSession | null> => {
  if (!forceRefresh && cachedSession.value && cachedSession.expiresAt > Date.now()) {
    return cachedSession.value;
  }
  if (!forceRefresh && cachedSession.pending) {
    return cachedSession.pending;
  }

  const pending = getFreshSession()
    .then((session) => cacheResolvedSession(session))
    .catch((error) => {
      clearCachedSession();
      throw error;
    });

  cachedSession.pending = pending;
  return pending;
};

export const awsAuth = {
  async getSession(): Promise<{ data: { session: AuthSession | null } }> {
    const session = await getCachedSession();
    return {
      data: {
        session,
      },
    };
  },

  async signUp(email: string, password: string, fullName: string) {
    try {
      const normalizedEmail = normalizeEmailAddress(email);
      const generatedUsername = buildCognitoUsername(normalizedEmail);
      await cognitoSignUp({
        username: generatedUsername,
        password,
        options: {
          userAttributes: {
            email: normalizedEmail,
            name: fullName,
          },
        },
      });
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  async signIn(email: string, password: string) {
    try {
      await cognitoSignIn({
        username: normalizeEmailAddress(email),
        password,
      });
      clearCachedSession();
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  async confirmSignUp(email: string, code: string) {
    try {
      await cognitoConfirmSignUp({
        username: buildCognitoUsername(email),
        confirmationCode: code,
      });
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  async resendSignUpCode(email: string) {
    try {
      await cognitoResendSignUpCode({
        username: buildCognitoUsername(email),
      });
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  async requestPasswordReset(email: string) {
    try {
      await cognitoResetPassword({
        username: normalizeEmailAddress(email),
      });
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  async confirmPasswordReset(email: string, code: string, newPassword: string) {
    try {
      await cognitoConfirmResetPassword({
        username: normalizeEmailAddress(email),
        confirmationCode: code.trim(),
        newPassword,
      });
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  async signOut() {
    await cognitoSignOut();
    clearCachedSession();
  },

  async signInWithGoogle() {
    if (!oauthConfigured) {
      return { error: new Error("Google sign-in is not configured yet.") };
    }
    try {
      await signInWithRedirect({ provider: "Google" });
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  isGoogleAuthEnabled() {
    return oauthConfigured;
  },
};

const authorizedRequest = async (
  path: string,
  init?: RequestInit,
  timeoutMs = API_TIMEOUT_MS
): Promise<Response> => {
  if (!AWS_API_BASE_URL) {
    throw new Error("Missing VITE_API_BASE_URL or VITE_AWS_API_BASE_URL environment variable");
  }

  const session = await getCachedSession();
  if (!session?.accessToken) {
    throw new Error("Not authenticated");
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${AWS_API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(IS_NGROK_TUNNEL ? { "ngrok-skip-browser-warning": "true" } : {}),
        Authorization: `Bearer ${session.accessToken}`,
        ...(init?.headers ?? {}),
      },
    });
  } catch (error) {
    if ((error as DOMException)?.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.message === "string") {
        message = body.message;
      } else if (typeof body?.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Ignore JSON parse failures and keep status message.
    }
    throw new Error(message);
  }

  return response;
};

const authFetch = async <T>(
  path: string,
  init?: RequestInit,
  timeoutMs = API_TIMEOUT_MS
): Promise<T> => {
  const response = await authorizedRequest(path, init, timeoutMs);
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
};

const authFetchText = async (
  path: string,
  init?: RequestInit,
  timeoutMs = API_TIMEOUT_MS
): Promise<string> => {
  const response = await authorizedRequest(path, init, timeoutMs);
  return await response.text();
};

export interface ComplaintRecord {
  id: string;
  user_id: string;
  title: string;
  description: string;
  category: string;
  priority: string;
  department?: string | null;
  assigned_to?: string | null;
  admin_notes?: string | null;
  status: string;
  is_anonymous?: boolean;
  attachments?: string[];
  evidence_types?: string[];
  student_name?: string | null;
  student_email?: string | null;
  student_phone?: string | null;
  student_department?: string | null;
  student_registration_number?: string | null;
  analysis?: ComplaintAnalysisBundle;
  source_language?: string | null;
  decision_state?: string;
  risk_score?: number;
  routing_confidence?: number;
  decision_source?: string | null;
  decision_reason?: Record<string, unknown>;
  fairness_flags?: string[];
  requires_human_review?: boolean;
  escalation_level?: string | null;
  sla_due_at?: string | null;
  quarantined_reason?: string | null;
  auto_route_version?: string | null;
  submitted_at?: string | null;
  pending_at?: string | null;
  in_progress_at?: string | null;
  resolved_at?: string | null;
  last_student_update_at?: string | null;
  last_public_admin_update_at?: string | null;
  last_user_viewed_updates_at?: string | null;
  last_admin_viewed_updates_at?: string | null;
  has_unread_updates_for_user?: boolean;
  has_unread_updates_for_admin?: boolean;
  resolution_summary?: string | null;
  reopened_at?: string | null;
  reopen_count?: number;
  created_at: string;
  updated_at?: string;
}

export interface ComplaintUpdateRecord {
  id: string;
  complaint_id: string;
  author_role: string;
  author_id?: string | null;
  body: string;
  is_internal: boolean;
  created_at: string;
}

export interface CreateComplaintPayload {
  title: string;
  description: string;
  category?: string;
  priority?: string;
  attachments?: string[];
  evidence_types?: string[];
  attachment_contexts?: Array<{
    key?: string | null;
    mime_type?: string | null;
    language?: string | null;
    ocr_text?: string | null;
    transcript_text?: string | null;
    image_summary?: string | null;
  }>;
  analysis?: ComplaintAnalysisBundle;
  source_language?: string;
  is_anonymous?: boolean;
  user_id?: string;
  status?: string;
  student_name?: string;
  student_email?: string;
  student_phone?: string;
  student_department?: string;
  student_registration_number?: string;
}

export interface ComplaintListFilters {
  status?: string;
  category?: string;
}

export interface FAQItem {
  id: string;
  question: string;
  answer: string;
}

export interface PresignedUploadRequest {
  fileName: string;
  contentType: string;
  fileSize?: number;
}

export interface PresignedUploadResponse {
  uploadUrl: string;
  key: string;
  expiresIn: number;
  warnings?: string[];
}

export interface PresignedDownloadRequest {
  key: string;
}

export interface PresignedDownloadResponse {
  downloadUrl: string;
  key: string;
  expiresIn: number;
}

export interface AdminComplaintUpdatePayload {
  category?: string;
  priority?: string;
  department?: string;
  status?: string;
  decision_state?: string;
  assigned_to?: string | null;
  admin_notes?: string | null;
  resolution_summary?: string | null;
}

export interface AdminComplaintFilters {
  status?: string;
  department?: string;
  assigned_to?: string;
  review_state?: string;
  q?: string;
}

export interface ComplaintUpdatePayload {
  body: string;
}

export interface AdminComplaintUpdateMessagePayload {
  body: string;
  is_internal?: boolean;
}

export interface ComplaintReopenPayload {
  reason: string;
}

export interface PredictionResult {
  label: string;
  label_confidence: number;
  priority: string;
  priority_confidence: number;
  department: string;
}

export interface ComplaintAnalysisBundle {
  classification: PredictionResult;
  sentiment: {
    sentiment_score: number;
    sentiment_label: string;
    emotion: string;
    emotion_intensity: number;
    urgency_score: number;
  };
  abuse: {
    toxicity_score: number;
    spam_score: number;
    flags: string[];
    user_behavior?: {
      risk_score: number;
      recent_submissions_30d: number;
    };
  };
  duplicate_detection: {
    is_duplicate: boolean;
    score: number;
    method: string;
    matches: Array<{
      id: string;
      title: string;
      category?: string;
      status?: string;
      score: number;
    }>;
  };
  multimodal_evidence?: {
    attachment_count: number;
    evidence_types: string[];
    available_modalities: string[];
    summary: string;
    extracted_text: string;
  };
  explainability?: {
    summary: string;
    confidence_band: string;
    rationale_items: Array<{
      target: string;
      feature: string;
      value: number;
      reason: string;
    }>;
    text_preview: string;
  };
  recommendations: Array<{
    complaint_id: string;
    title: string;
    score: number;
    suggested_department: string;
    suggested_action: string;
  }>;
  knowledge_graph: {
    department: string;
    issue_type: string;
    priority: string;
    entities: string[];
  };
  attachment_checks?: {
    attachments: Array<{
      file_name: string;
      extension: string;
      kind: string;
      has_extracted_text?: boolean;
      warnings: string[];
    }>;
    image_count: number;
    audio_count?: number;
    document_count?: number;
    warnings: string[];
  };
  submission_guard?: {
    allow_submission: boolean;
    warnings: string[];
    reasons: string[];
  };
  source_language: string;
}

export interface AnalyticsResponse {
  summary: {
    complaints_analyzed: number;
    urgent_count: number;
    abusive_or_spam_count: number;
    duplicate_count: number;
    auto_routed_count: number;
    human_review_count: number;
    escalated_count: number;
    quarantined_count: number;
    overdue_sla_count: number;
    average_risk_score: number;
  };
  workload: {
    departments: Array<{
      department: string;
      total: number;
      active: number;
      urgent: number;
      unassigned: number;
    }>;
    assignees: Array<{
      assignee: string;
      total: number;
      active: number;
      resolved: number;
    }>;
  };
  emotion_distribution: Record<string, number>;
  explainability_summary: {
    top_rationales: Array<{
      feature: string;
      count: number;
    }>;
  };
  fairness_summary: {
    alert_count: number;
    top_flags: Array<{
      flag: string;
      count: number;
    }>;
    group_breakdown: Record<string, Array<{
      group: string;
      count: number;
      human_review_count: number;
    }>>;
  };
  trend_forecast: {
    overall: {
      recent_average: number;
      predicted_next_7_days: number;
      trend: string;
    };
    top_categories: Array<{
      category: string;
      recent_average: number;
      predicted_next_7_days: number;
      trend: string;
    }>;
  };
}

export interface ComplaintAuditLogRecord {
  id: string;
  complaint_id: string;
  actor_type: string;
  actor_id?: string | null;
  event_type: string;
  previous_state: Record<string, unknown>;
  new_state: Record<string, unknown>;
  reason: Record<string, unknown>;
  model_version?: string | null;
  rule_version?: string | null;
  created_at: string;
}

export interface NotificationItem {
  complaint_id: string;
  title: string;
  category: string;
  status: string;
  timestamp?: string | null;
  group_key: string;
  group_label: string;
  department?: string | null;
  priority?: string | null;
  preview?: string | null;
}

export interface NotificationGroup {
  key: string;
  label: string;
  count: number;
  items: NotificationItem[];
}

export interface NotificationsResponse {
  total: number;
  groups: NotificationGroup[];
}

export interface NotificationMarkReadPayload {
  complaint_id?: string;
  mark_all?: boolean;
}

export interface AdminAccessRecord {
  id: string;
  email: string;
  role: "admin" | "super_admin";
  status: "pending" | "active" | "suspended" | "revoked";
  invite_token?: string | null;
  invite_expires_at?: string | null;
  invited_by?: string | null;
  accepted_by_user_id?: string | null;
  accepted_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface AccessProfile {
  user_id: string;
  email?: string | null;
  role: "user" | "admin" | "super_admin";
  is_admin: boolean;
  is_super_admin: boolean;
  pending_invites: AdminAccessRecord[];
}

export interface AdminInvitePayload {
  email: string;
  role: "admin" | "super_admin";
}

export interface AdminAccessUpdatePayload {
  role?: "admin" | "super_admin";
  status?: "pending" | "active" | "suspended" | "revoked";
}

export interface ChatbotResponse {
  reply: string;
  intent: string;
  intent_confidence?: number;
  status_summary?: {
    total: number;
    pending: number;
    in_progress: number;
    resolved: number;
    rejected: number;
  };
  context_snippets?: Array<{
    id: string;
    title: string;
    category?: string;
    status?: string;
    department?: string;
    score: number;
  }>;
  duplicate_detection?: ComplaintAnalysisBundle["duplicate_detection"];
  analysis_preview?: ComplaintAnalysisBundle;
  follow_up_questions?: string[];
  suggested_title?: string;
}

export interface ChatTurnPayload {
  role: "user" | "assistant";
  text: string;
}

export const complaintsApi = {
  predictText: (text: string) =>
    authFetch<PredictionResult>("/predict", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  analyzeComplaint: (payload: {
    title: string;
    description: string;
    attachments?: string[];
    evidence_types?: string[];
    source_language?: string;
    is_anonymous?: boolean;
    submitted_at?: string;
    attachment_contexts?: CreateComplaintPayload["attachment_contexts"];
  }) =>
    authFetch<ComplaintAnalysisBundle>("/complaints/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  list: (filters?: ComplaintListFilters) => {
    const params = new URLSearchParams();
    if (filters?.status) {
      params.set("status", filters.status);
    }
    if (filters?.category && filters.category.toLowerCase() !== "all") {
      params.set("category", filters.category);
    }
    const query = params.toString();
    return authFetch<ComplaintRecord[]>(`/complaints${query ? `?${query}` : ""}`);
  },
  getComplaint: (complaintId: string) => authFetch<ComplaintRecord>(`/complaints/${complaintId}`),
  deleteComplaint: (complaintId: string) =>
    authFetch<{ ok: boolean; id: string }>(`/complaints/${complaintId}`, {
      method: "DELETE",
    }),
  listComplaintUpdates: (complaintId: string) =>
    authFetch<ComplaintUpdateRecord[]>(`/complaints/${complaintId}/updates`),
  createComplaintUpdate: (complaintId: string, payload: ComplaintUpdatePayload) =>
    authFetch<ComplaintUpdateRecord>(`/complaints/${complaintId}/updates`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  reopenComplaint: (complaintId: string, payload: ComplaintReopenPayload) =>
    authFetch<ComplaintRecord>(`/complaints/${complaintId}/reopen`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  create: (payload: CreateComplaintPayload) =>
    authFetch<ComplaintRecord>("/complaints", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  createUploadUrl: (payload: PresignedUploadRequest) =>
    authFetch<PresignedUploadResponse>("/uploads/presigned-url", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  createDownloadUrl: (payload: PresignedDownloadRequest) =>
    authFetch<PresignedDownloadResponse>("/uploads/presigned-download", {
      method: "POST",
      body: JSON.stringify(payload),
    }, 12_000),
  uploadToS3: async (uploadUrl: string, file: Blob, contentType: string) => {
    const response = await fetch(uploadUrl, {
      method: "PUT",
      headers: {
        "Content-Type": contentType,
      },
      body: file,
    });
    if (!response.ok) {
      throw new Error(`Attachment upload failed with ${response.status}`);
    }
  },
  listAllForAdmin: (filters?: AdminComplaintFilters) => {
    const params = new URLSearchParams();
    if (filters?.status && filters.status !== "all") params.set("status", filters.status);
    if (filters?.department && filters.department !== "all") params.set("department", filters.department);
    if (filters?.assigned_to && filters.assigned_to !== "all") params.set("assigned_to", filters.assigned_to);
    if (filters?.review_state && filters.review_state !== "all") params.set("review_state", filters.review_state);
    if (filters?.q?.trim()) params.set("q", filters.q.trim());
    const query = params.toString();
    return authFetch<ComplaintRecord[]>(`/admin/complaints${query ? `?${query}` : ""}`);
  },
  listComplaintUpdatesForAdmin: (complaintId: string) =>
    authFetch<ComplaintUpdateRecord[]>(`/admin/complaints/${complaintId}/updates`),
  createComplaintUpdateForAdmin: (
    complaintId: string,
    payload: AdminComplaintUpdateMessagePayload
  ) =>
    authFetch<ComplaintUpdateRecord>(`/admin/complaints/${complaintId}/updates`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  predictForComplaint: (complaintId: string) =>
    authFetch<ComplaintAnalysisBundle>(`/admin/complaints/${complaintId}/predict`, {
      method: "POST",
    }),
  autoApplyPrediction: (complaintId: string) =>
    authFetch<{ prediction: PredictionResult; complaint: ComplaintRecord }>(
      `/admin/complaints/${complaintId}/auto-apply`,
      {
        method: "POST",
      }
    ),
  updateComplaintByAdmin: (complaintId: string, payload: AdminComplaintUpdatePayload) =>
    authFetch<ComplaintRecord>(`/admin/complaints/${complaintId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  approveComplaint: (complaintId: string) =>
    authFetch<ComplaintRecord>(`/admin/complaints/${complaintId}/approve`, {
      method: "POST",
    }),
  autoClassifyAll: (onlyPending = true) =>
    authFetch<{ updatedCount: number; items: Array<{ prediction: PredictionResult; complaint: ComplaintRecord }> }>(
      "/admin/complaints/auto-classify",
      {
        method: "POST",
        body: JSON.stringify({ only_pending: onlyPending }),
      },
      120_000
    ),
  triggerModelRetrain: () =>
    authFetch<{ ok: boolean; status: string }>("/admin/retrain", {
      method: "POST",
    }),
  getAdminAnalytics: () => authFetch<AnalyticsResponse>("/admin/analytics"),
  getAdminAuditLog: (limit = 50) =>
    authFetch<ComplaintAuditLogRecord[]>(`/admin/audit-log?limit=${encodeURIComponent(String(limit))}`),
  getComplaintAuditLog: (complaintId: string) =>
    authFetch<ComplaintAuditLogRecord[]>(`/admin/complaints/${complaintId}/audit-log`),
  getNotifications: () => authFetch<NotificationsResponse>("/notifications"),
  markNotificationsRead: (payload: NotificationMarkReadPayload) =>
    authFetch<{ ok: boolean }>("/notifications/mark-read", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  downloadAdminComplaintsReport: (
    format: "csv" | "json" = "csv",
    filters?: AdminComplaintFilters
  ) => {
    const params = new URLSearchParams();
    params.set("format", format);
    if (filters?.status && filters.status !== "all") params.set("status", filters.status);
    if (filters?.department && filters.department !== "all") params.set("department", filters.department);
    if (filters?.assigned_to && filters.assigned_to !== "all") params.set("assigned_to", filters.assigned_to);
    if (filters?.review_state && filters.review_state !== "all") params.set("review_state", filters.review_state);
    if (filters?.q?.trim()) params.set("q", filters.q.trim());
    return authFetchText(`/admin/reports/complaints?${params.toString()}`);
  },
  getFaq: () => authFetch<FAQItem[]>("/faq"),
  getAccessProfile: () => authFetch<AccessProfile>("/me/access"),
  listAdminUsers: () => authFetch<AdminAccessRecord[]>("/super-admin/admin-users"),
  inviteAdminUser: (payload: AdminInvitePayload) =>
    authFetch<AdminAccessRecord>("/super-admin/admin-users/invite", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateAdminUserAccess: (accessId: string, payload: AdminAccessUpdatePayload) =>
    authFetch<AdminAccessRecord>(`/super-admin/admin-users/${accessId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  acceptAdminInvite: (token: string) =>
    authFetch<AdminAccessRecord>("/admin-access/accept-invite", {
      method: "POST",
      body: JSON.stringify({ token }),
    }),
  chatbotRespond: (message: string, history: ChatTurnPayload[] = []) =>
    authFetch<ChatbotResponse>("/chatbot/respond", {
      method: "POST",
      body: JSON.stringify({ message, history }),
    }),
};
