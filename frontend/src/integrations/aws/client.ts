import { Amplify } from "aws-amplify";
import {
  confirmResetPassword as cognitoConfirmResetPassword,
  confirmSignUp as cognitoConfirmSignUp,
  fetchAuthSession,
  getCurrentUser,
  resetPassword as cognitoResetPassword,
  resendSignUpCode as cognitoResendSignUpCode,
  signIn as cognitoSignIn,
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

const AWS_REGION = import.meta.env.VITE_AWS_REGION;
const AWS_USER_POOL_ID = import.meta.env.VITE_AWS_USER_POOL_ID;
const AWS_USER_POOL_CLIENT_ID = import.meta.env.VITE_AWS_USER_POOL_CLIENT_ID;
const RAW_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? import.meta.env.VITE_AWS_API_BASE_URL ?? "";
const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 30000);

const normalizeApiBaseUrl = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return "http://localhost:8000";
  }

  const withoutTrailingSlash = trimmed.replace(/\/+$/, "");
  if (/^https?:\/\/(localhost|127\.0\.0\.1)$/i.test(withoutTrailingSlash)) {
    return `${withoutTrailingSlash}:8000`;
  }

  if (withoutTrailingSlash.startsWith("/")) {
    return `${window.location.origin}${withoutTrailingSlash}`;
  }

  return withoutTrailingSlash;
};

const AWS_API_BASE_URL = normalizeApiBaseUrl(RAW_API_BASE_URL);

if (AWS_REGION && AWS_USER_POOL_ID && AWS_USER_POOL_CLIENT_ID) {
  Amplify.configure({
    Auth: {
      Cognito: {
        userPoolId: AWS_USER_POOL_ID,
        userPoolClientId: AWS_USER_POOL_CLIENT_ID,
        loginWith: {
          email: true,
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

const buildCognitoUsername = (email: string) => {
  const normalized = email.trim().toLowerCase();
  return `u_${normalized.replace(/[^a-z0-9]/g, "_")}`;
};

const getToken = async () => {
  const session = await fetchAuthSession();
  return (
    session.tokens?.idToken?.toString() ??
    session.tokens?.accessToken?.toString() ??
    null
  );
};

const getAuthUser = async (): Promise<AuthUser | null> => {
  try {
    const [current, session] = await Promise.all([getCurrentUser(), fetchAuthSession()]);
    const signInDetails = current.signInDetails;
    const tokenEmail =
      typeof session.tokens?.idToken?.payload?.email === "string"
        ? session.tokens.idToken.payload.email
        : undefined;
    const email =
      typeof signInDetails?.loginId === "string" ? signInDetails.loginId : tokenEmail;

    return {
      id: current.userId,
      email,
    };
  } catch {
    return null;
  }
};

export const awsAuth = {
  async getSession(): Promise<{ data: { session: AuthSession | null } }> {
    const [token, user] = await Promise.all([getToken(), getAuthUser()]);
    if (!token || !user) {
      return { data: { session: null } };
    }

    return {
      data: {
        session: {
          accessToken: token,
          user,
        },
      },
    };
  },

  async signUp(email: string, password: string, fullName: string) {
    try {
      const generatedUsername = buildCognitoUsername(email);
      await cognitoSignUp({
        username: generatedUsername,
        password,
        options: {
          userAttributes: {
            email,
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
        username: email,
        password,
      });
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
        username: email.trim().toLowerCase(),
      });
      return { error: null as Error | null };
    } catch (error) {
      return { error: normalizeError(error) };
    }
  },

  async confirmPasswordReset(email: string, code: string, newPassword: string) {
    try {
      await cognitoConfirmResetPassword({
        username: email.trim().toLowerCase(),
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
  },
};

const authFetch = async <T>(
  path: string,
  init?: RequestInit,
  timeoutMs = API_TIMEOUT_MS
): Promise<T> => {
  if (!AWS_API_BASE_URL) {
    throw new Error("Missing VITE_API_BASE_URL or VITE_AWS_API_BASE_URL environment variable");
  }

  const token = await getToken();
  if (!token) {
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
        Authorization: `Bearer ${token}`,
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

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
};

export interface ComplaintRecord {
  id: string;
  user_id: string;
  title: string;
  description: string;
  category: string;
  priority: string;
  department?: string | null;
  status: string;
  attachments?: string[];
  evidence_types?: string[];
  analysis?: ComplaintAnalysisBundle;
  source_language?: string | null;
  created_at: string;
  updated_at?: string;
}

export interface CreateComplaintPayload {
  title: string;
  description: string;
  category?: string;
  priority?: string;
  attachments?: string[];
  evidence_types?: string[];
  analysis?: ComplaintAnalysisBundle;
  source_language?: string;
  user_id?: string;
  status?: string;
}

export interface PresignedUploadRequest {
  fileName: string;
  contentType: string;
}

export interface PresignedUploadResponse {
  uploadUrl: string;
  key: string;
  expiresIn: number;
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
  source_language: string;
}

export interface AnalyticsResponse {
  summary: {
    complaints_analyzed: number;
    urgent_count: number;
    abusive_or_spam_count: number;
    duplicate_count: number;
  };
  emotion_distribution: Record<string, number>;
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
  analyzeComplaint: (payload: { title: string; description: string }) =>
    authFetch<ComplaintAnalysisBundle>("/complaints/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  list: () => authFetch<ComplaintRecord[]>("/complaints"),
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
    }),
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
  listAllForAdmin: () => authFetch<ComplaintRecord[]>("/admin/complaints"),
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
  chatbotRespond: (message: string, history: ChatTurnPayload[] = []) =>
    authFetch<ChatbotResponse>("/chatbot/respond", {
      method: "POST",
      body: JSON.stringify({ message, history }),
    }),
};
