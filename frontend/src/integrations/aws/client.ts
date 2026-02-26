import { Amplify } from "aws-amplify";
import {
  confirmSignUp as cognitoConfirmSignUp,
  fetchAuthSession,
  getCurrentUser,
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
const AWS_API_BASE_URL = import.meta.env.VITE_AWS_API_BASE_URL;

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

  async signOut() {
    await cognitoSignOut();
  },
};

const authFetch = async <T>(path: string, init?: RequestInit): Promise<T> => {
  if (!AWS_API_BASE_URL) {
    throw new Error("Missing VITE_AWS_API_BASE_URL environment variable");
  }

  const token = await getToken();
  if (!token) {
    throw new Error("Not authenticated");
  }

  const response = await fetch(`${AWS_API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
  });

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
  created_at: string;
  updated_at?: string;
}

export interface CreateComplaintPayload {
  title: string;
  description: string;
  category?: string;
  priority?: string;
  attachments?: string[];
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

export const complaintsApi = {
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
    authFetch<PredictionResult>(`/admin/complaints/${complaintId}/predict`, {
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
      }
    ),
};
