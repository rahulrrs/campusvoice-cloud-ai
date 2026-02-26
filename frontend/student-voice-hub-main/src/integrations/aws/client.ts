import { Amplify } from "aws-amplify";
import {
  fetchAuthSession,
  getCurrentUser,
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

const getToken = async () => {
  const session = await fetchAuthSession();
  return session.tokens?.accessToken?.toString() ?? null;
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
      await cognitoSignUp({
        username: email,
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
  status: string;
  created_at: string;
  updated_at?: string;
}

export interface CreateComplaintPayload {
  title: string;
  description: string;
  category: string;
  priority: string;
  user_id?: string;
  status?: string;
}

export const complaintsApi = {
  list: () => authFetch<ComplaintRecord[]>("/complaints"),
  create: (payload: CreateComplaintPayload) =>
    authFetch<ComplaintRecord>("/complaints", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
