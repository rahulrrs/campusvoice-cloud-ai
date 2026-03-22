import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { awsAuth, type AuthSession, type AuthUser } from "@/integrations/aws/client";

interface AuthContextType {
  user: AuthUser | null;
  session: AuthSession | null;
  loading: boolean;
  googleAuthEnabled: boolean;
  signUp: (email: string, password: string, fullName: string) => Promise<{ error: Error | null }>;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signInWithGoogle: () => Promise<{ error: Error | null }>;
  confirmSignUp: (email: string, code: string) => Promise<{ error: Error | null }>;
  resendSignUpCode: (email: string) => Promise<{ error: Error | null }>;
  requestPasswordReset: (email: string) => Promise<{ error: Error | null }>;
  confirmPasswordReset: (email: string, code: string, newPassword: string) => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [session, setSession] = useState<AuthSession | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const initializeAuth = async () => {
      try {
        const sessionResult = await awsAuth.getSession();

        if (!isMounted) return;
        const currentSession = sessionResult.data.session;
        setSession(currentSession);
        setUser(currentSession?.user ?? null);
      } catch (error) {
        console.error("Failed to initialize auth session:", error);
        if (!isMounted) return;
        setSession(null);
        setUser(null);
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    void initializeAuth();

    return () => {
      isMounted = false;
    };
  }, []);

  const signUp = async (email: string, password: string, fullName: string) => {
    const { error } = await awsAuth.signUp(email, password, fullName);
    return { error };
  };

  const signIn = async (email: string, password: string) => {
    const firstAttempt = await awsAuth.signIn(email, password);

    if (firstAttempt.error) {
      return firstAttempt;
    }

    const sessionResult = await awsAuth.getSession();
    const currentSession = sessionResult.data.session;
    setSession(currentSession);
    setUser(currentSession?.user ?? null);

    return { error: null };
  };

  const signOut = async () => {
    await awsAuth.signOut();
    setSession(null);
    setUser(null);
  };

  const signInWithGoogle = async () => {
    return awsAuth.signInWithGoogle();
  };

  const confirmSignUp = async (email: string, code: string) => {
    return awsAuth.confirmSignUp(email, code);
  };

  const resendSignUpCode = async (email: string) => {
    return awsAuth.resendSignUpCode(email);
  };

  const requestPasswordReset = async (email: string) => {
    return awsAuth.requestPasswordReset(email);
  };

  const confirmPasswordReset = async (email: string, code: string, newPassword: string) => {
    return awsAuth.confirmPasswordReset(email, code, newPassword);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        loading,
        googleAuthEnabled: awsAuth.isGoogleAuthEnabled(),
        signUp,
        signIn,
        signInWithGoogle,
        confirmSignUp,
        resendSignUpCode,
        requestPasswordReset,
        confirmPasswordReset,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
