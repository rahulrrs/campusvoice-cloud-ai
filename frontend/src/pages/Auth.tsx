import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  AlertCircle,
  Circle,
  CheckCircle2,
  Eye,
  EyeOff,
  Lock,
  Mail,
  MessageSquare,
  User,
} from "lucide-react";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";

const emailSchema = z.string().email("Please enter a valid email address");
const nameSchema = z.string().min(2, "Name must be at least 2 characters");

const passwordRequirements = [
  {
    label: "At least 8 characters",
    test: (value: string) => value.length >= 8,
  },
  {
    label: "At least 1 letter",
    test: (value: string) => /[A-Za-z]/.test(value),
  },
  {
    label: "At least 1 number or special character",
    test: (value: string) => /[\d\W_]/.test(value),
  },
] as const;

const validatePassword = (value: string) => {
  const failed = passwordRequirements.filter((rule) => !rule.test(value));
  if (failed.length > 0) {
    return failed[0].label;
  }
  return null;
};

const PasswordChecklist = ({ value }: { value: string }) => (
  <div className="rounded-xl border bg-muted/20 p-3">
    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Password rules</p>
    <div className="mt-2 space-y-2">
      {passwordRequirements.map((rule) => {
        const passed = rule.test(value);
        return (
          <div key={rule.label} className="flex items-center gap-2 text-sm">
            {passed ? (
              <CheckCircle2 className="h-4 w-4 text-success" />
            ) : (
              <Circle className="h-4 w-4 text-muted-foreground/50" />
            )}
            <span className={passed ? "text-foreground" : "text-muted-foreground"}>{rule.label}</span>
          </div>
        );
      })}
    </div>
  </div>
);

const GoogleIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true" className="h-4 w-4">
    <path fill="#EA4335" d="M12 10.2v3.9h5.5c-.2 1.3-1.6 3.9-5.5 3.9-3.3 0-6-2.7-6-6s2.7-6 6-6c1.9 0 3.2.8 4 1.5l2.7-2.6C17 3.3 14.8 2.4 12 2.4 6.8 2.4 2.6 6.6 2.6 11.8S6.8 21.2 12 21.2c6.9 0 9.2-4.8 9.2-7.3 0-.5 0-.8-.1-1.1H12Z" />
    <path fill="#34A853" d="M3.7 7.5l3.2 2.3C7.7 7.9 9.6 6.5 12 6.5c1.9 0 3.2.8 4 1.5l2.7-2.6C17 3.3 14.8 2.4 12 2.4c-3.7 0-6.8 2.1-8.3 5.1Z" />
    <path fill="#FBBC05" d="M12 21.2c2.7 0 5-.9 6.7-2.5l-3.1-2.5c-.8.6-1.9 1-3.6 1-3.8 0-5.1-2.5-5.4-3.7l-3.2 2.5c1.5 3 4.6 5.2 8.6 5.2Z" />
    <path fill="#4285F4" d="M21.2 13.9c0-.5 0-.8-.1-1.1H12v3.9h5.5c-.3 1.3-1.1 2.2-1.9 2.8l3.1 2.5c1.8-1.7 2.8-4.1 2.8-7.1Z" />
  </svg>
);

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [resetStep, setResetStep] = useState<"request" | "confirm">("request");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [fullName, setFullName] = useState("");
  const [verificationCode, setVerificationCode] = useState("");
  const [resetCode, setResetCode] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [errors, setErrors] = useState<{ email?: string; password?: string; fullName?: string }>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    signIn,
    signInWithGoogle,
    googleAuthEnabled,
    signUp,
    confirmSignUp,
    resendSignUpCode,
    requestPasswordReset,
    confirmPasswordReset,
    user,
    loading,
  } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    if (!loading && user) {
      navigate("/dashboard");
    }
  }, [user, loading, navigate]);

  const activePasswordValue = useMemo(() => {
    if (isResetting && resetStep === "confirm") {
      return newPassword;
    }
    return password;
  }, [isResetting, resetStep, newPassword, password]);

  const showPasswordRules = (!isLogin && !isVerifying && !isResetting) || (isResetting && resetStep === "confirm");

  const validateForm = () => {
    const nextErrors: { email?: string; password?: string; fullName?: string } = {};

    const emailResult = emailSchema.safeParse(email);
    if (!emailResult.success) {
      nextErrors.email = emailResult.error.errors[0].message;
    }

    if (isResetting) {
      if (resetStep === "confirm") {
        if (!resetCode.trim()) {
          nextErrors.password = "Reset code is required";
        } else {
          const passwordError = validatePassword(newPassword);
          if (passwordError) {
            nextErrors.password = passwordError;
          }
        }
      }
    } else if (!isVerifying) {
      const passwordError = validatePassword(password);
      if (passwordError) {
        nextErrors.password = passwordError;
      }
    } else if (!verificationCode.trim()) {
      nextErrors.password = "Verification code is required";
    }

    if (!isLogin && !isVerifying && !isResetting) {
      const nameResult = nameSchema.safeParse(fullName);
      if (!nameResult.success) {
        nextErrors.fullName = nameResult.error.errors[0].message;
      }
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleGoogleSignIn = async () => {
    const { error } = await signInWithGoogle();
    if (error) {
      toast({
        title: "Google sign-in unavailable",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!validateForm()) return;
    setIsSubmitting(true);

    try {
      if (isResetting) {
        if (resetStep === "request") {
          const { error } = await requestPasswordReset(email);
          if (error) {
            toast({
              title: "Reset failed",
              description: error.message,
              variant: "destructive",
            });
          } else {
            setResetStep("confirm");
            toast({
              title: "Reset code sent",
              description: "Check your email for the password reset code.",
            });
          }
        } else {
          const { error } = await confirmPasswordReset(email, resetCode, newPassword);
          if (error) {
            toast({
              title: "Could not reset password",
              description: error.message,
              variant: "destructive",
            });
          } else {
            toast({
              title: "Password reset successful",
              description: "You can now sign in with your new password.",
            });
            setIsResetting(false);
            setResetStep("request");
            setResetCode("");
            setNewPassword("");
            setIsLogin(true);
          }
        }
      } else if (isVerifying) {
        const { error } = await confirmSignUp(email, verificationCode.trim());
        if (error) {
          toast({
            title: "Verification failed",
            description: error.message,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Email verified",
            description: "Your account is verified. Please sign in.",
          });
          setIsVerifying(false);
          setIsLogin(true);
          setVerificationCode("");
        }
      } else if (isLogin) {
        const { error } = await signIn(email, password);
        if (error) {
          if (
            error.message.includes("Invalid login credentials") ||
            error.message.includes("NotAuthorizedException")
          ) {
            toast({
              title: "Login failed",
              description: "Invalid email or password. Please try again.",
              variant: "destructive",
            });
          } else if (
            error.message.includes("Email not confirmed") ||
            error.message.includes("UserNotConfirmedException")
          ) {
            setIsVerifying(true);
            toast({
              title: "Email not verified",
              description: "Enter the verification code sent to your email.",
              variant: "destructive",
            });
          } else {
            toast({
              title: "Login failed",
              description: error.message,
              variant: "destructive",
            });
          }
        } else {
          toast({
            title: "Welcome back",
            description: "You have successfully logged in.",
          });
          navigate("/dashboard");
        }
      } else {
        const { error } = await signUp(email, password, fullName);
        if (error) {
          if (
            error.message.includes("User already registered") ||
            error.message.includes("UsernameExistsException")
          ) {
            toast({
              title: "Account exists",
              description: "An account with this email already exists. Please sign in instead.",
              variant: "destructive",
            });
          } else {
            toast({
              title: "Sign up failed",
              description: error.message,
              variant: "destructive",
            });
          }
        } else {
          setIsVerifying(true);
          toast({
            title: "Account created",
            description: "Enter the verification code sent to your email.",
          });
        }
      }
    } catch {
      toast({
        title: "Error",
        description: "An unexpected error occurred. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mx-auto grid min-h-[calc(100vh-4rem)] w-full max-w-6xl overflow-hidden rounded-[36px] border border-white/70 bg-white/70 shadow-elevated backdrop-blur-sm lg:grid-cols-[0.92fr_1.08fr]"
      >
        <div className="gradient-hero relative hidden overflow-hidden p-10 lg:block">
          <div className="absolute inset-0 mesh-grid opacity-50" />
          <div className="relative flex h-full flex-col justify-between">
            <div className="space-y-8">
              <Link to="/" className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl gradient-primary shadow-card">
                  <MessageSquare className="h-6 w-6 text-primary-foreground" />
                </div>
                <div>
                  <p className="heading-display text-2xl font-bold text-foreground">CampusVoice</p>
                  <p className="text-sm text-muted-foreground">Student support with visible progress</p>
                </div>
              </Link>

              <div className="space-y-4">
                <p className="text-sm font-semibold uppercase tracking-[0.18em] text-primary">Secure access</p>
                <h1 className="heading-display max-w-md text-5xl font-bold leading-[0.96]">
                  Sign in once, then track every complaint clearly.
                </h1>
                <p className="max-w-md text-base leading-8 text-muted-foreground">
                  CampusVoice keeps complaint submission, notifications, and resolution updates in one place, with Google sign-in and anonymous reporting support.
                </p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                "Anonymous complaint submission by default",
                "Notifications for replies, assignments, and resolutions",
                "Timeline-based tracking from submission to closure",
              ].map((item) => (
                <div key={item} className="surface-soft flex items-center gap-3 px-4 py-3 text-sm">
                  <CheckCircle2 className="h-4 w-4 text-success" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-center p-5 sm:p-8 lg:p-10">
          <Card className="w-full max-w-xl border-white/70 bg-white/90 shadow-none">
            <CardHeader className="space-y-3 text-left">
              <div className="flex items-center gap-3 lg:hidden">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl gradient-primary shadow-card">
                  <MessageSquare className="h-5 w-5 text-primary-foreground" />
                </div>
                <div>
                  <p className="heading-display text-xl font-bold text-foreground">CampusVoice</p>
                  <p className="text-xs text-muted-foreground">Student support with visible progress</p>
                </div>
              </div>
              <CardTitle className="heading-display text-3xl font-bold">
                {isResetting
                  ? resetStep === "request"
                    ? "Forgot password"
                    : "Reset your password"
                  : isVerifying
                    ? "Verify your email"
                    : isLogin
                      ? "Welcome back"
                      : "Create account"}
              </CardTitle>
              <CardDescription className="max-w-md text-sm leading-7">
                {isResetting
                  ? resetStep === "request"
                    ? "Enter your email and we will send a reset code."
                    : "Enter the code from your inbox and choose a new password."
                  : isVerifying
                    ? "Enter the verification code sent to your email address."
                    : isLogin
                      ? "Sign in to manage your complaints, notifications, and updates."
                      : "Create your account to start submitting complaints and tracking their progress."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!isVerifying && !isResetting && googleAuthEnabled && (
                <div className="mb-5 space-y-3">
                  <Button
                    type="button"
                    variant="outline"
                    className="h-11 w-full justify-center rounded-2xl bg-background/80"
                    onClick={() => void handleGoogleSignIn()}
                  >
                    <GoogleIcon />
                    Sign in with Google
                  </Button>
                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <span className="w-full border-t" />
                    </div>
                    <div className="relative flex justify-center text-[11px] uppercase tracking-[0.18em]">
                      <span className="bg-card px-3 text-muted-foreground">or use email</span>
                    </div>
                  </div>
                </div>
              )}

            <form onSubmit={handleSubmit} className="space-y-4">
              {!isLogin && !isVerifying && !isResetting && (
                <div className="space-y-2">
                  <Label htmlFor="fullName">Full Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <Input
                      id="fullName"
                      type="text"
                      placeholder="John Doe"
                      value={fullName}
                      onChange={(event) => setFullName(event.target.value)}
                      className="pl-10"
                    />
                  </div>
                  {errors.fullName && (
                    <p className="flex items-center gap-1 text-sm text-destructive">
                      <AlertCircle className="h-3 w-3" />
                      {errors.fullName}
                    </p>
                  )}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="student@university.edu"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    className="pl-10"
                  />
                </div>
                {errors.email && (
                  <p className="flex items-center gap-1 text-sm text-destructive">
                    <AlertCircle className="h-3 w-3" />
                    {errors.email}
                  </p>
                )}
              </div>

              {isResetting ? (
                resetStep === "request" ? (
                  <div className="rounded-md border bg-muted/20 p-3 text-sm text-muted-foreground">
                    We will send a password reset code to your email.
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Label htmlFor="resetCode">Reset Code</Label>
                    <Input
                      id="resetCode"
                      type="text"
                      placeholder="Enter reset code"
                      value={resetCode}
                      onChange={(event) => setResetCode(event.target.value)}
                    />
                    <Label htmlFor="newPassword">New Password</Label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        id="newPassword"
                        type={showNewPassword ? "text" : "password"}
                        placeholder="Choose a secure password"
                        value={newPassword}
                        onChange={(event) => setNewPassword(event.target.value)}
                        className="pl-10 pr-10"
                      />
                      <button
                        type="button"
                        onClick={() => setShowNewPassword((value) => !value)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                      >
                        {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </button>
                    </div>
                    {errors.password && (
                      <p className="flex items-center gap-1 text-sm text-destructive">
                        <AlertCircle className="h-3 w-3" />
                        {errors.password}
                      </p>
                    )}
                  </div>
                )
              ) : isVerifying ? (
                <div className="space-y-2">
                  <Label htmlFor="verificationCode">Verification Code</Label>
                  <Input
                    id="verificationCode"
                    type="text"
                    placeholder="Enter 6-digit code"
                    value={verificationCode}
                    onChange={(event) => setVerificationCode(event.target.value)}
                  />
                  {errors.password && (
                    <p className="flex items-center gap-1 text-sm text-destructive">
                      <AlertCircle className="h-3 w-3" />
                      {errors.password}
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      className="pl-10 pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((value) => !value)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  {errors.password && (
                    <p className="flex items-center gap-1 text-sm text-destructive">
                      <AlertCircle className="h-3 w-3" />
                      {errors.password}
                    </p>
                  )}
                </div>
              )}

              {showPasswordRules && <PasswordChecklist value={activePasswordValue} />}

              <Button type="submit" variant="hero" className="mt-2 h-11 w-full rounded-2xl" disabled={isSubmitting}>
                {isSubmitting
                  ? "Please wait..."
                  : isResetting
                    ? resetStep === "request"
                      ? "Send Reset Code"
                      : "Reset Password"
                    : isVerifying
                      ? "Verify Email"
                      : isLogin
                        ? "Sign In"
                        : "Create Account"}
              </Button>
            </form>

            {isVerifying && (
              <div className="mt-4 text-center text-sm">
                <button
                  type="button"
                  onClick={async () => {
                    const { error } = await resendSignUpCode(email);
                    if (error) {
                      toast({
                        title: "Could not resend code",
                        description: error.message,
                        variant: "destructive",
                      });
                    } else {
                      toast({
                        title: "Code sent",
                        description: "A new verification code has been sent to your email.",
                      });
                    }
                  }}
                  className="font-medium text-primary hover:underline"
                >
                  Resend verification code
                </button>
              </div>
            )}

            {isLogin && !isVerifying && !isResetting && (
              <div className="mt-3 text-center text-sm">
                <button
                  type="button"
                  className="font-medium text-primary hover:underline"
                  onClick={() => {
                    setIsResetting(true);
                    setResetStep("request");
                    setErrors({});
                  }}
                >
                  Forgot password?
                </button>
              </div>
            )}

            <div className="mt-6 text-center text-sm">
              <span className="text-muted-foreground">
                {isResetting
                  ? "Remembered your password? "
                  : isVerifying
                    ? "Want to use a different account? "
                    : isLogin
                      ? "Don't have an account? "
                      : "Already have an account? "}
              </span>
              <button
                type="button"
                onClick={() => {
                  if (isResetting) {
                    setIsResetting(false);
                    setResetStep("request");
                    setResetCode("");
                    setNewPassword("");
                    setIsLogin(true);
                  } else if (isVerifying) {
                    setIsVerifying(false);
                    setIsLogin(true);
                    setVerificationCode("");
                  } else {
                    setIsLogin(!isLogin);
                  }
                  setErrors({});
                }}
                className="font-medium text-primary hover:underline"
              >
                {isResetting ? "Go to sign in" : isVerifying ? "Go to sign in" : isLogin ? "Sign up" : "Sign in"}
              </button>
            </div>

            {!isVerifying && !isResetting && (
              <div className="mt-3 text-center text-sm">
                <Link to="/admin-login" className="font-medium text-primary hover:underline">
                  Admin login
                </Link>
              </div>
            )}
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  );
};

export default Auth;
