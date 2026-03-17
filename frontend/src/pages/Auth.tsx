import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { MessageSquare, Mail, Lock, User, AlertCircle, Eye, EyeOff } from "lucide-react";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";

const emailSchema = z.string().email("Please enter a valid email address");
const passwordSchema = z.string().min(6, "Password must be at least 6 characters");
const nameSchema = z.string().min(2, "Name must be at least 2 characters");

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

  const validateForm = () => {
    const newErrors: { email?: string; password?: string; fullName?: string } = {};

    const emailResult = emailSchema.safeParse(email);
    if (!emailResult.success) {
      newErrors.email = emailResult.error.errors[0].message;
    }

    if (isResetting) {
      if (resetStep === "confirm") {
        if (!resetCode.trim()) {
          newErrors.password = "Reset code is required";
        } else {
          const newPasswordResult = passwordSchema.safeParse(newPassword);
          if (!newPasswordResult.success) {
            newErrors.password = newPasswordResult.error.errors[0].message;
          }
        }
      }
    } else if (!isVerifying) {
      const passwordResult = passwordSchema.safeParse(password);
      if (!passwordResult.success) {
        newErrors.password = passwordResult.error.errors[0].message;
      }
    } else if (!verificationCode.trim()) {
      newErrors.password = "Verification code is required";
    }

    if (!isLogin && !isVerifying) {
      const nameResult = nameSchema.safeParse(fullName);
      if (!nameResult.success) {
        newErrors.fullName = nameResult.error.errors[0].message;
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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
            title: "Welcome back!",
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
              description: "An account with this email already exists. Please log in instead.",
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
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md">
        <div className="flex justify-center mb-8">
          <div className="flex items-center gap-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl gradient-primary">
              <MessageSquare className="h-6 w-6 text-primary-foreground" />
            </div>
            <span className="text-2xl font-bold text-foreground">CampusVoice</span>
          </div>
        </div>

        <Card className="border-border/50 shadow-card">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">
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
            <CardDescription>
              {isResetting
                ? resetStep === "request"
                  ? "Enter your email to get a reset code"
                  : "Enter the code and set a new password"
                : isVerifying
                ? "Enter the verification code from your email"
                : isLogin
                ? "Sign in to manage your complaints"
                : "Sign up to start submitting complaints"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {!isLogin && !isVerifying && !isResetting && (
                <div className="space-y-2">
                  <Label htmlFor="fullName">Full Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="fullName"
                      type="text"
                      placeholder="John Doe"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  {errors.fullName && (
                    <p className="text-sm text-destructive flex items-center gap-1">
                      <AlertCircle className="h-3 w-3" />
                      {errors.fullName}
                    </p>
                  )}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="student@university.edu"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-10"
                  />
                </div>
                {errors.email && (
                  <p className="text-sm text-destructive flex items-center gap-1">
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
                      onChange={(e) => setResetCode(e.target.value)}
                    />
                    <Label htmlFor="newPassword">New Password</Label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="newPassword"
                        type={showNewPassword ? "text" : "password"}
                        placeholder="Enter new password"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        className="pl-10 pr-10"
                      />
                      <button
                        type="button"
                        onClick={() => setShowNewPassword((v) => !v)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                      >
                        {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </button>
                    </div>
                    {errors.password && (
                      <p className="text-sm text-destructive flex items-center gap-1">
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
                    onChange={(e) => setVerificationCode(e.target.value)}
                  />
                  {errors.password && (
                    <p className="text-sm text-destructive flex items-center gap-1">
                      <AlertCircle className="h-3 w-3" />
                      {errors.password}
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="********"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="pl-10 pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((v) => !v)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  {errors.password && (
                    <p className="text-sm text-destructive flex items-center gap-1">
                      <AlertCircle className="h-3 w-3" />
                      {errors.password}
                    </p>
                  )}
                </div>
              )}

              <Button type="submit" variant="hero" className="w-full" disabled={isSubmitting}>
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
                  className="text-primary hover:underline font-medium"
                >
                  Resend verification code
                </button>
              </div>
            )}

            {isLogin && !isVerifying && !isResetting && (
              <div className="mt-3 text-center text-sm">
                <button
                  type="button"
                  className="text-primary hover:underline font-medium"
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
                className="text-primary hover:underline font-medium"
              >
                {isResetting ? "Go to sign in" : isVerifying ? "Go to sign in" : isLogin ? "Sign up" : "Sign in"}
              </button>
            </div>
            {!isVerifying && !isResetting && (
              <div className="mt-3 text-center text-sm">
                <Link to="/admin-login" className="text-primary hover:underline font-medium">
                  Admin login
                </Link>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default Auth;
