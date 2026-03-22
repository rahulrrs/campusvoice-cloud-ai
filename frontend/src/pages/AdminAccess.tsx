import { useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ShieldCheck, UserCog } from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { complaintsApi } from "@/integrations/aws/client";
import { useToast } from "@/hooks/use-toast";
import { useAccessProfile } from "@/hooks/useAccessProfile";

const AdminAccess = () => {
  const { user, loading } = useAuth();
  const { data: accessProfile, isLoading } = useAccessProfile();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [loading, navigate, user]);

  useEffect(() => {
    if (accessProfile?.is_super_admin) {
      navigate("/super-admin");
    } else if (accessProfile?.is_admin) {
      navigate("/admin");
    }
  }, [accessProfile?.is_admin, accessProfile?.is_super_admin, navigate]);

  const params = new URLSearchParams(location.search);
  const prefilledToken = params.get("token")?.trim() ?? "";

  const acceptMutation = useMutation({
    mutationFn: (token: string) => complaintsApi.acceptAdminInvite(token),
    onSuccess: async (record) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["access-profile"] }),
        queryClient.invalidateQueries({ queryKey: ["admin-users"] }),
      ]);
      toast({
        title: "Admin access enabled",
        description: `You now have ${record.role === "super_admin" ? "super admin" : "admin"} access.`,
      });
      navigate(record.role === "super_admin" ? "/super-admin" : "/admin");
    },
    onError: (error: Error) => {
      toast({
        title: "Could not accept invite",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  if (loading || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  const invites = accessProfile?.pending_invites ?? [];

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="container mx-auto flex-1 px-4 py-10">
        <div className="mx-auto max-w-3xl space-y-6">
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldCheck className="h-5 w-5 text-primary" />
                Admin Access
              </CardTitle>
              <CardDescription>
                Accept the invite that matches your signed-in email to unlock the admin workspace.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <p>
                Signed in as <strong>{user?.email ?? "unknown user"}</strong>.
              </p>
              {prefilledToken ? (
                <Button
                  type="button"
                  variant="hero"
                  disabled={acceptMutation.isPending}
                  onClick={() => acceptMutation.mutate(prefilledToken)}
                >
                  {acceptMutation.isPending ? "Accepting..." : "Accept Invite From Link"}
                </Button>
              ) : null}
            </CardContent>
          </Card>

          {invites.length === 0 ? (
            <Card className="shadow-card">
              <CardContent className="py-10 text-center text-sm text-muted-foreground">
                <UserCog className="mx-auto mb-3 h-8 w-8 text-primary/70" />
                No pending admin invite was found for this email yet.
                <div className="mt-4">
                  <Link to="/admin-login">
                    <Button variant="outline">Back to Admin Login</Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          ) : (
            invites.map((invite) => (
              <Card key={invite.id} className="shadow-card">
                <CardHeader>
                  <CardTitle className="text-lg capitalize">
                    {invite.role === "super_admin" ? "Super Admin Invite" : "Admin Invite"}
                  </CardTitle>
                  <CardDescription>
                    Invited by {invite.invited_by ?? "system"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4 text-sm">
                  <p>Status: <strong className="capitalize">{invite.status}</strong></p>
                  {invite.invite_expires_at ? (
                    <p>Expires: <strong>{new Date(invite.invite_expires_at).toLocaleString()}</strong></p>
                  ) : null}
                  <Button
                    type="button"
                    variant="hero"
                    disabled={acceptMutation.isPending || !invite.invite_token}
                    onClick={() => invite.invite_token && acceptMutation.mutate(invite.invite_token)}
                  >
                    {acceptMutation.isPending ? "Accepting..." : "Accept This Invite"}
                  </Button>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default AdminAccess;
