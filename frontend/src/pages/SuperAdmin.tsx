import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, ShieldCheck, UserCog, Users } from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAuth } from "@/contexts/AuthContext";
import { complaintsApi, type AdminAccessRecord } from "@/integrations/aws/client";
import { useToast } from "@/hooks/use-toast";
import { useAccessProfile } from "@/hooks/useAccessProfile";

const SuperAdmin = () => {
  const { user, loading } = useAuth();
  const { data: accessProfile, isLoading: accessLoading } = useAccessProfile();
  const navigate = useNavigate();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"admin" | "super_admin">("admin");

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [loading, navigate, user]);

  useEffect(() => {
    if (!accessLoading && accessProfile && !accessProfile.is_super_admin) {
      navigate(accessProfile.is_admin ? "/admin" : "/");
    }
  }, [accessLoading, accessProfile, navigate]);

  const adminUsersQuery = useQuery({
    queryKey: ["admin-users"],
    queryFn: complaintsApi.listAdminUsers,
    enabled: Boolean(accessProfile?.is_super_admin),
    staleTime: 30_000,
  });

  const inviteMutation = useMutation({
    mutationFn: complaintsApi.inviteAdminUser,
    onSuccess: async () => {
      setInviteEmail("");
      setInviteRole("admin");
      await queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      toast({
        title: "Invite created",
        description: "The access record is ready. Share the invite link with that email owner.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Could not create invite",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: { role?: "admin" | "super_admin"; status?: "pending" | "active" | "suspended" | "revoked" } }) =>
      complaintsApi.updateAdminUserAccess(id, payload),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      await queryClient.invalidateQueries({ queryKey: ["access-profile"] });
      toast({
        title: "Access updated",
        description: "The admin role or status was updated successfully.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Could not update access",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const records = adminUsersQuery.data ?? [];
  const summary = useMemo(() => {
    return {
      total: records.length,
      active: records.filter((item) => item.status === "active").length,
      pending: records.filter((item) => item.status === "pending").length,
      superAdmins: records.filter((item) => item.role === "super_admin").length,
    };
  }, [records]);

  const copyInviteLink = async (record: AdminAccessRecord) => {
    if (!record.invite_token) {
      toast({
        title: "Invite already accepted",
        description: "This record no longer has a pending invite token.",
      });
      return;
    }
    const url = `${window.location.origin}/admin-access?token=${encodeURIComponent(record.invite_token)}`;
    await navigator.clipboard.writeText(url);
    toast({
      title: "Invite link copied",
      description: `Share the link with ${record.email}.`,
    });
  };

  if (loading || accessLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  if (!accessProfile?.is_super_admin) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="container mx-auto flex-1 px-4 py-12">
          <Card className="mx-auto max-w-2xl shadow-card">
            <CardHeader>
              <CardTitle>Super Admin Access Required</CardTitle>
              <CardDescription>
                Only super admins can manage other admin accounts and invitations.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/admin">
                <Button variant="outline">Go to Admin Dashboard</Button>
              </Link>
            </CardContent>
          </Card>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="container mx-auto flex-1 space-y-6 px-4 py-8">
        <section className="grid gap-4 md:grid-cols-4">
          <Card className="shadow-card">
            <CardHeader><CardTitle className="text-sm">Admin Records</CardTitle></CardHeader>
            <CardContent className="text-3xl font-semibold">{summary.total}</CardContent>
          </Card>
          <Card className="shadow-card">
            <CardHeader><CardTitle className="text-sm">Active</CardTitle></CardHeader>
            <CardContent className="text-3xl font-semibold text-emerald-600">{summary.active}</CardContent>
          </Card>
          <Card className="shadow-card">
            <CardHeader><CardTitle className="text-sm">Pending</CardTitle></CardHeader>
            <CardContent className="text-3xl font-semibold text-amber-600">{summary.pending}</CardContent>
          </Card>
          <Card className="shadow-card">
            <CardHeader><CardTitle className="text-sm">Super Admins</CardTitle></CardHeader>
            <CardContent className="text-3xl font-semibold text-primary">{summary.superAdmins}</CardContent>
          </Card>
        </section>

        <section className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserCog className="h-5 w-5 text-primary" />
                Grant Admin Access
              </CardTitle>
              <CardDescription>
                Create an invite linked to a verified email. The user must sign in with the same email before they can accept it.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Email</label>
                <Input
                  type="email"
                  value={inviteEmail}
                  onChange={(event) => setInviteEmail(event.target.value)}
                  placeholder="staff@college.edu"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Role</label>
                <Select value={inviteRole} onValueChange={(value) => setInviteRole(value as "admin" | "super_admin")}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="admin">Admin</SelectItem>
                    <SelectItem value="super_admin">Super Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button
                type="button"
                variant="hero"
                className="w-full"
                disabled={inviteMutation.isPending || !inviteEmail.trim()}
                onClick={() => inviteMutation.mutate({ email: inviteEmail.trim(), role: inviteRole })}
              >
                {inviteMutation.isPending ? "Creating Invite..." : "Create Invite"}
              </Button>
            </CardContent>
          </Card>

          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5 text-primary" />
                Access Directory
              </CardTitle>
              <CardDescription>
                Review pending and active admin accounts, then promote, suspend, revoke, or reactivate them.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {adminUsersQuery.isLoading ? (
                <div className="flex items-center justify-center py-10">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
                </div>
              ) : records.length === 0 ? (
                <div className="rounded-2xl border border-dashed p-8 text-center text-sm text-muted-foreground">
                  No admin records yet.
                </div>
              ) : (
                records.map((record) => (
                  <div key={record.id} className="rounded-2xl border bg-card p-4">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <ShieldCheck className="h-4 w-4 text-primary" />
                          <span className="font-semibold">{record.email}</span>
                        </div>
                        <div className="flex flex-wrap gap-2 text-xs">
                          <span className="rounded-full border px-2.5 py-1 capitalize">{record.role.replace("_", " ")}</span>
                          <span className="rounded-full border px-2.5 py-1 capitalize">{record.status}</span>
                          {record.invite_expires_at ? (
                            <span className="rounded-full border px-2.5 py-1">
                              Expires {new Date(record.invite_expires_at).toLocaleDateString()}
                            </span>
                          ) : null}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Invited by {record.invited_by ?? "system"}
                          {record.accepted_at ? ` • Accepted ${new Date(record.accepted_at).toLocaleString()}` : ""}
                        </p>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        {record.invite_token ? (
                          <Button type="button" variant="outline" size="sm" onClick={() => void copyInviteLink(record)}>
                            <Copy className="mr-2 h-4 w-4" />
                            Copy Invite
                          </Button>
                        ) : null}
                        {record.status !== "active" ? (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            disabled={updateMutation.isPending}
                            onClick={() => updateMutation.mutate({ id: record.id, payload: { status: "active" } })}
                          >
                            Activate
                          </Button>
                        ) : null}
                        {record.status === "active" ? (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            disabled={updateMutation.isPending}
                            onClick={() => updateMutation.mutate({ id: record.id, payload: { status: "suspended" } })}
                          >
                            Suspend
                          </Button>
                        ) : null}
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          disabled={updateMutation.isPending}
                          onClick={() =>
                            updateMutation.mutate({
                              id: record.id,
                              payload: { role: record.role === "super_admin" ? "admin" : "super_admin" },
                            })
                          }
                        >
                          {record.role === "super_admin" ? "Make Admin" : "Promote"}
                        </Button>
                        <Button
                          type="button"
                          variant="destructive"
                          size="sm"
                          disabled={updateMutation.isPending}
                          onClick={() => updateMutation.mutate({ id: record.id, payload: { status: "revoked" } })}
                        >
                          Revoke
                        </Button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default SuperAdmin;
