import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Sparkles, ShieldAlert, RefreshCw } from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/contexts/AuthContext";
import { complaintsApi, type ComplaintRecord, type PredictionResult } from "@/integrations/aws/client";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const getAdminEmails = () =>
  (import.meta.env.VITE_ADMIN_EMAILS ?? "")
    .split(",")
    .map((v: string) => v.trim().toLowerCase())
    .filter((v: string) => v.length > 0);

const Admin = () => {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [predictions, setPredictions] = useState<Record<string, PredictionResult>>({});
  const [manualDrafts, setManualDrafts] = useState<
    Record<string, { category: string; priority: string; department: string }>
  >({});

  const isAdmin = useMemo(() => {
    if (!user?.email) return false;
    return getAdminEmails().includes(user.email.toLowerCase());
  }, [user?.email]);

  useEffect(() => {
    if (!loading && !user) {
      navigate("/admin-login");
    }
  }, [loading, navigate, user]);

  const complaintsQuery = useQuery({
    queryKey: ["admin-complaints"],
    queryFn: complaintsApi.listAllForAdmin,
    enabled: !!user && isAdmin,
    staleTime: 30_000,
  });

  useEffect(() => {
    const complaints = (complaintsQuery.data ?? []) as ComplaintRecord[];
    if (complaints.length === 0) return;

    setManualDrafts((prev) => {
      const next = { ...prev };
      for (const complaint of complaints) {
        if (!next[complaint.id]) {
          next[complaint.id] = {
            category: complaint.category ?? "Uncategorized",
            priority: complaint.priority ?? "medium",
            department: complaint.department ?? "",
          };
        }
      }
      return next;
    });
  }, [complaintsQuery.data]);

  const predictMutation = useMutation({
    mutationFn: async (complaintId: string) => complaintsApi.predictForComplaint(complaintId),
    onSuccess: (data, complaintId) => {
      setPredictions((prev) => ({ ...prev, [complaintId]: data }));
    },
    onError: (error: Error) => {
      toast({
        title: "Prediction failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const autoApplyMutation = useMutation({
    mutationFn: complaintsApi.autoApplyPrediction,
    onSuccess: (data) => {
      setPredictions((prev) => ({ ...prev, [data.complaint.id]: data.prediction }));
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      toast({
        title: "AI applied",
        description: "Category and priority updated from model prediction.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Auto-apply failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const autoClassifyMutation = useMutation({
    mutationFn: () => complaintsApi.autoClassifyAll(true),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      toast({
        title: "Auto-classification complete",
        description: `${data.updatedCount} pending complaints updated by AI.`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Bulk classify failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const updateStatusMutation = useMutation({
    mutationFn: async ({ complaintId, status }: { complaintId: string; status: string }) =>
      complaintsApi.updateComplaintByAdmin(complaintId, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Status update failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const approveMutation = useMutation({
    mutationFn: complaintsApi.approveComplaint,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      toast({
        title: "Complaint approved",
        description: "Status updated to in-progress.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Approve failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const manualUpdateMutation = useMutation({
    mutationFn: async ({
      complaintId,
      category,
      priority,
      department,
    }: {
      complaintId: string;
      category: string;
      priority: string;
      department: string;
    }) =>
      complaintsApi.updateComplaintByAdmin(complaintId, {
        category,
        priority,
        department,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      toast({
        title: "Manual update saved",
        description: "Category, priority, and department updated.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Manual update failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  if (loading || complaintsQuery.isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex-1 flex items-center justify-center p-6">
          <Card className="max-w-lg w-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldAlert className="h-5 w-5 text-destructive" />
                Admin access required
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              This account is not configured as admin.
            </CardContent>
          </Card>
        </main>
        <Footer />
      </div>
    );
  }

  const complaints = (complaintsQuery.data ?? []) as ComplaintRecord[];
  const pendingUnknown = complaints.filter(
    (c) =>
      c.status === "pending" &&
      (!c.category || c.category === "Uncategorized" || c.category === "Unknown")
  ).length;

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 py-8 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold">Admin Console</h1>
            <p className="text-sm text-muted-foreground">
              Review all complaints, auto-classify with AI, and manage status.
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => complaintsQuery.refetch()}
              disabled={complaintsQuery.isFetching}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button
              type="button"
              variant="hero"
              onClick={() => autoClassifyMutation.mutate()}
              disabled={autoClassifyMutation.isPending || pendingUnknown === 0}
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Auto Classify Pending ({pendingUnknown})
            </Button>
          </div>
        </div>

        {complaints.map((complaint) => {
          const prediction = predictions[complaint.id];
          const draft = manualDrafts[complaint.id] ?? {
            category: complaint.category ?? "Uncategorized",
            priority: complaint.priority ?? "medium",
            department: complaint.department ?? "",
          };
          return (
            <Card key={complaint.id}>
              <CardHeader>
                <CardTitle className="text-lg">{complaint.title}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">{complaint.description}</p>
                <div className="grid md:grid-cols-3 gap-3 text-sm">
                  <div>
                    Category: <strong>{complaint.category ?? "Uncategorized"}</strong>
                  </div>
                  <div>
                    Priority: <strong>{complaint.priority ?? "medium"}</strong>
                  </div>
                  <div>
                    Status: <strong>{complaint.status}</strong>
                  </div>
                </div>
                <div className="text-sm">
                  Department: <strong>{complaint.department ?? "Not assigned"}</strong>
                </div>

                {prediction && (
                  <div className="rounded-md border p-3 text-sm">
                    <div>
                      AI Category: <strong>{prediction.label}</strong> (
                      {(prediction.label_confidence * 100).toFixed(1)}%)
                    </div>
                    <div>
                      AI Priority: <strong>{prediction.priority}</strong> (
                      {(prediction.priority_confidence * 100).toFixed(1)}%)
                    </div>
                    <div>
                      Department: <strong>{prediction.department}</strong>
                    </div>
                  </div>
                )}

                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => predictMutation.mutate(complaint.id)}
                    disabled={predictMutation.isPending}
                  >
                    Predict Only
                  </Button>
                  <Button
                    type="button"
                    variant="hero"
                    onClick={() => autoApplyMutation.mutate(complaint.id)}
                    disabled={autoApplyMutation.isPending}
                  >
                    Auto Apply AI
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => approveMutation.mutate(complaint.id)}
                    disabled={approveMutation.isPending || complaint.status !== "pending"}
                  >
                    Approve Complaint
                  </Button>
                  <Select
                    value={complaint.status}
                    onValueChange={(status) =>
                      updateStatusMutation.mutate({ complaintId: complaint.id, status })
                    }
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Change status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pending">pending</SelectItem>
                      <SelectItem value="in-progress">in-progress</SelectItem>
                      <SelectItem value="resolved">resolved</SelectItem>
                      <SelectItem value="rejected">rejected</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="rounded-md border p-3 space-y-2">
                  <div className="text-sm font-medium">Manual Update</div>
                  <div className="grid md:grid-cols-4 gap-2">
                    <input
                      className="h-10 rounded-md border bg-background px-3 text-sm"
                      placeholder="Category"
                      value={draft.category}
                      onChange={(event) =>
                        setManualDrafts((prev) => ({
                          ...prev,
                          [complaint.id]: {
                            ...draft,
                            category: event.target.value,
                          },
                        }))
                      }
                    />
                    <Select
                      value={draft.priority}
                      onValueChange={(priority) =>
                        setManualDrafts((prev) => ({
                          ...prev,
                          [complaint.id]: {
                            ...draft,
                            priority,
                          },
                        }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Priority" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">low</SelectItem>
                        <SelectItem value="medium">medium</SelectItem>
                        <SelectItem value="high">high</SelectItem>
                        <SelectItem value="unknown">unknown</SelectItem>
                      </SelectContent>
                    </Select>
                    <input
                      className="h-10 rounded-md border bg-background px-3 text-sm"
                      placeholder="Department"
                      value={draft.department}
                      onChange={(event) =>
                        setManualDrafts((prev) => ({
                          ...prev,
                          [complaint.id]: {
                            ...draft,
                            department: event.target.value,
                          },
                        }))
                      }
                    />
                    <Button
                      type="button"
                      variant="outline"
                      disabled={manualUpdateMutation.isPending}
                      onClick={() =>
                        manualUpdateMutation.mutate({
                          complaintId: complaint.id,
                          category: draft.category,
                          priority: draft.priority,
                          department: draft.department,
                        })
                      }
                    >
                      Save Manual Update
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </main>
      <Footer />
    </div>
  );
};

export default Admin;
