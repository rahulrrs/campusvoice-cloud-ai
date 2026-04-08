import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Search, ShieldAlert } from "lucide-react";
import Header from "@/components/layout/Header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/contexts/AuthContext";
import {
  complaintsApi,
  type ComplaintAnalysisBundle,
  type ComplaintRecord,
} from "@/integrations/aws/client";
import { useToast } from "@/hooks/use-toast";
import { useAccessProfile } from "@/hooks/useAccessProfile";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import ComplaintUpdatesPanel from "@/components/complaints/ComplaintUpdatesPanel";
import AdminComplaintQueue from "@/components/admin/AdminComplaintQueue";

type AdminStatusFilter = "all" | "submitted" | "pending" | "in-progress" | "resolved" | "rejected";
type VisibilityFilter = "all" | "anonymous" | "identified";

const normalizeStatus = (status?: string | null) => (!status ? "submitted" : status === "in_progress" ? "in-progress" : status);
const statusFeedback = (rawStatus?: string) => {
  const status = normalizeStatus(rawStatus);
  switch (status) {
    case "submitted":
      return {
        title: "Case moved to submitted",
        description: "The complaint is back in the intake queue.",
      };
    case "pending":
      return {
        title: "Case moved to pending",
        description: "The complaint is queued for active handling.",
      };
    case "in-progress":
      return {
        title: "Case marked in progress",
        description: "The complaint is now in active handling.",
      };
    case "resolved":
      return {
        title: "Resolution published",
        description: "The complaint was closed with the current summary.",
      };
    case "rejected":
      return {
        title: "Case rejected",
        description: "The complaint was closed without resolution.",
      };
    default:
      return {
        title: "Case updated",
        description: "The complaint status was updated.",
      };
  }
};

const ADMIN_REPLY_TEMPLATES = [
  {
    label: "Ask for Details",
    body: "Thank you for reporting this. Please share any missing details such as the exact location, time, and any photo or screenshot if available so we can proceed faster.",
  },
  {
    label: "Assigned",
    body: "Your complaint has been assigned to the concerned team. We are reviewing it now and will update you once there is progress.",
  },
  {
    label: "Resolved Update",
    body: "We have completed the current action on this complaint. Please review the resolution summary, and if the issue continues, you can reopen the complaint from your dashboard.",
  },
  {
    label: "Internal Escalation",
    body: "Escalate to the department lead and confirm whether any blocker is preventing closure.",
    isInternal: true,
  },
] satisfies Array<{ label: string; body: string; isInternal?: boolean }>;

const ComplaintThreadCard = ({ complaintId }: { complaintId: string }) => {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const updatesQuery = useQuery({
    queryKey: ["admin-complaint-updates", complaintId],
    queryFn: () => complaintsApi.listComplaintUpdatesForAdmin(complaintId),
  });

  useEffect(() => {
    if (updatesQuery.data) {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
    }
  }, [updatesQuery.data, queryClient]);

  const mutation = useMutation({
    mutationFn: ({ body, isInternal }: { body: string; isInternal?: boolean }) =>
      complaintsApi.createComplaintUpdateForAdmin(complaintId, {
        body,
        is_internal: isInternal,
      }),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaint-updates", complaintId] });
      toast({
        title: variables.isInternal ? "Internal note saved" : "Public update sent",
        description: variables.isInternal
          ? "This note is only visible to admins."
          : "The student can now see this message in their complaint conversation and notifications.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Could not post update",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  return (
    <ComplaintUpdatesPanel
      title="Conversation"
      updates={updatesQuery.data ?? []}
      placeholder="Post a public update or an internal admin note."
      submitLabel="Post Message"
      canPostInternal
      embedded
      hideTitle
      templates={ADMIN_REPLY_TEMPLATES}
      onSubmit={async (body, isInternal) => {
        await mutation.mutateAsync({ body, isInternal });
      }}
      isSubmitting={mutation.isPending}
    />
  );
};

const Admin = () => {
  const { user, loading } = useAuth();
  const accessProfileQuery = useAccessProfile();
  const accessProfile = accessProfileQuery.data;
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [predictions, setPredictions] = useState<Record<string, ComplaintAnalysisBundle>>({});
  const [attachmentPreviews, setAttachmentPreviews] = useState<Record<string, Array<{ key: string; url: string }>>>({});
  const [attachmentLoading, setAttachmentLoading] = useState<Record<string, boolean>>({});
  const [assignmentDrafts, setAssignmentDrafts] = useState<Record<string, string>>({});
  const [noteDrafts, setNoteDrafts] = useState<Record<string, string>>({});
  const [resolutionDrafts, setResolutionDrafts] = useState<Record<string, string>>({});
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<AdminStatusFilter>("all");
  const [visibilityFilter, setVisibilityFilter] = useState<VisibilityFilter>("all");
  const [departmentFilter, setDepartmentFilter] = useState("all");

  const deferredSearchQuery = useDeferredValue(searchQuery);
  const isAdmin = Boolean(accessProfile?.is_admin);
  const isAccessLoading = Boolean(user) && accessProfileQuery.isPending;

  useEffect(() => {
    if (!loading && !user) {
      navigate("/admin-login");
    }
  }, [loading, navigate, user]);

  const complaintsQuery = useQuery({
    queryKey: ["admin-complaints", statusFilter, departmentFilter],
    queryFn: () =>
      complaintsApi.listAllForAdmin({
        status: statusFilter,
        department: departmentFilter,
      }),
    enabled: !!user && isAdmin,
    staleTime: 30_000,
    retry: 1,
  });

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
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      setPredictions((prev) => ({
        ...prev,
        [data.complaint.id]: {
          classification: data.prediction,
          sentiment: {
            sentiment_score: 0,
            sentiment_label: "neutral",
            emotion: "neutral",
            emotion_intensity: 0,
            urgency_score: 0,
          },
          abuse: { toxicity_score: 0, spam_score: 0, flags: [] },
          duplicate_detection: { is_duplicate: false, score: 0, method: "n/a", matches: [] },
          recommendations: [],
          knowledge_graph: {
            department: data.prediction.department,
            issue_type: data.prediction.label,
            priority: data.prediction.priority,
            entities: [data.prediction.department, data.prediction.label],
          },
          source_language: "en",
        },
      }));
      toast({
        title: "AI applied",
        description: "Category and priority were updated from the model prediction.",
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

  const updateStatusMutation = useMutation({
    mutationFn: async ({
      complaintId,
      status,
      resolution_summary,
    }: {
      complaintId: string;
      status: string;
      resolution_summary?: string;
    }) => complaintsApi.updateComplaintByAdmin(complaintId, { status, resolution_summary }),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      const feedback = statusFeedback(variables.status);
      toast({
        title: feedback.title,
        description: feedback.description,
      });
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
        title: "Complaint routed",
        description: "The complaint moved into the active queue.",
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

  const workflowMutation = useMutation({
    mutationFn: async ({
      complaintId,
      assigned_to,
      admin_notes,
    }: {
      complaintId: string;
      assigned_to?: string | null;
      admin_notes?: string | null;
    }) => complaintsApi.updateComplaintByAdmin(complaintId, { assigned_to, admin_notes }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      toast({
        title: "Case updated",
        description: "Assignment or notes were saved.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Update failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const getAttachmentKind = (key: string) => {
    const lower = key.toLowerCase();
    if (/\.(png|jpg|jpeg|gif|webp|bmp|svg|avif)$/.test(lower)) return "image";
    if (/\.(mp3|wav|ogg|m4a|aac|webm)$/.test(lower)) return "audio";
    return "other";
  };

  const getAttachmentMimeType = (key: string) => {
    const lower = key.toLowerCase();
    if (lower.endsWith(".mp3")) return "audio/mpeg";
    if (lower.endsWith(".wav")) return "audio/wav";
    if (lower.endsWith(".ogg")) return "audio/ogg";
    if (lower.endsWith(".m4a")) return "audio/mp4";
    if (lower.endsWith(".aac")) return "audio/aac";
    if (lower.endsWith(".webm")) return "audio/webm";
    return undefined;
  };

  const loadAttachmentPreviews = async (complaint: ComplaintRecord) => {
    const keys = Array.isArray(complaint.attachments) ? complaint.attachments : [];
    if (keys.length === 0 || attachmentPreviews[complaint.id]?.length) return;
    setAttachmentLoading((prev) => ({ ...prev, [complaint.id]: true }));

    try {
      const results = await Promise.allSettled(
        keys.map(async (key) => {
          const data = await complaintsApi.createDownloadUrl({ key });
          return { key, url: data.downloadUrl };
        })
      );

      const urls = results
        .filter((result): result is PromiseFulfilledResult<{ key: string; url: string }> => result.status === "fulfilled")
        .map((result) => result.value);

      if (urls.length > 0) {
        setAttachmentPreviews((prev) => ({ ...prev, [complaint.id]: urls }));
      }
    } catch (error) {
      toast({
        title: "Could not load attachments",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setAttachmentLoading((prev) => ({ ...prev, [complaint.id]: false }));
    }
  };

  const complaints = ((complaintsQuery.data ?? []) as ComplaintRecord[]).map((item) => ({
    ...item,
    status: normalizeStatus(item.status),
  }));

  const departmentOptions = Array.from(
    new Set(
      complaints
        .map((item) => item.department)
        .filter((value): value is string => !!value && value.trim().length > 0)
    )
  ).sort((a, b) => a.localeCompare(b));

  const filteredComplaints = useMemo(() => {
    const needle = deferredSearchQuery.trim().toLowerCase();
    let items = complaints;

    if (visibilityFilter === "anonymous") {
      items = items.filter((item) => item.is_anonymous);
    } else if (visibilityFilter === "identified") {
      items = items.filter((item) => !item.is_anonymous);
    }

    if (!needle) return items;

    return items.filter((complaint) =>
      [
        complaint.id,
        complaint.title,
        complaint.description,
        complaint.category,
        complaint.department,
        complaint.assigned_to,
        complaint.student_name,
        complaint.student_email,
        complaint.student_phone,
        complaint.student_department,
        complaint.student_registration_number,
      ]
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [complaints, deferredSearchQuery, visibilityFilter]);

  const attentionCount = complaints.filter(
    (item) => item.has_unread_updates_for_admin || item.requires_human_review || item.decision_state === "escalated"
  ).length;

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-md">
            <CardHeader><CardTitle>Loading admin access</CardTitle></CardHeader>
            <CardContent className="flex items-center gap-3 text-sm text-muted-foreground">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-muted border-t-primary" />
              Checking your account and permissions...
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (isAccessLoading) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-md">
            <CardHeader><CardTitle>Loading admin access</CardTitle></CardHeader>
            <CardContent className="flex items-center gap-3 text-sm text-muted-foreground">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-muted border-t-primary" />
              Confirming your admin permissions...
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (accessProfileQuery.isError) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-xl">
            <CardHeader><CardTitle>Could not verify admin access</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>{(accessProfileQuery.error as Error)?.message ?? "Unknown error"}</p>
              <Button type="button" variant="outline" onClick={() => accessProfileQuery.refetch()}>
                Retry
              </Button>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldAlert className="h-5 w-5 text-destructive" />
                Admin access required
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              This account is not configured as an admin account.
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (complaintsQuery.isPending) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-md">
            <CardHeader><CardTitle>Loading admin workspace</CardTitle></CardHeader>
            <CardContent className="flex items-center gap-3 text-sm text-muted-foreground">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-muted border-t-primary" />
              Fetching complaint queue...
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (complaintsQuery.isError) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-xl">
            <CardHeader><CardTitle>Could not load admin complaints</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>{(complaintsQuery.error as Error)?.message ?? "Unknown error"}</p>
              <Button type="button" variant="outline" onClick={() => complaintsQuery.refetch()}>
                Retry
              </Button>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="container mx-auto flex-1 space-y-5 px-4 py-6">
        <section className="rounded-[26px] border border-slate-200/80 bg-white/95 px-5 py-4 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight text-slate-950">Admin Review</h1>
              <p className="max-w-3xl text-sm leading-6 text-slate-600">
                Review one case at a time. The queue stays visible, the active complaint stays focused, and only the next useful decision stays in view.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <p className="hidden text-sm text-slate-500 lg:block">
                {filteredComplaints.length} in queue{attentionCount ? `, ${attentionCount} need attention` : ""}
              </p>
              <Button
                type="button"
                variant="outline"
                className="bg-white"
                onClick={() => complaintsQuery.refetch()}
                disabled={complaintsQuery.isFetching}
              >
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </div>
          </div>
        </section>

        <AdminComplaintQueue
          complaints={filteredComplaints}
          predictions={predictions}
          noteDrafts={noteDrafts}
          resolutionDrafts={resolutionDrafts}
          assignmentDrafts={assignmentDrafts}
          attachmentPreviews={attachmentPreviews}
          attachmentLoading={attachmentLoading}
          userEmail={user?.email}
          queueToolbar={
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-950">Queue</p>
                  <p className="text-xs leading-5 text-slate-500">Search, narrow the list, then review one case at a time.</p>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="text-slate-600"
                  onClick={() => {
                    setSearchQuery("");
                    setStatusFilter("all");
                    setVisibilityFilter("all");
                    setDepartmentFilter("all");
                  }}
                >
                  Clear
                </Button>
              </div>

              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <Input
                  placeholder="Search title, text, student, or ID"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  className="border-slate-200 bg-slate-50 pl-10 text-slate-900 placeholder:text-slate-500"
                />
              </div>

              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
                <Select value={statusFilter} onValueChange={(value) => setStatusFilter(value as AdminStatusFilter)}>
                  <SelectTrigger className="border-slate-200 bg-slate-50 text-slate-900">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="submitted">Submitted</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="in-progress">In Progress</SelectItem>
                    <SelectItem value="resolved">Resolved</SelectItem>
                    <SelectItem value="rejected">Rejected</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={visibilityFilter} onValueChange={(value) => setVisibilityFilter(value as VisibilityFilter)}>
                  <SelectTrigger className="border-slate-200 bg-slate-50 text-slate-900">
                    <SelectValue placeholder="Visibility" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Complaints</SelectItem>
                    <SelectItem value="anonymous">Anonymous Only</SelectItem>
                    <SelectItem value="identified">Identified Only</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={departmentFilter} onValueChange={setDepartmentFilter}>
                  <SelectTrigger className="border-slate-200 bg-slate-50 text-slate-900 sm:col-span-2 xl:col-span-1">
                    <SelectValue placeholder="Department" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Departments</SelectItem>
                    {departmentOptions.map((department) => (
                      <SelectItem key={department} value={department}>
                        {department}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          }
          onAssignmentDraftChange={(complaintId, value) =>
            setAssignmentDrafts((prev) => ({ ...prev, [complaintId]: value }))
          }
          onNoteDraftChange={(complaintId, value) =>
            setNoteDrafts((prev) => ({ ...prev, [complaintId]: value }))
          }
          onResolutionDraftChange={(complaintId, value) =>
            setResolutionDrafts((prev) => ({ ...prev, [complaintId]: value }))
          }
          onLoadAttachmentPreviews={loadAttachmentPreviews}
          onPredict={(complaintId) => predictMutation.mutate(complaintId)}
          onAutoApply={(complaintId) => autoApplyMutation.mutate(complaintId)}
          onApprove={(complaintId) => approveMutation.mutate(complaintId)}
          onAssignToMe={(complaintId) =>
            workflowMutation.mutate({
              complaintId,
              assigned_to: user?.email ?? null,
            })
          }
          onSaveAssignee={(complaintId, assignee) =>
            workflowMutation.mutate({
              complaintId,
              assigned_to: assignee,
            })
          }
          onSaveNotes={(complaintId, notes) =>
            workflowMutation.mutate({
              complaintId,
              admin_notes: notes,
            })
          }
          onUpdateStatus={(complaintId, status, resolutionSummary) =>
            updateStatusMutation.mutate({
              complaintId,
              status,
              ...(resolutionSummary !== undefined ? { resolution_summary: resolutionSummary } : {}),
            })
          }
          predictPending={predictMutation.isPending}
          autoApplyPending={autoApplyMutation.isPending}
          approvePending={approveMutation.isPending}
          workflowPending={workflowMutation.isPending || updateStatusMutation.isPending}
          renderThread={(complaintId) => <ComplaintThreadCard complaintId={complaintId} />}
          getAttachmentKind={getAttachmentKind}
          getAttachmentMimeType={getAttachmentMimeType}
        />
      </main>
    </div>
  );
};

export default Admin;
