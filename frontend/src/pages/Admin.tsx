import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Copy,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/contexts/AuthContext";
import {
  complaintsApi,
  type ComplaintAuditLogRecord,
  type ComplaintAnalysisBundle,
  type ComplaintRecord,
} from "@/integrations/aws/client";
import { useToast } from "@/hooks/use-toast";
import { useAccessProfile } from "@/hooks/useAccessProfile";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ComplaintUpdatesPanel from "@/components/complaints/ComplaintUpdatesPanel";
import AdminComplaintQueue from "@/components/admin/AdminComplaintQueue";

type ReviewFilter = "all" | "needs_attention" | "duplicates" | "clean" | "human_review" | "escalated" | "quarantined";
type AdminStatusFilter = "all" | "submitted" | "pending" | "in-progress" | "resolved" | "rejected";

const normalizeStatus = (status?: string | null) => {
  if (!status) return "submitted";
  return status === "in_progress" ? "in-progress" : status;
};

const hasEscalationLevel = (value?: string | null) => {
  if (!value) return false;
  const parsed = Number(value);
  if (Number.isFinite(parsed)) {
    return parsed > 0;
  }
  return value.trim().length > 0;
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
  const { data: accessProfile } = useAccessProfile();
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
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>("all");
  const [departmentFilter, setDepartmentFilter] = useState("all");

  const isAdmin = Boolean(accessProfile?.is_admin);
  const shouldLoadAdminData = !!user && isAdmin;

  useEffect(() => {
    if (!loading && !user) {
      navigate("/admin-login");
    }
  }, [loading, navigate, user]);

  const complaintsQuery = useQuery({
    queryKey: ["admin-complaints", statusFilter, reviewFilter, departmentFilter],
    queryFn: () =>
      complaintsApi.listAllForAdmin({
        status: statusFilter,
        review_state: reviewFilter,
        department: departmentFilter,
      }),
    enabled: !!user && isAdmin,
    staleTime: 30_000,
    retry: 1,
  });

  const analyticsQuery = useQuery({
    queryKey: ["admin-analytics"],
    queryFn: complaintsApi.getAdminAnalytics,
    enabled: !!user && isAdmin,
    staleTime: 30_000,
    retry: 1,
  });

  const auditLogQuery = useQuery({
    queryKey: ["admin-audit-log"],
    queryFn: () => complaintsApi.getAdminAuditLog(40),
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
      queryClient.invalidateQueries({ queryKey: ["admin-analytics"] });
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

  const autoClassifyMutation = useMutation({
    mutationFn: () => complaintsApi.autoClassifyAll(true),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      queryClient.invalidateQueries({ queryKey: ["admin-analytics"] });
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
    mutationFn: async ({
      complaintId,
      status,
      resolution_summary,
    }: {
      complaintId: string;
      status: string;
      resolution_summary?: string;
    }) =>
      complaintsApi.updateComplaintByAdmin(complaintId, {
        status,
        resolution_summary,
      }),
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
        title: "Complaint routed",
        description: "The complaint moved from exception review into the active queue.",
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
      department,
    }: {
      complaintId: string;
      assigned_to?: string | null;
      admin_notes?: string | null;
      department?: string;
    }) =>
      complaintsApi.updateComplaintByAdmin(complaintId, {
        assigned_to,
        admin_notes,
        department,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      toast({
        title: "Workflow updated",
        description: "Assignment or internal notes were saved.",
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
    if (keys.length === 0) return;
    if (attachmentPreviews[complaint.id]?.length) return;
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
      const failedCount = results.length - urls.length;

      if (urls.length > 0) {
        setAttachmentPreviews((prev) => ({ ...prev, [complaint.id]: urls }));
      }

      if (failedCount > 0) {
        toast({
          title: urls.length > 0 ? "Some attachments could not load" : "Could not load attachments",
          description:
            urls.length > 0
              ? `${failedCount} attachment preview${failedCount === 1 ? "" : "s"} timed out or failed.`
              : "Attachment previews timed out or failed to load.",
          variant: "destructive",
        });
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
  const pendingUnknown = complaints.filter(
    (complaint) =>
      (complaint.status === "submitted" || complaint.status === "pending") &&
      (!complaint.category || complaint.category === "Uncategorized" || complaint.category === "Unknown")
  ).length;
  const escalatedCount = complaints.filter(
    (complaint) => hasEscalationLevel(complaint.escalation_level) || complaint.decision_state === "escalated"
  ).length;
  const analytics = analyticsQuery.data;

  const filteredComplaints = useMemo(() => {
    const needle = searchQuery.trim().toLowerCase();
    let items = complaints;
    if (reviewFilter === "human_review") {
      items = items.filter((complaint) => complaint.requires_human_review);
    } else if (reviewFilter === "escalated") {
      items = items.filter((complaint) => complaint.decision_state === "escalated");
    } else if (reviewFilter === "quarantined") {
      items = items.filter((complaint) => complaint.decision_state === "quarantined");
    }
    if (!needle) return items;

    return items.filter((complaint) => {
      const haystack = [
        complaint.id,
        complaint.title,
        complaint.description,
        complaint.category,
        complaint.department,
        complaint.assigned_to,
        complaint.decision_state,
      ]
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        .join(" ")
        .toLowerCase();

      return haystack.includes(needle);
    });
  }, [complaints, reviewFilter, searchQuery]);

  const topAuditEvents = (auditLogQuery.data ?? []) as ComplaintAuditLogRecord[];

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Loading admin access</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-3 text-sm text-muted-foreground">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-muted border-t-primary" />
              Checking your account and permissions...
            </CardContent>
          </Card>
        </main>
        <Footer />
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
        <Footer />
      </div>
    );
  }

  if (shouldLoadAdminData && complaintsQuery.isPending) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Loading admin dashboard</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-3 text-sm text-muted-foreground">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-muted border-t-primary" />
              Fetching complaints and analytics...
            </CardContent>
          </Card>
        </main>
        <Footer />
      </div>
    );
  }

  if (complaintsQuery.isError) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex flex-1 items-center justify-center p-6">
          <Card className="w-full max-w-xl">
            <CardHeader>
              <CardTitle>Could not load admin complaints</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>{(complaintsQuery.error as Error)?.message ?? "Unknown error"}</p>
              <Button type="button" variant="outline" onClick={() => complaintsQuery.refetch()}>
                Retry
              </Button>
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
      <main className="container mx-auto flex-1 space-y-5 px-4 py-8">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold">Admin Console</h1>
            <p className="text-sm text-muted-foreground">
              Monitor automated routing, escalation, fairness signals, and handler workload without manually approving complaint validity.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            className="justify-center"
            onClick={() => {
              complaintsQuery.refetch();
              analyticsQuery.refetch();
            }}
            disabled={complaintsQuery.isFetching || analyticsQuery.isFetching}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh Data
          </Button>
        </div>

        <Tabs defaultValue="queue" className="space-y-5">
          <TabsList className="grid h-auto w-full grid-cols-2 gap-2 rounded-2xl bg-transparent p-0 md:grid-cols-5">
            <TabsTrigger value="queue" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Complaint Queue</TabsTrigger>
            <TabsTrigger value="overview" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Overview</TabsTrigger>
            <TabsTrigger value="automation" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Automation</TabsTrigger>
            <TabsTrigger value="workload" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Workload</TabsTrigger>
            <TabsTrigger value="tools" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Tools</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-5">
            {analytics && (
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            <Card className="border-blue-100/80 bg-gradient-to-br from-white via-blue-50/40 to-cyan-50/55">
              <CardHeader>
                <CardTitle className="text-sm">Complaints analyzed</CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{analytics.summary.complaints_analyzed}</CardContent>
            </Card>
            <Card className="border-amber-100/80 bg-gradient-to-br from-white via-amber-50/55 to-orange-50/65">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <AlertTriangle className="h-4 w-4 text-amber-400" />
                  Urgent complaints
                </CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold text-amber-600">{analytics.summary.urgent_count}</CardContent>
            </Card>
            <Card className="border-rose-100/80 bg-gradient-to-br from-white via-rose-50/55 to-pink-50/60">
              <CardHeader>
                <CardTitle className="text-sm">Abuse or spam risk</CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold text-rose-600">{analytics.summary.abusive_or_spam_count}</CardContent>
            </Card>
            <Card className="border-sky-100/80 bg-gradient-to-br from-white via-sky-50/55 to-cyan-50/65">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Copy className="h-4 w-4 text-blue-400" />
                  Duplicate complaints
                </CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold text-blue-600">{analytics.summary.duplicate_count}</CardContent>
            </Card>
              </div>
            )}

            {analytics && (
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                <Card className="border-cyan-100/80 bg-gradient-to-br from-white via-cyan-50/55 to-blue-50/60">
                  <CardHeader><CardTitle className="text-sm">Auto-routed</CardTitle></CardHeader>
                  <CardContent className="text-2xl font-semibold">{analytics.summary.auto_routed_count}</CardContent>
                </Card>
                <Card className="border-amber-100/80 bg-gradient-to-br from-white via-amber-50/55 to-yellow-50/60">
                  <CardHeader><CardTitle className="text-sm">Human review</CardTitle></CardHeader>
                  <CardContent className="text-2xl font-semibold text-amber-600">{analytics.summary.human_review_count}</CardContent>
                </Card>
                <Card className="border-rose-100/80 bg-gradient-to-br from-white via-rose-50/55 to-orange-50/50">
                  <CardHeader><CardTitle className="text-sm">Escalated</CardTitle></CardHeader>
                  <CardContent className="text-2xl font-semibold text-rose-600">{analytics.summary.escalated_count}</CardContent>
                </Card>
                <Card className="border-indigo-100/80 bg-gradient-to-br from-white via-indigo-50/55 to-sky-50/65">
                  <CardHeader><CardTitle className="text-sm">Fairness alerts</CardTitle></CardHeader>
                  <CardContent className="text-2xl font-semibold text-sky-700">{analytics.fairness_summary.alert_count}</CardContent>
                </Card>
              </div>
            )}

            {analytics && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <TrendingUp className="h-4 w-4 text-sky-400" />
                    Predictive Trends (next 7 days)
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <p>
                    Overall trend: <strong>{analytics.trend_forecast.overall.trend}</strong>, projected complaints:{" "}
                    <strong>{analytics.trend_forecast.overall.predicted_next_7_days.toFixed(1)}</strong>
                  </p>
                  {analytics.trend_forecast.top_categories.map((item) => (
                    <p key={item.category}>
                      {item.category}: <strong>{item.trend}</strong> ({item.predicted_next_7_days.toFixed(1)} projected)
                    </p>
                  ))}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="automation" className="space-y-5">
            {analytics && (
              <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Automation Health</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    <p>Average risk score: <strong>{analytics.summary.average_risk_score.toFixed(2)}</strong></p>
                    <p>Quarantined items: <strong>{analytics.summary.quarantined_count}</strong></p>
                    <p>Overdue SLA items: <strong>{analytics.summary.overdue_sla_count}</strong></p>
                    <p>Auto-routed items: <strong>{analytics.summary.auto_routed_count}</strong></p>
                    <p>Human-review exceptions: <strong>{analytics.summary.human_review_count}</strong></p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Bias / Fairness Watch</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    {analytics.fairness_summary.top_flags.length ? (
                      analytics.fairness_summary.top_flags.map((item) => (
                        <div key={item.flag} className="flex items-center justify-between rounded-xl border bg-muted/10 px-3 py-2">
                          <span>{item.flag.replace(/-/g, " ")}</span>
                          <strong>{item.count}</strong>
                        </div>
                      ))
                    ) : (
                      <p className="text-muted-foreground">No fairness flags are active right now.</p>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}

            {analytics && (
              <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Explainability Signals</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    {analytics.explainability_summary.top_rationales.length ? (
                      analytics.explainability_summary.top_rationales.map((item) => (
                        <div key={item.feature} className="flex items-center justify-between rounded-xl border bg-muted/10 px-3 py-2">
                          <span>{item.feature.replace(/_/g, " ")}</span>
                          <strong>{item.count}</strong>
                        </div>
                      ))
                    ) : (
                      <p className="text-muted-foreground">Rationale summaries will appear as new AI-reviewed complaints are analyzed.</p>
                    )}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Fairness Group Monitor</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    {(analytics.fairness_summary.group_breakdown.language ?? []).slice(0, 4).map((item) => (
                      <div key={item.group} className="rounded-xl border bg-muted/10 px-3 py-2">
                        <div className="flex items-center justify-between gap-3">
                          <span>{item.group}</span>
                          <strong>{item.count}</strong>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">Human review cases: {item.human_review_count}</p>
                      </div>
                    ))}
                    {!(analytics.fairness_summary.group_breakdown.language ?? []).length && (
                      <p className="text-muted-foreground">Group-level monitoring will appear once complaint analysis accumulates.</p>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Recent Audit Trail</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {topAuditEvents.length ? (
                  topAuditEvents.map((item) => (
                    <div key={item.id} className="rounded-xl border bg-muted/10 p-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="font-medium">{item.event_type.replace(/_/g, " ")}</div>
                        <div className="text-xs text-muted-foreground">{new Date(item.created_at).toLocaleString()}</div>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Complaint {item.complaint_id} • {item.actor_type}{item.actor_id ? `: ${item.actor_id}` : ""}
                      </p>
                    </div>
                  ))
                ) : (
                  <p className="text-muted-foreground">Audit entries will appear here once the automated flow starts processing cases.</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="workload" className="space-y-5">
            {analytics && (
              <div className="grid gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Department Workload</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {analytics.workload.departments.map((item) => (
                  <div key={item.department} className="rounded-xl border bg-muted/10 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div className="font-medium">{item.department}</div>
                      <div className="text-xs text-muted-foreground">{item.total} total</div>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-xs">
                      <span className="rounded-full bg-primary/10 px-2.5 py-1 text-primary">
                        {item.active} active
                      </span>
                      <span className="rounded-full bg-amber-100 px-2.5 py-1 text-amber-800">
                        {item.urgent} urgent
                      </span>
                      <span className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-700">
                        {item.unassigned} unassigned
                      </span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Assignee Load</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {analytics.workload.assignees.map((item) => (
                  <div key={item.assignee} className="rounded-xl border bg-muted/10 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div className="font-medium">{item.assignee}</div>
                      <div className="text-xs text-muted-foreground">{item.total} total</div>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-xs">
                      <span className="rounded-full bg-primary/10 px-2.5 py-1 text-primary">
                        {item.active} active
                      </span>
                      <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-emerald-800">
                        {item.resolved} resolved
                      </span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
              </div>
            )}
          </TabsContent>

          <TabsContent value="queue" className="space-y-5">
            <Card className="border border-slate-800 bg-slate-950 text-slate-50 shadow-sm">
              <CardContent className="grid gap-4 pt-6 lg:grid-cols-[minmax(0,1.2fr)_320px] lg:items-center">
                <div>
                  <p className="text-sm font-semibold text-slate-900">Classification comes first</p>
                  <p className="mt-1 text-sm leading-6 text-slate-600">
                    Start with complaints that still need a solid category or priority decision, then work through the broader queue.
                  </p>
                </div>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
                  <div className="rounded-2xl border border-blue-100/80 bg-white/85 p-4 shadow-sm">
                    <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">Unclassified pending</p>
                    <p className="mt-2 text-2xl font-semibold text-slate-900">{pendingUnknown}</p>
                  </div>
                  <div className="rounded-2xl border border-cyan-100/80 bg-white/85 p-4 shadow-sm">
                    <p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">Escalated right now</p>
                    <p className="mt-2 text-2xl font-semibold text-slate-900">{escalatedCount}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="overflow-hidden border border-blue-200/70 bg-[linear-gradient(135deg,rgba(255,255,255,0.96),rgba(239,246,255,0.95),rgba(236,254,255,0.92))] text-slate-900 shadow-elevated">
              <CardContent className="pt-6">
                <div className="mb-4 flex flex-col gap-1">
                  <p className="text-sm font-semibold text-slate-100">Find the right complaint fast</p>
                  <p className="text-xs text-slate-400">
                    Narrow the queue by search, status, review state, or department before opening a case.
                  </p>
                </div>
                <div className="grid gap-3 md:grid-cols-[minmax(0,1.2fr)_170px_170px_200px]">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                    <Input
                      placeholder="Search complaints, category, or description"
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      className="border-slate-700 bg-slate-900 pl-10 text-slate-100 placeholder:text-slate-500"
                    />
                  </div>
                  <Select value={statusFilter} onValueChange={(value) => setStatusFilter(value as AdminStatusFilter)}>
                    <SelectTrigger className="border-slate-700 bg-slate-900 text-slate-100">
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
                  <Select value={reviewFilter} onValueChange={(value) => setReviewFilter(value as ReviewFilter)}>
                    <SelectTrigger className="border-slate-700 bg-slate-900 text-slate-100">
                      <SelectValue placeholder="Review" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Review States</SelectItem>
                      <SelectItem value="needs_attention">Needs Attention</SelectItem>
                      <SelectItem value="human_review">Human Review</SelectItem>
                      <SelectItem value="escalated">Escalated</SelectItem>
                      <SelectItem value="quarantined">Quarantined</SelectItem>
                      <SelectItem value="duplicates">Duplicates</SelectItem>
                      <SelectItem value="clean">Clean</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={departmentFilter} onValueChange={setDepartmentFilter}>
                    <SelectTrigger className="border-slate-700 bg-slate-900 text-slate-100">
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
              </CardContent>
            </Card>

            <AdminComplaintQueue
          complaints={filteredComplaints}
          predictions={predictions}
          noteDrafts={noteDrafts}
          resolutionDrafts={resolutionDrafts}
          assignmentDrafts={assignmentDrafts}
          attachmentPreviews={attachmentPreviews}
          attachmentLoading={attachmentLoading}
          userEmail={user?.email}
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
          </TabsContent>

        </Tabs>
      </main>
      <Footer />
    </div>
  );
};

export default Admin;
