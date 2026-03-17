import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Sparkles, ShieldAlert, RefreshCw, TrendingUp, AlertTriangle, Copy } from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/contexts/AuthContext";
import {
  complaintsApi,
  type ComplaintAnalysisBundle,
  type ComplaintRecord,
} from "@/integrations/aws/client";
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

const toSafeAnalysis = (value: unknown): ComplaintAnalysisBundle | null => {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<ComplaintAnalysisBundle>;
  if (!candidate.classification || !candidate.sentiment || !candidate.abuse || !candidate.duplicate_detection) {
    return null;
  }
  return candidate as ComplaintAnalysisBundle;
};

const Admin = () => {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [predictions, setPredictions] = useState<Record<string, ComplaintAnalysisBundle>>({});
  const [attachmentPreviews, setAttachmentPreviews] = useState<
    Record<string, Array<{ key: string; url: string }>>
  >({});
  const [attachmentLoading, setAttachmentLoading] = useState<Record<string, boolean>>({});

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
    retry: 1,
  });

  const analyticsQuery = useQuery({
    queryKey: ["admin-analytics"],
    queryFn: complaintsApi.getAdminAnalytics,
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
      queryClient.invalidateQueries({ queryKey: ["admin-complaints"] });
      queryClient.invalidateQueries({ queryKey: ["admin-analytics"] });
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

  const retrainMutation = useMutation({
    mutationFn: complaintsApi.triggerModelRetrain,
    onSuccess: (data) => {
      toast({
        title: "Retrain request submitted",
        description:
          data.status === "already_running"
            ? "Retraining is already in progress."
            : "Model retraining started in background.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Retrain failed",
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

  const getAttachmentKind = (key: string) => {
    const lower = key.toLowerCase();
    if (/\.(png|jpg|jpeg|gif|webp|bmp|svg)$/.test(lower)) return "image";
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
    setAttachmentLoading((prev) => ({ ...prev, [complaint.id]: true }));
    try {
      const urls = await Promise.all(
        keys.map(async (key) => {
          const data = await complaintsApi.createDownloadUrl({ key });
          return { key, url: data.downloadUrl };
        })
      );
      setAttachmentPreviews((prev) => ({ ...prev, [complaint.id]: urls }));
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

  if (loading || complaintsQuery.isPending) {
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

  if (complaintsQuery.isError) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex-1 flex items-center justify-center p-6">
          <Card className="max-w-xl w-full">
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

  const complaints = (complaintsQuery.data ?? []) as ComplaintRecord[];
  const pendingUnknown = complaints.filter(
    (c) =>
      c.status === "pending" &&
      (!c.category || c.category === "Uncategorized" || c.category === "Unknown")
  ).length;
  const analytics = analyticsQuery.data;

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 py-8 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold">Admin Console</h1>
            <p className="text-sm text-muted-foreground">
              Review complaints with multimodal AI analysis, abuse checks, and trend forecasting.
            </p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                complaintsQuery.refetch();
                analyticsQuery.refetch();
              }}
              disabled={complaintsQuery.isFetching || analyticsQuery.isFetching}
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
            <Button
              type="button"
              variant="outline"
              onClick={() => retrainMutation.mutate()}
              disabled={retrainMutation.isPending}
            >
              Retrain Model
            </Button>
          </div>
        </div>

        {analytics && (
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Complaints analyzed</CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{analytics.summary.complaints_analyzed}</CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                  Urgent complaints
                </CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{analytics.summary.urgent_count}</CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Abuse or spam risk</CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{analytics.summary.abusive_or_spam_count}</CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Copy className="h-4 w-4 text-blue-600" />
                  Duplicate complaints
                </CardTitle>
              </CardHeader>
              <CardContent className="text-2xl font-semibold">{analytics.summary.duplicate_count}</CardContent>
            </Card>
          </div>
        )}

        {analytics && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-primary" />
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

        {complaints.map((complaint) => {
          const prediction = predictions[complaint.id];
          const savedAnalysis = toSafeAnalysis(complaint.analysis);
          const activePrediction = prediction?.classification;
          const analysisToShow = prediction ?? savedAnalysis;

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
                {Array.isArray(complaint.evidence_types) && complaint.evidence_types.length > 0 && (
                  <div className="text-sm">
                    Evidence types: <strong>{complaint.evidence_types.join(", ")}</strong>
                  </div>
                )}
                {Array.isArray(complaint.attachments) && complaint.attachments.length > 0 && (
                  <div className="rounded-md border p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">
                        Attachments ({complaint.attachments.length})
                      </p>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => loadAttachmentPreviews(complaint)}
                        disabled={attachmentLoading[complaint.id]}
                      >
                        {attachmentLoading[complaint.id] ? "Loading..." : "Load Attachments"}
                      </Button>
                    </div>
                    {attachmentPreviews[complaint.id]?.length ? (
                      <div className="space-y-3">
                        {attachmentPreviews[complaint.id].map((item) => {
                          const kind = getAttachmentKind(item.key);
                          return (
                            <div key={item.key} className="space-y-2">
                              <p className="text-xs text-muted-foreground break-all">{item.key}</p>
                              {kind === "image" ? (
                                <img
                                  src={item.url}
                                  alt={item.key}
                                  className="max-h-64 rounded-md border object-contain"
                                />
                              ) : kind === "audio" ? (
                                <audio controls preload="none" className="w-full">
                                  <source src={item.url} type={getAttachmentMimeType(item.key)} />
                                </audio>
                              ) : (
                                <a
                                  href={item.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="text-sm text-primary underline"
                                >
                                  Open attachment
                                </a>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        Click "Load Attachments" to view images/voice files.
                      </p>
                    )}
                  </div>
                )}

                {activePrediction && (
                  <div className="rounded-md border p-3 text-sm">
                    <div>
                      AI Category: <strong>{activePrediction.label}</strong> (
                      {(activePrediction.label_confidence * 100).toFixed(1)}%)
                    </div>
                    <div>
                      AI Priority: <strong>{activePrediction.priority}</strong> (
                      {(activePrediction.priority_confidence * 100).toFixed(1)}%)
                    </div>
                    <div>
                      Department: <strong>{activePrediction.department}</strong>
                    </div>
                  </div>
                )}

                {analysisToShow && (
                  <div className="rounded-md border p-3 text-sm bg-muted/20 space-y-1">
                    <p>
                      Urgency score: <strong>{analysisToShow.sentiment.urgency_score.toFixed(2)}</strong>, emotion:{" "}
                      <strong>{analysisToShow.sentiment.emotion}</strong>
                    </p>
                    <p>
                      Toxicity: <strong>{analysisToShow.abuse.toxicity_score.toFixed(2)}</strong>, spam:{" "}
                      <strong>{analysisToShow.abuse.spam_score.toFixed(2)}</strong>
                    </p>
                    <p>
                      Duplicate: <strong>{analysisToShow.duplicate_detection.is_duplicate ? "yes" : "no"}</strong>{" "}
                      ({analysisToShow.duplicate_detection.method})
                    </p>
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
