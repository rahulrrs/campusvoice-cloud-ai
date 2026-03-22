import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Download, FileText, ShieldOff, Tag } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import ComplaintTimeline from "@/components/complaints/ComplaintTimeline";
import ComplaintUpdatesPanel from "@/components/complaints/ComplaintUpdatesPanel";
import StatusBadge from "@/components/complaints/StatusBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useComplaintDetail } from "@/hooks/useComplaints";
import { complaintsApi } from "@/integrations/aws/client";
import { useToast } from "@/hooks/use-toast";

const ComplaintDetail = () => {
  const { id } = useParams<{ id: string }>();
  const isLocalComplaint = Boolean(id?.startsWith("local-"));
  const [reopenReason, setReopenReason] = useState("");
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const { data: complaint, isLoading } = useComplaintDetail(id);
  const updatesQuery = useQuery({
    queryKey: ["complaint-updates", id],
    queryFn: () => complaintsApi.listComplaintUpdates(id ?? ""),
    enabled: !!id && !isLocalComplaint,
  });
  useEffect(() => {
    if (updatesQuery.data && !isLocalComplaint) {
      queryClient.invalidateQueries({ queryKey: ["complaints"] });
      if (id) {
        queryClient.invalidateQueries({ queryKey: ["complaint-detail", id] });
      }
    }
  }, [updatesQuery.data, isLocalComplaint, queryClient, id]);
  const updateMutation = useMutation({
    mutationFn: (body: string) => complaintsApi.createComplaintUpdate(id ?? "", { body }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["complaint-updates", id] });
      toast({
        title: "Update posted",
        description: "Your message was added to the complaint thread.",
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
  const reopenMutation = useMutation({
    mutationFn: (reason: string) => complaintsApi.reopenComplaint(id ?? "", { reason }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["complaints"] });
      queryClient.invalidateQueries({ queryKey: ["complaint-detail"] });
      queryClient.invalidateQueries({ queryKey: ["complaint-updates", id] });
      setReopenReason("");
      toast({
        title: "Complaint reopened",
        description: "The complaint is back in review and your reason was shared with the team.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Could not reopen complaint",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const attachmentItems = useMemo(() => complaint?.attachments ?? [], [complaint?.attachments]);

  const handleAttachmentOpen = async (key: string) => {
    try {
      const response = await complaintsApi.createDownloadUrl({ key });
      window.open(response.downloadUrl, "_blank", "noopener,noreferrer");
    } catch (error) {
      console.error("Failed to open attachment", error);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  if (!complaint) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex-1 container mx-auto px-4 py-12">
          <Card className="mx-auto max-w-2xl">
            <CardContent className="py-10 text-center">
              <p className="text-lg font-semibold text-foreground">Complaint not found</p>
              <p className="mt-2 text-sm text-muted-foreground">
                This complaint may have been removed or is not available yet.
              </p>
              <Link to="/dashboard" className="mt-6 inline-flex">
                <Button variant="outline">Back to Dashboard</Button>
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

      <main className="flex-1">
        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-8">
            <Link to="/dashboard" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" />
              Back to Dashboard
            </Link>
            <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusBadge status={complaint.status as "submitted" | "pending" | "in-progress" | "resolved" | "pending_sync"} />
                  <span className="inline-flex items-center gap-1 rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground">
                    <Tag className="h-3.5 w-3.5" />
                    {complaint.category ?? "Uncategorized"}
                  </span>
                  {complaint.is_anonymous && (
                    <span className="inline-flex items-center gap-1 rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground">
                      <ShieldOff className="h-3.5 w-3.5" />
                      Anonymous
                    </span>
                  )}
                </div>
                <h1 className="text-3xl font-bold text-foreground">{complaint.title}</h1>
                <p className="text-sm text-muted-foreground">
                  Submitted on {new Date(complaint.submitted_at ?? complaint.created_at).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="container mx-auto px-4 py-8">
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.9fr)] xl:items-start">
            <div className="space-y-6">
              <Card className="shadow-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Complaint Details
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Description</p>
                    <p className="mt-2 whitespace-pre-line text-sm leading-7 text-foreground">
                      {complaint.description}
                    </p>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="rounded-xl border bg-muted/20 p-4">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">Priority</p>
                      <p className="mt-2 text-sm font-semibold capitalize text-foreground">
                        {complaint.priority ?? "Medium"}
                      </p>
                    </div>
                    <div className="rounded-xl border bg-muted/20 p-4">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">Latest Update</p>
                      <p className="mt-2 text-sm font-semibold text-foreground">
                        {new Date(complaint.updated_at ?? complaint.created_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <ComplaintTimeline
                status={complaint.status as "submitted" | "pending" | "in-progress" | "resolved" | "pending_sync"}
                submittedAt={complaint.submitted_at ?? complaint.created_at}
                pendingAt={complaint.pending_at}
                inProgressAt={complaint.in_progress_at}
                resolvedAt={complaint.resolved_at}
              />

              {complaint.resolution_summary && (
                <Card className="shadow-card">
                  <CardHeader>
                    <CardTitle>Resolution Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="whitespace-pre-line text-sm leading-7 text-foreground">
                      {complaint.resolution_summary}
                    </p>
                    <div className="grid gap-3 sm:grid-cols-2 text-sm">
                      <div className="rounded-xl border bg-muted/20 p-4">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Resolved On</p>
                        <p className="mt-2 font-semibold text-foreground">
                          {complaint.resolved_at ? new Date(complaint.resolved_at).toLocaleString() : "Not available"}
                        </p>
                      </div>
                      <div className="rounded-xl border bg-muted/20 p-4">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Reopened Count</p>
                        <p className="mt-2 font-semibold text-foreground">{complaint.reopen_count ?? 0}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {complaint.status === "resolved" && !isLocalComplaint && (
                <Card className="shadow-card">
                  <CardHeader>
                    <CardTitle>Need More Help?</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                      If the issue is still unresolved or has returned, you can reopen this complaint with a short reason.
                    </p>
                    <textarea
                      value={reopenReason}
                      onChange={(event) => setReopenReason(event.target.value)}
                      rows={3}
                      className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      placeholder="Explain why the complaint needs to be reopened."
                    />
                    <Button
                      type="button"
                      variant="outline"
                      disabled={reopenMutation.isPending || !reopenReason.trim()}
                      onClick={() => void reopenMutation.mutateAsync(reopenReason)}
                    >
                      Reopen Complaint
                    </Button>
                  </CardContent>
                </Card>
              )}

              <ComplaintUpdatesPanel
                title="Conversation"
                updates={updatesQuery.data ?? []}
                placeholder={
                  isLocalComplaint
                    ? "This complaint is still pending sync. Conversation will be available after upload."
                    : "Add a follow-up or extra details for the complaint team."
                }
                submitLabel="Post Update"
                onSubmit={async (body) => {
                  if (isLocalComplaint) return;
                  await updateMutation.mutateAsync(body);
                }}
                isSubmitting={updateMutation.isPending}
                canSubmit={!isLocalComplaint}
              />
            </div>

            <div className="space-y-6">
              <Card className="shadow-card">
                <CardHeader>
                  <CardTitle>Attachments</CardTitle>
                </CardHeader>
                <CardContent>
                  {attachmentItems.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No attachments were included with this complaint.</p>
                  ) : (
                    <div className="space-y-3">
                      {attachmentItems.map((key) => (
                        <button
                          key={key}
                          type="button"
                          onClick={() => void handleAttachmentOpen(key)}
                          className="flex w-full items-center justify-between rounded-xl border bg-background px-4 py-3 text-left transition hover:border-primary/40 hover:bg-primary/5"
                        >
                          <span className="truncate text-sm text-foreground">{key.split("/").pop()}</span>
                          <Download className="h-4 w-4 text-muted-foreground" />
                        </button>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default ComplaintDetail;
