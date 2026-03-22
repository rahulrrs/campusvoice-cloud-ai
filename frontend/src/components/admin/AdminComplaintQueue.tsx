import type { ReactNode } from "react";
import { ChevronRight, Inbox } from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ComplaintTimeline from "@/components/complaints/ComplaintTimeline";
import StatusBadge from "@/components/complaints/StatusBadge";
import type { ComplaintAnalysisBundle, ComplaintRecord } from "@/integrations/aws/client";

type QueueStatus = "submitted" | "pending" | "in-progress" | "resolved" | "rejected";

type AdminComplaintQueueProps = {
  complaints: ComplaintRecord[];
  predictions: Record<string, ComplaintAnalysisBundle>;
  noteDrafts: Record<string, string>;
  resolutionDrafts: Record<string, string>;
  assignmentDrafts: Record<string, string>;
  attachmentPreviews: Record<string, Array<{ key: string; url: string }>>;
  attachmentLoading: Record<string, boolean>;
  userEmail?: string | null;
  onAssignmentDraftChange: (complaintId: string, value: string) => void;
  onNoteDraftChange: (complaintId: string, value: string) => void;
  onResolutionDraftChange: (complaintId: string, value: string) => void;
  onLoadAttachmentPreviews: (complaint: ComplaintRecord) => void;
  onPredict: (complaintId: string) => void;
  onAutoApply: (complaintId: string) => void;
  onApprove: (complaintId: string) => void;
  onAssignToMe: (complaintId: string) => void;
  onSaveAssignee: (complaintId: string, assignee: string | null) => void;
  onSaveNotes: (complaintId: string, notes: string) => void;
  onUpdateStatus: (complaintId: string, status: string, resolutionSummary?: string) => void;
  predictPending: boolean;
  autoApplyPending: boolean;
  approvePending: boolean;
  workflowPending: boolean;
  renderThread: (complaintId: string) => ReactNode;
  getAttachmentKind: (key: string) => string;
  getAttachmentMimeType: (key: string) => string | undefined;
};

const formatDateTime = (value?: string | null) => (value ? new Date(value).toLocaleString() : "Not available");

const toSafeAnalysis = (value: unknown): ComplaintAnalysisBundle | null => {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<ComplaintAnalysisBundle>;
  if (!candidate.classification || !candidate.sentiment || !candidate.abuse || !candidate.duplicate_detection) {
    return null;
  }
  return candidate as ComplaintAnalysisBundle;
};

const AdminComplaintQueue = ({
  complaints,
  predictions,
  noteDrafts,
  resolutionDrafts,
  assignmentDrafts,
  attachmentPreviews,
  attachmentLoading,
  userEmail,
  onAssignmentDraftChange,
  onNoteDraftChange,
  onResolutionDraftChange,
  onLoadAttachmentPreviews,
  onPredict,
  onAutoApply,
  onApprove,
  onAssignToMe,
  onSaveAssignee,
  onSaveNotes,
  onUpdateStatus,
  predictPending,
  autoApplyPending,
  approvePending,
  workflowPending,
  renderThread,
  getAttachmentKind,
  getAttachmentMimeType,
}: AdminComplaintQueueProps) => {
  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xl font-semibold text-foreground">
          <Inbox className="h-5 w-5 text-primary" />
          Automated Case Queue
        </div>
        <p className="text-sm text-muted-foreground">
          Open a complaint to inspect the system decision, monitor risk, handle escalations, or step in on exception cases.
        </p>
      </div>
      {complaints.length === 0 ? (
        <div className="rounded-2xl border border-dashed p-8 text-center">
          <p className="text-base font-medium text-foreground">No complaints match these filters.</p>
          <p className="mt-2 text-sm text-muted-foreground">
            Try clearing one or more filters to bring cases back into the queue.
          </p>
        </div>
      ) : (
        <Accordion type="single" collapsible className="space-y-4">
          {complaints.map((complaint) => {
              const savedAnalysis = toSafeAnalysis(complaint.analysis);
              const prediction = predictions[complaint.id];
              const analysisToShow = prediction ?? savedAnalysis;
              const moderationWarnings = analysisToShow?.submission_guard?.warnings ?? [];
              const attachmentWarnings = analysisToShow?.attachment_checks?.warnings ?? [];
              const needsAttention =
                !!analysisToShow &&
                (
                  analysisToShow.abuse.toxicity_score >= 0.35 ||
                  analysisToShow.abuse.spam_score >= 0.35 ||
                  analysisToShow.duplicate_detection.is_duplicate ||
                  moderationWarnings.length > 0
                );

              return (
                <AccordionItem
                  key={complaint.id}
                  value={complaint.id}
                  id={`complaint-${complaint.id}`}
                  className={`overflow-hidden rounded-2xl border bg-card ${
                    needsAttention ? "border-amber-300 shadow-sm" : "shadow-sm"
                  }`}
                >
                  <AccordionTrigger className="px-5 py-5 hover:no-underline">
                    <div className="flex w-full flex-col gap-4 text-left xl:flex-row xl:items-center xl:justify-between">
                      <div className="min-w-0 space-y-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <StatusBadge status={complaint.status as QueueStatus} />
                          <span className="rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground">
                            {complaint.category ?? "Uncategorized"}
                          </span>
                          <span className="rounded-full border bg-background px-3 py-1 text-xs capitalize text-muted-foreground">
                            Priority {complaint.priority ?? "medium"}
                          </span>
                          {complaint.is_anonymous && (
                            <span className="rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground">
                              Anonymous
                            </span>
                          )}
                          {needsAttention && (
                            <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-medium text-amber-800">
                              Needs attention
                            </span>
                          )}
                          {complaint.has_unread_updates_for_admin && (
                            <span className="rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                              New student update
                            </span>
                          )}
                        </div>
                        <div className="min-w-0">
                          <p className="truncate text-xl font-semibold text-foreground">{complaint.title}</p>
                          <p className="mt-1 line-clamp-2 text-sm leading-6 text-muted-foreground">{complaint.description}</p>
                          <p className="mt-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                            <ChevronRight className="h-3.5 w-3.5" />
                            Expand complaint
                          </p>
                        </div>
                      </div>

                      <div className="grid gap-3 sm:grid-cols-3 xl:min-w-[520px]">
                        <div className="rounded-xl border bg-muted/5 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Submitted</p>
                          <p className="mt-1 text-sm font-medium text-foreground">{formatDateTime(complaint.submitted_at ?? complaint.created_at)}</p>
                        </div>
                        <div className="rounded-xl border bg-muted/5 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Department</p>
                          <p className="mt-1 text-sm font-medium text-foreground">{complaint.department ?? "Not assigned"}</p>
                        </div>
                        <div className="rounded-xl border bg-muted/5 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Assignee</p>
                          <p className="mt-1 text-sm font-medium text-foreground">{complaint.assigned_to ?? "Unassigned"}</p>
                        </div>
                      </div>
                    </div>
                  </AccordionTrigger>

                  <AccordionContent className="border-t bg-background px-5 pb-5 pt-5">
                    <div className="flex flex-wrap items-center gap-2 rounded-2xl border bg-background px-4 py-3 text-sm">
                      <span className="font-medium text-foreground">
                        {complaint.has_unread_updates_for_admin ? "New student reply waiting" : "No unread updates"}
                      </span>
                      <span className="text-muted-foreground">&bull;</span>
                      <span className="text-muted-foreground">
                        Automation state:{" "}
                        <span className="font-medium text-foreground">
                          {complaint.decision_state?.replace(/_/g, " ") ?? "submitted"}
                        </span>
                      </span>
                    </div>

                    <Tabs defaultValue="review" className="mt-5 space-y-4">
                      <TabsList className="grid h-auto w-full grid-cols-2 gap-2 rounded-2xl bg-transparent p-0 md:grid-cols-4">
                        <TabsTrigger value="review" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Review</TabsTrigger>
                        <TabsTrigger value="evidence" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Evidence</TabsTrigger>
                        <TabsTrigger value="conversation" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Conversation</TabsTrigger>
                        <TabsTrigger value="actions" className="rounded-xl border bg-background px-4 py-2.5 text-muted-foreground data-[state=active]:border-slate-800 data-[state=active]:bg-slate-950 data-[state=active]:text-white">Admin Actions</TabsTrigger>
                      </TabsList>

                      <TabsContent value="review" className="space-y-4">
                        <ComplaintTimeline
                          status={complaint.status as "submitted" | "pending" | "in-progress" | "resolved" | "pending_sync"}
                          submittedAt={complaint.submitted_at ?? complaint.created_at}
                          pendingAt={complaint.pending_at}
                          inProgressAt={complaint.in_progress_at}
                          resolvedAt={complaint.resolved_at}
                        />

                        {analysisToShow && (
                          <div className="grid gap-3 lg:grid-cols-3">
                            <div className="rounded-xl border bg-background p-4 text-sm">
                              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">AI signals</p>
                              <div className="mt-3 space-y-2">
                                <p>Category: <strong>{analysisToShow.classification.label}</strong></p>
                                <p>Priority: <strong>{analysisToShow.classification.priority}</strong></p>
                                <p>Emotion: <strong>{analysisToShow.sentiment.emotion}</strong></p>
                                <p>Urgency: <strong>{analysisToShow.sentiment.urgency_score.toFixed(2)}</strong></p>
                                <p>Route confidence: <strong>{(complaint.routing_confidence ?? 0).toFixed(2)}</strong></p>
                              </div>
                            </div>

                            <div className="rounded-xl border bg-background p-4 text-sm">
                              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Moderation</p>
                              <div className="mt-3 space-y-2">
                                <p>Toxicity: <strong>{analysisToShow.abuse.toxicity_score.toFixed(2)}</strong></p>
                                <p>Spam: <strong>{analysisToShow.abuse.spam_score.toFixed(2)}</strong></p>
                                <p>Duplicate: <strong>{analysisToShow.duplicate_detection.is_duplicate ? "Yes" : "No"}</strong></p>
                                <p>Risk score: <strong>{(complaint.risk_score ?? 0).toFixed(2)}</strong></p>
                                <p>
                                  Review outcome:{" "}
                                  <strong>{analysisToShow.submission_guard?.allow_submission === false ? "Would be blocked" : "Allowed"}</strong>
                                </p>
                              </div>
                            </div>

                            <div className="rounded-xl border bg-background p-4 text-sm">
                              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Routing summary</p>
                              <div className="mt-3 space-y-2">
                                <p>Decision: <strong>{complaint.decision_state?.replace(/_/g, " ") ?? "submitted"}</strong></p>
                                <p>Escalation: <strong>{complaint.escalation_level ?? "none"}</strong></p>
                                <p>SLA: <strong>{formatDateTime(complaint.sla_due_at)}</strong></p>
                                <p>Human review: <strong>{complaint.requires_human_review ? "Required" : "No"}</strong></p>
                              </div>
                            </div>
                          </div>
                        )}

                        {(complaint.fairness_flags?.length ?? 0) > 0 && (
                          <div className="rounded-xl border border-sky-200 bg-sky-50/60 p-4 text-sm text-sky-950">
                            <p className="font-semibold">Fairness watch</p>
                            <ul className="mt-2 space-y-1">
                              {complaint.fairness_flags?.map((flag) => (
                                <li key={flag}>- {flag.replace(/-/g, " ")}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {complaint.quarantined_reason && (
                          <div className="rounded-xl border border-rose-200 bg-rose-50/60 p-4 text-sm text-rose-900">
                            <p className="font-semibold">Quarantine note</p>
                            <p className="mt-2">{complaint.quarantined_reason}</p>
                          </div>
                        )}

                        {(moderationWarnings.length > 0 || attachmentWarnings.length > 0) && (
                          <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 text-sm text-amber-900">
                            <p className="font-semibold">Warnings</p>
                            <ul className="mt-2 space-y-1">
                              {[...moderationWarnings, ...attachmentWarnings].map((warning) => (
                                <li key={warning}>- {warning}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {complaint.resolution_summary && (
                          <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-4 text-sm">
                            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Resolution Summary</p>
                            <p className="mt-2 whitespace-pre-line text-foreground">{complaint.resolution_summary}</p>
                            <div className="mt-3 flex flex-wrap gap-3 text-xs text-muted-foreground">
                              <span>Resolved: {formatDateTime(complaint.resolved_at)}</span>
                              {complaint.reopen_count ? <span>Reopened {complaint.reopen_count} time(s)</span> : null}
                            </div>
                          </div>
                        )}
                      </TabsContent>

                      <TabsContent value="evidence" className="space-y-4">
                        {Array.isArray(complaint.attachments) && complaint.attachments.length > 0 ? (
                          <div className="rounded-xl border bg-background p-4 space-y-3">
                            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                              <div>
                                <p className="text-sm font-medium">Attachments ({complaint.attachments.length})</p>
                                <p className="text-xs text-muted-foreground">Open evidence files only when you need to review them.</p>
                              </div>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => onLoadAttachmentPreviews(complaint)}
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
                                    <div key={item.key} className="space-y-2 rounded-lg border bg-muted/10 p-3">
                                      <p className="break-all text-xs text-muted-foreground">{item.key}</p>
                                      {kind === "image" ? (
                                        <img src={item.url} alt={item.key} className="max-h-64 rounded-md border object-contain" />
                                      ) : kind === "audio" ? (
                                        <audio controls preload="none" className="w-full">
                                          <source src={item.url} type={getAttachmentMimeType(item.key)} />
                                        </audio>
                                      ) : (
                                        <a href={item.url} target="_blank" rel="noreferrer" className="text-sm text-primary underline">
                                          Open attachment
                                        </a>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            ) : (
                              <p className="text-xs text-muted-foreground">Click "Load Attachments" to view evidence files.</p>
                            )}
                          </div>
                        ) : (
                          <div className="rounded-xl border border-dashed bg-background p-8 text-center text-sm text-muted-foreground">
                            No evidence files were attached to this complaint.
                          </div>
                        )}
                      </TabsContent>

                      <TabsContent value="conversation">
                        {renderThread(complaint.id)}
                      </TabsContent>

                      <TabsContent value="actions" className="space-y-5">
                        <div className="rounded-2xl border bg-background p-5 shadow-sm">
                          <div className="space-y-1">
                            <p className="text-sm font-semibold text-foreground">Case actions</p>
                            <p className="text-sm text-muted-foreground">
                              Move the complaint forward, assign ownership, and keep your working notes here.
                            </p>
                          </div>

                          <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_340px]">
                            <div className="space-y-5">
                              <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_240px]">
                                <div className="space-y-2">
                                  <label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                    Change status
                                  </label>
                                  <Select
                                    value={complaint.status}
                                    onValueChange={(status) =>
                                      onUpdateStatus(
                                        complaint.id,
                                        status,
                                        status === "resolved"
                                          ? resolutionDrafts[complaint.id] ?? complaint.resolution_summary ?? ""
                                          : undefined
                                      )
                                    }
                                  >
                                    <SelectTrigger className="w-full bg-background">
                                      <SelectValue placeholder="Change status" />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="submitted">submitted</SelectItem>
                                      <SelectItem value="pending">pending</SelectItem>
                                      <SelectItem value="in-progress">in-progress</SelectItem>
                                      <SelectItem value="resolved">resolved</SelectItem>
                                      <SelectItem value="rejected">rejected</SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>

                                <div className="space-y-2">
                                  <label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                    Current assignee
                                  </label>
                                  <div className="flex h-10 items-center rounded-xl border bg-muted/10 px-3 text-sm font-medium text-foreground">
                                    {complaint.assigned_to ?? "Unassigned"}
                                  </div>
                                </div>
                              </div>

                              <div className="grid gap-2 md:grid-cols-3">
                                  <Button
                                    type="button"
                                    variant="outline"
                                    onClick={() => onApprove(complaint.id)}
                                    disabled={approvePending || complaint.status !== "submitted"}
                                  >
                                  Move To Active Queue
                                  </Button>
                                <Button
                                  type="button"
                                  variant="outline"
                                  onClick={() => onPredict(complaint.id)}
                                  disabled={predictPending}
                                >
                                  Predict
                                </Button>
                                <Button
                                  type="button"
                                  variant="hero"
                                  onClick={() => onAutoApply(complaint.id)}
                                  disabled={autoApplyPending}
                                >
                                  Auto Apply AI
                                </Button>
                              </div>

                              <div className="space-y-2">
                                <div className="space-y-1">
                                  <p className="text-sm font-semibold text-foreground">Internal notes</p>
                                  <p className="text-xs text-muted-foreground">
                                    Keep short investigation notes, blockers, or handoff details here.
                                  </p>
                                </div>
                                <Textarea
                                  value={noteDrafts[complaint.id] ?? complaint.admin_notes ?? ""}
                                  onChange={(event) => onNoteDraftChange(complaint.id, event.target.value)}
                                  placeholder="Add internal notes for your team."
                                  rows={6}
                                  className="bg-background"
                                />
                                <div className="flex justify-end">
                                  <Button
                                    type="button"
                                    variant="outline"
                                    onClick={() => onSaveNotes(complaint.id, noteDrafts[complaint.id] ?? complaint.admin_notes ?? "")}
                                    disabled={workflowPending}
                                  >
                                    Save Notes
                                  </Button>
                                </div>
                              </div>
                            </div>

                            <div className="space-y-4">
                              <div className="rounded-2xl border bg-muted/5 p-4">
                                <div className="space-y-1">
                                  <p className="text-sm font-semibold text-foreground">Ownership</p>
                                  <p className="text-xs text-muted-foreground">
                                    Take the case yourself or assign it to the team that should handle the routed complaint.
                                  </p>
                                </div>

                                <div className="mt-4 space-y-3">
                                  <Button
                                    type="button"
                                    variant="outline"
                                    className="w-full"
                                    onClick={() => onAssignToMe(complaint.id)}
                                    disabled={workflowPending || !userEmail}
                                  >
                                    Assign to Me
                                  </Button>

                                  <div className="space-y-2">
                                    <label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                      Change assignee
                                    </label>
                                    <Input
                                      value={assignmentDrafts[complaint.id] ?? complaint.assigned_to ?? ""}
                                      onChange={(event) => onAssignmentDraftChange(complaint.id, event.target.value)}
                                      placeholder="Enter assignee email or name"
                                      className="bg-background"
                                    />
                                    <Button
                                      type="button"
                                      variant="outline"
                                      className="w-full"
                                      onClick={() =>
                                        onSaveAssignee(
                                          complaint.id,
                                          (assignmentDrafts[complaint.id] ?? complaint.assigned_to ?? "").trim() || null
                                        )
                                      }
                                      disabled={workflowPending}
                                    >
                                      Save Assignee
                                    </Button>
                                  </div>
                                </div>
                              </div>

                              <div className="rounded-2xl border bg-muted/5 p-4">
                                <div className="space-y-1">
                                  <p className="text-sm font-semibold text-foreground">Closeout summary</p>
                                  <p className="text-xs text-muted-foreground">
                                    Required before resolving. Keep it short and outcome-focused.
                                  </p>
                                </div>

                                <Textarea
                                  value={resolutionDrafts[complaint.id] ?? complaint.resolution_summary ?? ""}
                                  onChange={(event) => onResolutionDraftChange(complaint.id, event.target.value)}
                                  placeholder="Summarize what was fixed, confirmed, or communicated."
                                  rows={5}
                                  className="mt-4 bg-background"
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </AccordionContent>
                </AccordionItem>
              );
          })}
        </Accordion>
      )}
    </section>
  );
};

export default AdminComplaintQueue;
