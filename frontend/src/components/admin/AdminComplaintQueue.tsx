import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { ComplaintAnalysisBundle, ComplaintRecord } from "@/integrations/aws/client";
import QueueList from "@/components/admin/QueueList";
import CaseHeader from "@/components/admin/CaseHeader";
import CaseBody from "@/components/admin/CaseBody";
import AIInsightsPanel from "@/components/admin/AIInsightsPanel";
import ActionBar from "@/components/admin/ActionBar";
import NotesPanel from "@/components/admin/NotesPanel";
import {
  complaintNeedsAttention,
  statusWeight,
  toSafeAnalysis,
} from "@/components/admin/adminReviewUtils";

type AdminComplaintQueueProps = {
  complaints: ComplaintRecord[];
  predictions: Record<string, ComplaintAnalysisBundle>;
  noteDrafts: Record<string, string>;
  resolutionDrafts: Record<string, string>;
  assignmentDrafts: Record<string, string>;
  attachmentPreviews: Record<string, Array<{ key: string; url: string }>>;
  attachmentLoading: Record<string, boolean>;
  userEmail?: string | null;
  queueToolbar?: ReactNode;
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

const AdminComplaintQueue = ({
  complaints,
  predictions,
  noteDrafts,
  resolutionDrafts,
  assignmentDrafts,
  attachmentPreviews,
  attachmentLoading,
  userEmail,
  queueToolbar,
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
  const orderedComplaints = useMemo(
    () =>
      [...complaints].sort((a, b) => {
        const attentionA = Number(complaintNeedsAttention(a));
        const attentionB = Number(complaintNeedsAttention(b));
        if (attentionA !== attentionB) return attentionB - attentionA;

        const statusDelta = statusWeight(a.status) - statusWeight(b.status);
        if (statusDelta !== 0) return statusDelta;

        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }),
    [complaints]
  );

  const [selectedComplaintId, setSelectedComplaintId] = useState<string | null>(orderedComplaints[0]?.id ?? null);

  useEffect(() => {
    if (!orderedComplaints.length) {
      setSelectedComplaintId(null);
      return;
    }

    if (!selectedComplaintId || !orderedComplaints.some((item) => item.id === selectedComplaintId)) {
      setSelectedComplaintId(orderedComplaints[0].id);
    }
  }, [orderedComplaints, selectedComplaintId]);

  const selectedComplaint = orderedComplaints.find((item) => item.id === selectedComplaintId) ?? null;
  const analysis = selectedComplaint ? predictions[selectedComplaint.id] ?? toSafeAnalysis(selectedComplaint.analysis) : null;
  const warnings = selectedComplaint
    ? [...(analysis?.submission_guard?.warnings ?? []), ...(analysis?.attachment_checks?.warnings ?? [])]
    : [];

  const selectedAssignmentDraft = selectedComplaint
    ? assignmentDrafts[selectedComplaint.id] ?? selectedComplaint.assigned_to ?? ""
    : "";
  const selectedNoteDraft = selectedComplaint
    ? noteDrafts[selectedComplaint.id] ?? selectedComplaint.admin_notes ?? ""
    : "";
  const selectedResolutionDraft = selectedComplaint
    ? resolutionDrafts[selectedComplaint.id] ?? selectedComplaint.resolution_summary ?? ""
    : "";

  return (
    <section className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
      <aside className="min-h-0">
        <div className="sticky top-20 space-y-4">
          {queueToolbar ? (
            <div className="rounded-[22px] border border-slate-200/80 bg-white/95 p-4 shadow-sm">{queueToolbar}</div>
          ) : null}

          <QueueList
            complaints={orderedComplaints}
            selectedComplaintId={selectedComplaintId}
            onSelect={setSelectedComplaintId}
          />
        </div>
      </aside>

      <div className="min-w-0">
        {!selectedComplaint ? (
          <div className="rounded-[24px] border border-dashed border-slate-300 bg-white/90 p-12 text-center shadow-sm">
            <p className="text-lg font-semibold text-slate-950">Choose a complaint to start reviewing</p>
            <p className="mt-2 text-sm text-slate-600">The focused review panel will appear here once you select a case.</p>
          </div>
        ) : (
          <div className="space-y-4">
            <CaseHeader complaint={selectedComplaint} />
            <CaseBody complaint={selectedComplaint} />
            <AIInsightsPanel complaint={selectedComplaint} analysis={analysis} warnings={warnings} />
            <ActionBar
              complaint={selectedComplaint}
              analysis={analysis}
              assignmentDraft={selectedAssignmentDraft}
              resolutionDraft={selectedResolutionDraft}
              userEmail={userEmail}
              predictPending={predictPending}
              autoApplyPending={autoApplyPending}
              approvePending={approvePending}
              workflowPending={workflowPending}
              onAssignmentDraftChange={(value) => onAssignmentDraftChange(selectedComplaint.id, value)}
              onResolutionDraftChange={(value) => onResolutionDraftChange(selectedComplaint.id, value)}
              onPredict={() => onPredict(selectedComplaint.id)}
              onAutoApply={() => onAutoApply(selectedComplaint.id)}
              onApprove={() => onApprove(selectedComplaint.id)}
              onAssignToMe={() => onAssignToMe(selectedComplaint.id)}
              onSaveAssignee={() => onSaveAssignee(selectedComplaint.id, selectedAssignmentDraft.trim() || null)}
              onUpdateStatus={(status) =>
                onUpdateStatus(
                  selectedComplaint.id,
                  status,
                  status === "resolved" ? selectedResolutionDraft : undefined
                )
              }
            />
            <NotesPanel
              complaint={selectedComplaint}
              noteDraft={selectedNoteDraft}
              attachmentPreviews={attachmentPreviews[selectedComplaint.id] ?? []}
              attachmentLoading={attachmentLoading[selectedComplaint.id] ?? false}
              renderThread={() => renderThread(selectedComplaint.id)}
              getAttachmentKind={getAttachmentKind}
              getAttachmentMimeType={getAttachmentMimeType}
              onLoadAttachmentPreviews={() => onLoadAttachmentPreviews(selectedComplaint)}
              onNoteDraftChange={(value) => onNoteDraftChange(selectedComplaint.id, value)}
              onSaveNotes={() => onSaveNotes(selectedComplaint.id, selectedNoteDraft)}
              workflowPending={workflowPending}
            />
          </div>
        )}
      </div>
    </section>
  );
};

export default AdminComplaintQueue;
