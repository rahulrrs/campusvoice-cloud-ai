import { useRef } from "react";
import { Bot, CheckCheck, PlayCircle, Route, UserPlus2, WandSparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import type { ComplaintAnalysisBundle, ComplaintRecord } from "@/integrations/aws/client";

type ActionPanelProps = {
  complaint: ComplaintRecord;
  analysis: ComplaintAnalysisBundle | null;
  assignmentDraft: string;
  resolutionDraft: string;
  userEmail?: string | null;
  predictPending: boolean;
  autoApplyPending: boolean;
  approvePending: boolean;
  workflowPending: boolean;
  onAssignmentDraftChange: (value: string) => void;
  onResolutionDraftChange: (value: string) => void;
  onPredict: () => void;
  onAutoApply: () => void;
  onApprove: () => void;
  onAssignToMe: () => void;
  onSaveAssignee: () => void;
  onUpdateStatus: (status: string) => void;
};

const ActionPanel = ({
  complaint,
  analysis,
  assignmentDraft,
  resolutionDraft,
  userEmail,
  predictPending,
  autoApplyPending,
  approvePending,
  workflowPending,
  onAssignmentDraftChange,
  onResolutionDraftChange,
  onPredict,
  onAutoApply,
  onApprove,
  onAssignToMe,
  onSaveAssignee,
  onUpdateStatus,
}: ActionPanelProps) => {
  const { toast } = useToast();
  const resolutionRef = useRef<HTMLTextAreaElement>(null);
  const trimmedResolution = resolutionDraft.trim();
  const trimmedAssignment = assignmentDraft.trim();
  const trimmedCurrentAssignee = complaint.assigned_to?.trim() ?? "";
  const isAssignedToCurrentUser =
    !!userEmail && (complaint.assigned_to?.trim().toLowerCase() ?? "") === userEmail.trim().toLowerCase();

  const focusResolutionComposer = () => {
    resolutionRef.current?.focus();
    resolutionRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };

  const requestResolutionBeforeClose = () => {
    focusResolutionComposer();
    toast({
      title: "Add a resolution first",
      description: "Write a short closure summary before resolving this case.",
    });
  };

  const handleStatusChange = (nextStatus: string) => {
    if (nextStatus === complaint.status) return;
    if (nextStatus === "resolved" && !trimmedResolution) {
      requestResolutionBeforeClose();
      return;
    }
    onUpdateStatus(nextStatus);
  };

  const primaryAction = (() => {
    if (complaint.status === "submitted") {
      return {
        label: "Start Handling",
        icon: Route,
        onClick: onApprove,
        disabled: approvePending,
        variant: "hero" as const,
      };
    }
    if (complaint.status === "pending") {
      return {
        label: "Mark In Progress",
        icon: PlayCircle,
        onClick: () => onUpdateStatus("in-progress"),
        disabled: workflowPending,
        variant: "hero" as const,
      };
    }
    if (complaint.status === "in-progress") {
      return {
        label: trimmedResolution ? "Resolve Case" : "Add Resolution to Close",
        icon: CheckCheck,
        onClick: () => {
          if (!trimmedResolution) {
            requestResolutionBeforeClose();
            return;
          }
          onUpdateStatus("resolved");
        },
        disabled: workflowPending,
        variant: trimmedResolution ? ("hero" as const) : ("outline" as const),
      };
    }
    return {
      label: "Case Up To Date",
      icon: CheckCheck,
      onClick: () => undefined,
      disabled: true,
      variant: "outline" as const,
    };
  })();

  const PrimaryIcon = primaryAction.icon;

  return (
    <section className="rounded-[22px] border border-slate-200/80 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-950">Action Panel</p>
          <p className="mt-1 text-sm text-slate-600">Take the next step first. Secondary controls stay below.</p>
        </div>
      </div>

      <div className="mt-4 space-y-4">
        <Button type="button" variant={primaryAction.variant} className="w-full justify-center" onClick={primaryAction.onClick} disabled={primaryAction.disabled}>
          <PrimaryIcon className="mr-2 h-4 w-4" />
          {primaryAction.label}
        </Button>

        {(complaint.status === "in-progress" || complaint.status === "resolved") ? (
          <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Resolution summary</p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {complaint.status === "in-progress"
                    ? "Required before closing this case."
                    : "Keep the published resolution clear and current."}
                </p>
              </div>
              <span
                className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                  trimmedResolution
                    ? "bg-emerald-50 text-emerald-700"
                    : "bg-amber-50 text-amber-700"
                }`}
              >
                {trimmedResolution ? "Ready" : "Required"}
              </span>
            </div>

            <Textarea
              ref={resolutionRef}
              value={resolutionDraft}
              onChange={(event) => onResolutionDraftChange(event.target.value)}
              placeholder="Summarize what was fixed, confirmed, or communicated."
              rows={3}
              className="mt-3 min-h-[104px] bg-white"
            />
          </div>
        ) : null}

        <div className="grid gap-3 sm:grid-cols-[160px_minmax(0,1fr)] sm:items-end">
          <label className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Status</label>
          <Select value={complaint.status} onValueChange={handleStatusChange}>
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Change status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="submitted">Submitted</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="in-progress">In Progress</SelectItem>
              <SelectItem value="resolved" disabled={!trimmedResolution && complaint.status !== "resolved"}>
                Resolved
              </SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          <Button type="button" variant="outline" onClick={onAssignToMe} disabled={workflowPending || !userEmail || isAssignedToCurrentUser}>
            <UserPlus2 className="mr-2 h-4 w-4" />
            {isAssignedToCurrentUser ? "Assigned to You" : "Assign To Me"}
          </Button>
          <Button type="button" variant="outline" onClick={onAutoApply} disabled={autoApplyPending || !analysis}>
            <WandSparkles className="mr-2 h-4 w-4" />
            {autoApplyPending ? "Applying..." : "Apply Suggestion"}
          </Button>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Secondary actions</p>
          <div className="mt-3 flex gap-2">
            <Input
              value={assignmentDraft}
              onChange={(event) => onAssignmentDraftChange(event.target.value)}
              placeholder="Assignee email or name"
              className="bg-white"
            />
            <Button
              type="button"
              variant="outline"
              onClick={onSaveAssignee}
              disabled={workflowPending || trimmedAssignment === trimmedCurrentAssignee}
            >
              {workflowPending ? "Saving..." : "Save Assignee"}
            </Button>
          </div>
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          <Button type="button" variant="ghost" className="justify-start text-slate-600" onClick={onPredict} disabled={predictPending}>
            <Bot className="mr-2 h-4 w-4" />
            {predictPending ? "Refreshing..." : "Refresh AI Review"}
          </Button>
          <p className="flex items-center text-xs leading-5 text-slate-500">
            Use the conversation section below to send a public update or add an internal note.
          </p>
        </div>
      </div>
    </section>
  );
};

export default ActionPanel;
