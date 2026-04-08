import { EyeOff, Flag, UserRound } from "lucide-react";
import { cn } from "@/lib/utils";
import StatusBadge from "@/components/complaints/StatusBadge";
import type { ComplaintRecord } from "@/integrations/aws/client";
import {
  complaintNeedsAttention,
  formatRelativeDate,
  submitterName,
  type QueueStatus,
} from "@/components/admin/adminReviewUtils";

type QueueItemProps = {
  complaint: ComplaintRecord;
  selected: boolean;
  onSelect: () => void;
};

const QueueItem = ({ complaint, selected, onSelect }: QueueItemProps) => {
  const attention = complaintNeedsAttention(complaint);

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "w-full rounded-2xl border px-4 py-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
        selected
          ? "border-primary/30 bg-primary/[0.06] shadow-sm"
          : "border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/70"
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={complaint.status as QueueStatus} className="shrink-0" />
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600">
              {complaint.priority ?? "medium"}
            </span>
            <span
              className={cn(
                "rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
                complaint.is_anonymous ? "bg-amber-50 text-amber-800" : "bg-emerald-50 text-emerald-700"
              )}
            >
              {complaint.is_anonymous ? "Anonymous" : "Identified"}
            </span>
            {attention ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-rose-50 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-rose-700">
                <Flag className="h-3 w-3" />
                Needs review
              </span>
            ) : null}
          </div>

          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-950">{complaint.title}</p>
            <p className="mt-1 line-clamp-1 text-sm text-slate-600">{complaint.description}</p>
          </div>
        </div>

        <p className="shrink-0 text-xs font-medium text-slate-500">
          {formatRelativeDate(complaint.submitted_at ?? complaint.created_at)}
        </p>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2 text-xs text-slate-500">
        <span className="font-medium text-slate-600">{complaint.department ?? complaint.category ?? "Unassigned"}</span>
        <span className="inline-flex min-w-0 items-center gap-1">
          {complaint.is_anonymous ? (
            <EyeOff className="h-3.5 w-3.5 shrink-0 text-amber-700" />
          ) : (
            <UserRound className="h-3.5 w-3.5 shrink-0 text-emerald-700" />
          )}
          <span className="truncate">{submitterName(complaint)}</span>
        </span>
      </div>
    </button>
  );
};

export default QueueItem;
