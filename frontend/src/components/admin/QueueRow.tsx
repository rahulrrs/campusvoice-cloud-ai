import { Flag } from "lucide-react";
import { cn } from "@/lib/utils";
import StatusBadge from "@/components/complaints/StatusBadge";
import type { ComplaintRecord } from "@/integrations/aws/client";
import {
  complaintNeedsAttention,
  formatRelativeDate,
  type QueueStatus,
} from "@/components/admin/adminReviewUtils";

type QueueRowProps = {
  complaint: ComplaintRecord;
  selected: boolean;
  onSelect: () => void;
};

const QueueRow = ({ complaint, selected, onSelect }: QueueRowProps) => {
  const attention = complaintNeedsAttention(complaint);

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "w-full rounded-xl border px-3.5 py-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
        selected
          ? "border-primary/30 bg-primary/[0.05] shadow-sm"
          : "border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/80"
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {attention ? <Flag className="h-3.5 w-3.5 shrink-0 text-rose-600" /> : null}
            <p className="truncate text-sm font-semibold text-slate-950">{complaint.title}</p>
          </div>
          <p className="mt-1 line-clamp-1 text-xs leading-5 text-slate-600">{complaint.description}</p>
        </div>
        <p className="shrink-0 text-[11px] font-medium text-slate-500">
          {formatRelativeDate(complaint.submitted_at ?? complaint.created_at)}
        </p>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <StatusBadge status={complaint.status as QueueStatus} className="shrink-0" />
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600">
          {complaint.priority ?? "medium"}
        </span>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600">
          {complaint.department ?? complaint.category ?? "unassigned"}
        </span>
      </div>
    </button>
  );
};

export default QueueRow;
