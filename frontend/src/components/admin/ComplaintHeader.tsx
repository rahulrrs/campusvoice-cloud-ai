import { EyeOff, UserRound } from "lucide-react";
import StatusBadge from "@/components/complaints/StatusBadge";
import type { ComplaintRecord } from "@/integrations/aws/client";
import {
  formatDateTime,
  nextActionLabel,
  reviewerHint,
  type QueueStatus,
} from "@/components/admin/adminReviewUtils";

type ComplaintHeaderProps = {
  complaint: ComplaintRecord;
};

const ComplaintHeader = ({ complaint }: ComplaintHeaderProps) => {
  return (
    <section className="sticky top-20 z-10 rounded-[28px] border border-slate-200/80 bg-white/95 p-5 shadow-sm backdrop-blur">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="min-w-0 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={complaint.status as QueueStatus} />
            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">
              {complaint.priority ?? "medium"} priority
            </span>
            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">
              {complaint.department ?? complaint.category ?? "No department"}
            </span>
            <span
              className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${
                complaint.is_anonymous ? "bg-amber-50 text-amber-800" : "bg-emerald-50 text-emerald-700"
              }`}
            >
              {complaint.is_anonymous ? <EyeOff className="h-3.5 w-3.5" /> : <UserRound className="h-3.5 w-3.5" />}
              {complaint.is_anonymous ? "Anonymous" : "Identified"}
            </span>
          </div>

          <div className="space-y-2">
            <h2 className="text-2xl font-semibold tracking-tight text-slate-950">{complaint.title}</h2>
            <p className="max-w-3xl text-sm leading-7 text-slate-700">{complaint.description}</p>
          </div>
        </div>

        <div className="grid gap-3 text-sm xl:min-w-[280px]">
          <div className="rounded-2xl border border-slate-200 bg-slate-50/90 px-4 py-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Next best action</p>
            <p className="mt-2 text-base font-semibold text-slate-950">{nextActionLabel(complaint)}</p>
            <p className="mt-1 text-sm leading-6 text-slate-600">{reviewerHint(complaint)}</p>
          </div>

          <div className="grid gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 sm:grid-cols-2 xl:grid-cols-1">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Submitted</p>
              <p className="mt-1 text-sm font-medium text-slate-900">
                {formatDateTime(complaint.submitted_at ?? complaint.created_at)}
              </p>
            </div>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Assigned</p>
              <p className="mt-1 text-sm font-medium text-slate-900">{complaint.assigned_to ?? "No owner yet"}</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ComplaintHeader;
