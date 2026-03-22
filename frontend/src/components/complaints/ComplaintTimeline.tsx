import { CheckCircle2, Circle, Clock3 } from "lucide-react";
import { cn } from "@/lib/utils";

type TimelineStatus = "submitted" | "pending" | "in-progress" | "resolved" | "pending_sync";

interface ComplaintTimelineProps {
  status: TimelineStatus;
  submittedAt?: string | null;
  pendingAt?: string | null;
  inProgressAt?: string | null;
  resolvedAt?: string | null;
}

const timelineSteps = [
  { key: "submitted", label: "Submitted" },
  { key: "pending", label: "Pending" },
  { key: "in-progress", label: "In Progress" },
  { key: "resolved", label: "Resolved" },
] as const;

const formatTimestamp = (value?: string | null) => {
  if (!value) return "Not reached yet";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Not available";
  return date.toLocaleString();
};

const statusIndexMap: Record<TimelineStatus, number> = {
  submitted: 0,
  pending: 1,
  "in-progress": 2,
  resolved: 3,
  pending_sync: 0,
};

const ComplaintTimeline = ({
  status,
  submittedAt,
  pendingAt,
  inProgressAt,
  resolvedAt,
}: ComplaintTimelineProps) => {
  const currentIndex = statusIndexMap[status] ?? 0;
  const timestamps = [submittedAt, pendingAt, inProgressAt, resolvedAt];

  return (
    <div className="rounded-2xl border bg-card p-5">
      <div className="mb-4 flex items-center gap-2">
        <Clock3 className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-semibold text-foreground">Tracking Timeline</h3>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {timelineSteps.map((step, index) => {
          const complete = index < currentIndex || (status === "resolved" && index <= currentIndex);
          const active = index === currentIndex;

          return (
            <div key={step.key} className="relative flex gap-3 rounded-xl border bg-background p-4">
              <div className="mt-0.5">
                {complete ? (
                  <CheckCircle2 className="h-5 w-5 text-success" />
                ) : active ? (
                  <Clock3 className="h-5 w-5 text-primary" />
                ) : (
                  <Circle className="h-5 w-5 text-muted-foreground/60" />
                )}
              </div>
              <div className="min-w-0">
                <p
                  className={cn(
                    "text-sm font-semibold",
                    complete && "text-success",
                    active && "text-primary",
                    !complete && !active && "text-foreground"
                  )}
                >
                  {step.label}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {formatTimestamp(timestamps[index])}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ComplaintTimeline;
