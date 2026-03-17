import { cn } from "@/lib/utils";

export type Status =
  | "pending"
  | "in-progress"
  | "resolved"
  | "rejected"
  | "pending_sync";

interface StatusBadgeProps {
  status: Status;
  className?: string;
}

const statusConfig: Record<Status, { label: string; className: string }> = {
  pending: {
    label: "Pending",
    className: "bg-pending/10 text-pending border-pending/20",
  },
  "in-progress": {
    label: "In Progress",
    className: "bg-primary/10 text-primary border-primary/20",
  },
  resolved: {
    label: "Resolved",
    className: "bg-success/10 text-success border-success/20",
  },
  rejected: {
    label: "Rejected",
    className: "bg-destructive/10 text-destructive border-destructive/20",
  },

  // ⭐ NEW OFFLINE STATUS
  pending_sync: {
    label: "Pending Sync",
    className: "bg-yellow-100 text-yellow-800 border-yellow-300",
  },
};

const StatusBadge = ({ status, className }: StatusBadgeProps) => {
  const config = statusConfig[status];

  return (
    <span
      className={cn(
        "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border",
        config.className,
        className
      )}
    >
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full mr-1.5",
          status === "pending" && "bg-pending",
          status === "in-progress" && "bg-primary",
          status === "resolved" && "bg-success",
          status === "rejected" && "bg-destructive",
          status === "pending_sync" && "bg-yellow-500" // ⭐ new dot color
        )}
      />
      {config.label}
    </span>
  );
};

export default StatusBadge;