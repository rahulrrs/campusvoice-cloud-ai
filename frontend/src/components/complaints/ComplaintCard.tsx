import { Calendar, ChevronRight, Tag } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import StatusBadge from "./StatusBadge";

export type ComplaintStatus =
  | "submitted"
  | "pending"
  | "in_progress"
  | "in-progress"
  | "resolved"
  | "rejected"
  | "pending_sync";

interface ComplaintCardProps {
  id: string;
  title: string;
  description: string;
  category: string;
  status: ComplaintStatus;
  date: string;
  hasUnreadUpdates?: boolean;
  onClick?: () => void;
}

const ComplaintCard = ({
  id,
  title,
  description,
  category,
  status,
  date,
  hasUnreadUpdates = false,
  onClick,
}: ComplaintCardProps) => {
  const displayId = id.startsWith("local-") ? id.replace("local-", "OFF-") : id;

  return (
    <Card
      role="button"
      tabIndex={0}
      className="group cursor-pointer border bg-card shadow-card transition-colors duration-200 hover:border-primary/25 hover:shadow-card"
      onClick={onClick}
      onKeyDown={(event) => {
        if (!onClick) return;
        if (event.key === "Enter" || event.key === " ") onClick();
      }}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 space-y-1">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="font-mono">#{displayId}</span>
              <span>&bull;</span>
              <span className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                {date}
              </span>
            </div>

            <h3 className="line-clamp-1 font-semibold text-foreground transition-colors group-hover:text-primary">
              {title}
            </h3>
          </div>

          <StatusBadge status={status} />
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <p className="mb-4 line-clamp-2 text-sm text-muted-foreground">{description}</p>

        <div className="flex items-center justify-between">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-md bg-secondary px-2.5 py-1 text-xs font-medium text-muted-foreground">
              <Tag className="h-3 w-3" />
              {category}
            </span>
            {hasUnreadUpdates && (
              <span className="inline-flex items-center rounded-full bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">
                New update
              </span>
            )}
          </div>

          <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-primary" />
        </div>
      </CardContent>
    </Card>
  );
};

export default ComplaintCard;
