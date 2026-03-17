import { motion } from "framer-motion";
import { Calendar, Tag, ChevronRight } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import StatusBadge from "./StatusBadge";

export type ComplaintStatus =
  | "pending"
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
  onClick?: () => void;
}

const ComplaintCard = ({
  id,
  title,
  description,
  category,
  status,
  date,
  onClick,
}: ComplaintCardProps) => {
  // Optional: make the displayed id nicer for offline items (local-xxxx)
  const displayId = id.startsWith("local-") ? id.replace("local-", "OFF-") : id;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      transition={{ duration: 0.2 }}
    >
      <Card
        role="button"
        tabIndex={0}
        className="group cursor-pointer border shadow-card hover:shadow-elevated transition-all duration-300 bg-card"
        onClick={onClick}
        onKeyDown={(e) => {
          if (!onClick) return;
          if (e.key === "Enter" || e.key === " ") onClick();
        }}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1 flex-1">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="font-mono">#{displayId}</span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {date}
                </span>
              </div>

              <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors line-clamp-1">
                {title}
              </h3>
            </div>

            {/* ✅ Now supports pending_sync */}
            <StatusBadge status={status} />
          </div>
        </CardHeader>

        <CardContent className="pt-0">
          <p className="text-sm text-muted-foreground line-clamp-2 mb-4">
            {description}
          </p>

          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground bg-secondary px-2.5 py-1 rounded-md">
              <Tag className="h-3 w-3" />
              {category}
            </span>

            <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default ComplaintCard;