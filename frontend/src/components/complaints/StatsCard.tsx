import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  variant?: "default" | "primary" | "success" | "warning" | "pending";
  active?: boolean;
  onClick?: () => void;
}

const variantStyles = {
  default: "bg-card border",
  primary: "bg-primary/5 border-primary/20",
  success: "bg-success/5 border-success/20",
  warning: "bg-destructive/5 border-destructive/20",
  pending: "bg-pending/5 border-pending/20",
};

const iconStyles = {
  default: "bg-secondary text-foreground",
  primary: "bg-primary/10 text-primary",
  success: "bg-success/10 text-success",
  warning: "bg-destructive/10 text-destructive",
  pending: "bg-pending/10 text-pending",
};

const StatsCard = ({
  title,
  value,
  icon: Icon,
  trend,
  variant = "default",
  active = false,
  onClick,
}: StatsCardProps) => {
  return (
    <div
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={(event) => {
        if (!onClick) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick();
        }
      }}
      className={cn(
        "rounded-xl border p-6 shadow-card transition-colors duration-200",
        onClick && "cursor-pointer",
        active && "ring-2 ring-primary/30 shadow-elevated",
        variantStyles[variant]
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="mt-2 text-3xl font-bold text-foreground">{value}</p>
          {trend && (
            <p
              className={cn(
                "mt-1 text-xs font-medium",
                trend.isPositive ? "text-success" : "text-destructive"
              )}
            >
              {trend.isPositive ? "↑" : "↓"} {Math.abs(trend.value)}% from last month
            </p>
          )}
        </div>
        <div className={cn("rounded-xl p-3", iconStyles[variant])}>
          <Icon className="h-6 w-6" />
        </div>
      </div>
    </div>
  );
};

export default StatsCard;
