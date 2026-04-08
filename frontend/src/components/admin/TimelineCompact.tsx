import { CheckCircle2, Clock3 } from "lucide-react";

type TimelineStep = {
  key: string;
  label: string;
  time?: string | null;
  reached: boolean;
};

type TimelineCompactProps = {
  steps: TimelineStep[];
  formatDateTime: (value?: string | null) => string;
};

const TimelineCompact = ({ steps, formatDateTime }: TimelineCompactProps) => {
  return (
    <div className="space-y-3">
      {steps.map((step) => (
        <div key={step.key} className="flex items-start gap-3">
          <div className="mt-0.5">
            {step.reached ? (
              <CheckCircle2 className="h-4.5 w-4.5 text-emerald-600" />
            ) : (
              <Clock3 className="h-4.5 w-4.5 text-slate-400" />
            )}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-slate-900">{step.label}</p>
            <p className="mt-0.5 text-xs text-slate-500">{formatDateTime(step.time)}</p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default TimelineCompact;
