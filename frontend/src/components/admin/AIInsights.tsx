import { ShieldCheck, Sparkles, TriangleAlert } from "lucide-react";
import type { ComplaintAnalysisBundle, ComplaintRecord } from "@/integrations/aws/client";

type AIInsightsProps = {
  complaint: ComplaintRecord;
  analysis: ComplaintAnalysisBundle | null;
  warnings: string[];
};

const AIInsights = ({ complaint, analysis, warnings }: AIInsightsProps) => {
  if (!analysis && warnings.length === 0) {
    return null;
  }

  return (
    <section className="rounded-[22px] border border-slate-200/80 bg-slate-50/80 p-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
        <Sparkles className="h-4 w-4 text-primary" />
        AI Insights
      </div>

      {analysis ? (
        <div className="mt-3 grid gap-3 lg:grid-cols-3">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Suggested route</p>
            <p className="mt-1 text-sm font-semibold text-slate-950">
              {analysis.classification.label} to {analysis.classification.department}
            </p>
            <p className="mt-1 text-sm text-slate-600">Priority {analysis.classification.priority}</p>
          </div>

          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Moderation</p>
            <p className="mt-1 text-sm font-semibold text-slate-950">
              Toxicity {analysis.abuse.toxicity_score.toFixed(2)} and spam {analysis.abuse.spam_score.toFixed(2)}
            </p>
            <p className="mt-1 text-sm text-slate-600">
              {analysis.duplicate_detection.is_duplicate ? "Possible duplicate complaint." : "No duplicate detected."}
            </p>
            <p className="mt-1 text-sm text-slate-600">
              Duplicate score {analysis.duplicate_detection.score.toFixed(2)} via {analysis.duplicate_detection.method}
            </p>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-primary" />
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Review state</p>
            </div>
            <p className="mt-1 text-sm font-semibold text-slate-950">
              {complaint.requires_human_review ? "Needs human review" : "Can be handled directly"}
            </p>
            <p className="mt-1 text-sm text-slate-600">Risk {(complaint.risk_score ?? 0).toFixed(2)}</p>
          </div>
        </div>
      ) : null}

      {warnings.length ? (
        <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50/90 p-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-amber-950">
            <TriangleAlert className="h-4 w-4" />
            Review warnings
          </div>
          <ul className="mt-2 space-y-1 text-sm leading-6 text-amber-950">
            {warnings.map((warning) => (
              <li key={warning}>- {warning}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
};

export default AIInsights;
