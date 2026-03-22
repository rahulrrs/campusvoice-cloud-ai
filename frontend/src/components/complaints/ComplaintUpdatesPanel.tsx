import { useState } from "react";
import { MessageSquareText, Send, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ComplaintUpdateRecord } from "@/integrations/aws/client";

interface ComplaintUpdatesPanelProps {
  title?: string;
  updates: ComplaintUpdateRecord[];
  placeholder?: string;
  submitLabel?: string;
  onSubmit: (body: string, isInternal?: boolean) => Promise<void> | void;
  canPostInternal?: boolean;
  isSubmitting?: boolean;
  canSubmit?: boolean;
  templates?: Array<{
    label: string;
    body: string;
    isInternal?: boolean;
  }>;
}

const ComplaintUpdatesPanel = ({
  title = "Updates",
  updates,
  placeholder = "Write an update...",
  submitLabel = "Post Update",
  onSubmit,
  canPostInternal = false,
  isSubmitting = false,
  canSubmit = true,
  templates = [],
}: ComplaintUpdatesPanelProps) => {
  const [body, setBody] = useState("");
  const [isInternal, setIsInternal] = useState(false);

  const handleSubmit = async () => {
    const message = body.trim();
    if (!message) return;
    await onSubmit(message, canPostInternal ? isInternal : undefined);
    setBody("");
    setIsInternal(false);
  };

  return (
    <div className="rounded-2xl border bg-card p-5">
      <div className="mb-4 flex items-center gap-2">
        <MessageSquareText className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      </div>

      <div className="space-y-3">
        {updates.length === 0 ? (
          <div className="rounded-xl border border-dashed p-4 text-sm text-muted-foreground">
            No updates yet.
          </div>
        ) : (
          updates.map((update) => (
            <div key={update.id} className="rounded-xl border bg-background p-4">
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span className="font-medium capitalize text-foreground">{update.author_role}</span>
                {update.is_internal && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-amber-800">
                    <Shield className="h-3 w-3" />
                    Internal
                  </span>
                )}
                <span>{new Date(update.created_at).toLocaleString()}</span>
              </div>
              <p className="mt-2 whitespace-pre-line text-sm leading-6 text-foreground">{update.body}</p>
            </div>
          ))
        )}
      </div>

      <div className="mt-4 space-y-3">
        {templates.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {templates.map((template) => (
              <Button
                key={template.label}
                type="button"
                variant="secondary"
                size="sm"
                disabled={!canSubmit}
                onClick={() => {
                  setBody(template.body);
                  if (canPostInternal) {
                    setIsInternal(Boolean(template.isInternal));
                  }
                }}
              >
                {template.label}
              </Button>
            ))}
          </div>
        )}
        <Textarea
          value={body}
          onChange={(event) => setBody(event.target.value)}
          placeholder={placeholder}
          rows={3}
          disabled={!canSubmit}
        />
        {canPostInternal && (
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <input
                type="checkbox"
                checked={isInternal}
                onChange={(event) => setIsInternal(event.target.checked)}
              />
              Post as internal admin note
            </label>
            <p className="text-xs text-muted-foreground">
              {isInternal
                ? "Internal notes stay visible only to admins."
                : "This will be posted publicly and shown to the student in their complaint thread and notifications."}
            </p>
          </div>
        )}
        <Button type="button" variant="outline" onClick={() => void handleSubmit()} disabled={!canSubmit || isSubmitting || !body.trim()}>
          <Send className="mr-2 h-4 w-4" />
          {submitLabel}
        </Button>
      </div>
    </div>
  );
};

export default ComplaintUpdatesPanel;
