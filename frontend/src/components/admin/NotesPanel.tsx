import { EyeOff, MessageSquareText, Paperclip, UserRound } from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ComplaintRecord } from "@/integrations/aws/client";
import { buildTimeline, formatDateTime, submitterMeta, submitterName } from "@/components/admin/adminReviewUtils";
import CompactTimeline from "@/components/admin/CompactTimeline";

type NotesPanelProps = {
  complaint: ComplaintRecord;
  noteDraft: string;
  attachmentPreviews: Array<{ key: string; url: string }>;
  attachmentLoading: boolean;
  renderThread: () => React.ReactNode;
  getAttachmentKind: (key: string) => string;
  getAttachmentMimeType: (key: string) => string | undefined;
  onLoadAttachmentPreviews: () => void;
  onNoteDraftChange: (value: string) => void;
  onSaveNotes: () => void;
  workflowPending: boolean;
};

const NotesPanel = ({
  complaint,
  noteDraft,
  attachmentPreviews,
  attachmentLoading,
  renderThread,
  getAttachmentKind,
  getAttachmentMimeType,
  onLoadAttachmentPreviews,
  onNoteDraftChange,
  onSaveNotes,
  workflowPending,
}: NotesPanelProps) => {
  const hasEvidence = Array.isArray(complaint.attachments) && complaint.attachments.length > 0;

  return (
    <section className="rounded-[22px] border border-slate-200/80 bg-white p-5 shadow-sm">
      <Accordion type="multiple" defaultValue={[]} className="w-full">
        <AccordionItem value="conversation" className="border-b-0">
          <AccordionTrigger className="py-0 text-sm font-semibold text-slate-950 hover:no-underline">
            <span className="flex items-center gap-2">
              <MessageSquareText className="h-4 w-4 text-primary" />
              Conversation
            </span>
          </AccordionTrigger>
          <AccordionContent className="pt-4">
            {renderThread()}
          </AccordionContent>
        </AccordionItem>

        {hasEvidence ? (
          <AccordionItem value="evidence" className="border-b-0 border-t border-slate-100 pt-4">
            <AccordionTrigger className="py-0 text-sm font-semibold text-slate-950 hover:no-underline">
              <span className="flex items-center gap-2">
                <Paperclip className="h-4 w-4 text-primary" />
                Evidence
              </span>
            </AccordionTrigger>
            <AccordionContent className="pt-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-sm text-slate-600">
                  {complaint.attachments?.length ?? 0} attachment(s) linked to this complaint.
                </p>
                <Button type="button" variant="outline" size="sm" onClick={onLoadAttachmentPreviews} disabled={attachmentLoading}>
                  {attachmentLoading ? "Loading..." : "Load evidence"}
                </Button>
              </div>

              {attachmentPreviews.length ? (
                <div className="mt-4 grid gap-3">
                  {attachmentPreviews.map((item) => {
                    const kind = getAttachmentKind(item.key);
                    return (
                      <div key={item.key} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <p className="break-all text-xs text-slate-500">{item.key}</p>
                        <div className="mt-3">
                          {kind === "image" ? (
                            <img src={item.url} alt={item.key} className="max-h-80 rounded-xl border object-contain" />
                          ) : kind === "audio" ? (
                            <audio controls preload="none" className="w-full">
                              <source src={item.url} type={getAttachmentMimeType(item.key)} />
                            </audio>
                          ) : (
                            <a href={item.url} target="_blank" rel="noreferrer" className="text-sm font-medium text-primary underline">
                              Open attachment
                            </a>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </AccordionContent>
          </AccordionItem>
        ) : null}

        <AccordionItem value="timeline" className="border-b-0 border-t border-slate-100 pt-4">
          <AccordionTrigger className="py-0 text-sm font-semibold text-slate-950 hover:no-underline">
            Status Tracker
          </AccordionTrigger>
          <AccordionContent className="pt-4">
            <CompactTimeline steps={buildTimeline(complaint)} formatDateTime={formatDateTime} />
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="reporter" className="border-b-0 border-t border-slate-100 pt-4">
          <AccordionTrigger className="py-0 text-sm font-semibold text-slate-950 hover:no-underline">
            Reporter
          </AccordionTrigger>
          <AccordionContent className="pt-4">
            <div className="flex items-center gap-2 text-sm">
              {complaint.is_anonymous ? (
                <EyeOff className="h-4 w-4 text-amber-700" />
              ) : (
                <UserRound className="h-4 w-4 text-emerald-700" />
              )}
              <span className="font-medium text-slate-950">{submitterName(complaint)}</span>
            </div>

            {complaint.is_anonymous ? (
              <p className="mt-2 text-sm leading-6 text-slate-600">Identity is hidden because anonymous mode is on.</p>
            ) : submitterMeta(complaint).length ? (
              <div className="mt-3 space-y-2">
                {submitterMeta(complaint).map(([label, value]) => (
                  <div key={label} className="flex items-start justify-between gap-3 rounded-2xl border border-slate-200 bg-slate-50/80 px-3 py-2">
                    <span className="text-sm text-slate-500">{label}</span>
                    <span className="max-w-[58%] break-all text-right text-sm font-medium text-slate-900">{value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-sm leading-6 text-slate-600">No extra student profile fields were attached.</p>
            )}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="notes" className="border-b-0 border-t border-slate-100 pt-4">
          <AccordionTrigger className="py-0 text-sm font-semibold text-slate-950 hover:no-underline">
            Internal Notes
          </AccordionTrigger>
          <AccordionContent className="pt-4">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Internal note</p>
              <Textarea
                value={noteDraft}
                onChange={(event) => onNoteDraftChange(event.target.value)}
                placeholder="Keep short operational notes for the admin team."
                rows={4}
                className="mt-2 bg-white"
              />
              <div className="mt-3 flex justify-end">
                <Button type="button" variant="outline" onClick={onSaveNotes} disabled={workflowPending}>
                  Save Note
                </Button>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </section>
  );
};

export default NotesPanel;
