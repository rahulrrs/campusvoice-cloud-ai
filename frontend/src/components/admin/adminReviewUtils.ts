import type { ComplaintAnalysisBundle, ComplaintRecord } from "@/integrations/aws/client";

export type QueueStatus = "submitted" | "pending" | "in-progress" | "resolved" | "rejected";

export const formatDateTime = (value?: string | null) => {
  if (!value) return "Not available";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "Not available";
  return parsed.toLocaleString();
};

export const formatRelativeDate = (value?: string | null) => {
  if (!value) return "No timestamp";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "No timestamp";

  const diffMs = Date.now() - parsed.getTime();
  const diffMinutes = Math.round(diffMs / 60000);
  if (Math.abs(diffMinutes) < 60) {
    return `${Math.max(diffMinutes, 0)} min ago`;
  }

  const diffHours = Math.round(diffMinutes / 60);
  if (Math.abs(diffHours) < 24) {
    return `${Math.max(diffHours, 0)} hr ago`;
  }

  const diffDays = Math.round(diffHours / 24);
  return `${Math.max(diffDays, 0)} day${Math.abs(diffDays) === 1 ? "" : "s"} ago`;
};

export const toSafeAnalysis = (value: unknown): ComplaintAnalysisBundle | null => {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<ComplaintAnalysisBundle>;
  if (!candidate.classification || !candidate.sentiment || !candidate.abuse || !candidate.duplicate_detection) {
    return null;
  }
  return candidate as ComplaintAnalysisBundle;
};

export const statusWeight = (status?: string | null) => {
  switch (status) {
    case "submitted":
      return 0;
    case "pending":
      return 1;
    case "in-progress":
      return 2;
    case "resolved":
      return 3;
    case "rejected":
      return 4;
    default:
      return 5;
  }
};

export const complaintNeedsAttention = (complaint: ComplaintRecord) =>
  Boolean(complaint.has_unread_updates_for_admin || complaint.requires_human_review || complaint.decision_state === "escalated");

export const submitterName = (complaint: ComplaintRecord) => {
  if (complaint.is_anonymous) return "Anonymous complaint";
  return complaint.student_name || complaint.student_email || "Identified student";
};

export const submitterMeta = (complaint: ComplaintRecord) => {
  if (complaint.is_anonymous) return [];

  return [
    ["Name", complaint.student_name],
    ["Email", complaint.student_email],
    ["Phone", complaint.student_phone],
    ["Department", complaint.student_department],
    ["Student ID", complaint.student_registration_number],
  ].filter((item): item is [string, string] => Boolean(item[1]));
};

export const buildTimeline = (complaint: ComplaintRecord) => [
  {
    key: "submitted",
    label: "Submitted",
    time: complaint.submitted_at ?? complaint.created_at,
    reached: true,
  },
  {
    key: "pending",
    label: "Queued",
    time: complaint.pending_at,
    reached: ["pending", "in-progress", "resolved"].includes(complaint.status),
  },
  {
    key: "in-progress",
    label: "In progress",
    time: complaint.in_progress_at,
    reached: ["in-progress", "resolved"].includes(complaint.status),
  },
  {
    key: "resolved",
    label: "Resolved",
    time: complaint.resolved_at,
    reached: complaint.status === "resolved",
  },
];

export const nextActionLabel = (complaint: ComplaintRecord) => {
  switch (complaint.status) {
    case "submitted":
      return "Review & Route";
    case "pending":
      return "Mark In Progress";
    case "in-progress":
      return "Publish Resolution";
    case "resolved":
      return "Resolution Published";
    case "rejected":
      return "Case Closed";
    default:
      return "Review Case";
  }
};

export const reviewerHint = (complaint: ComplaintRecord) => {
  if (complaint.status === "submitted") {
    return "Confirm the routing decision and move this into the active queue.";
  }
  if (complaint.status === "pending") {
    return "Validate the context, assign ownership, and start active handling.";
  }
  if (complaint.status === "in-progress") {
    return "Send the next update or close the loop with a clear resolution.";
  }
  if (complaint.status === "resolved") {
    return "Review the closure summary and reopen only if something is incomplete.";
  }
  return "This complaint was closed without moving further.";
};
