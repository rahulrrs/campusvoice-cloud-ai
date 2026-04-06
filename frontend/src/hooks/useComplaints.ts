import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { complaintsApi, type ComplaintListFilters } from "@/integrations/aws/client";
import { useAuth } from "@/contexts/AuthContext";
import {
  savePendingComplaint,
  getPendingComplaints,
  deletePendingComplaint,
  type PendingComplaint,
  type QueuedAttachment,
} from "@/offline/db";
import type { ComplaintAnalysisBundle } from "@/integrations/aws/client";

export interface Complaint {
  id: string;
  user_id: string;
  title: string;
  description: string;
  category?: string;
  priority?: string;
  status: string;
  is_anonymous?: boolean;
  attachments?: string[];
  evidence_types?: string[];
  analysis?: ComplaintAnalysisBundle;
  source_language?: string | null;
  submitted_at?: string | null;
  pending_at?: string | null;
  in_progress_at?: string | null;
  resolved_at?: string | null;
  last_student_update_at?: string | null;
  last_public_admin_update_at?: string | null;
  last_user_viewed_updates_at?: string | null;
  last_admin_viewed_updates_at?: string | null;
  has_unread_updates_for_user?: boolean;
  has_unread_updates_for_admin?: boolean;
  resolution_summary?: string | null;
  reopened_at?: string | null;
  reopen_count?: number;
  created_at: string;
  updated_at?: string;
}

export interface CreateComplaintData {
  title: string;
  description: string;
  attachment_keys?: string[];
  queued_attachments?: QueuedAttachment[];
  evidence_types?: string[];
  is_anonymous?: boolean;
  already_queued?: boolean;
}

const normalizeServerStatus = (status: string | undefined) => {
  if (status === "in_progress") return "in-progress";
  return status ?? "submitted";
};

const normalizeOfflineComplaint = (p: PendingComplaint): Complaint => ({
  id: `local-${p.localId}`,
  user_id: p.data.user_id,
  title: p.data.title,
  description: p.data.description,
  category: p.data.category ?? "Uncategorized",
  priority: p.data.priority ?? "medium",
  status: "pending_sync",
  is_anonymous: p.data.is_anonymous ?? true,
  evidence_types: p.data.evidence_types ?? [],
  source_language: p.data.source_language,
  submitted_at: new Date(p.createdAt).toISOString(),
  last_student_update_at: new Date(p.createdAt).toISOString(),
  has_unread_updates_for_user: false,
  has_unread_updates_for_admin: false,
  created_at: new Date(p.createdAt).toISOString(),
});

export const useComplaints = (filters?: ComplaintListFilters) => {
  const { user } = useAuth();

  return useQuery({
    queryKey: ["complaints", user?.id, filters?.status ?? "all", filters?.category ?? "all"],
    queryFn: async () => {
      if (!user) return [];

      const pending = (await getPendingComplaints())
        .filter((p) => p?.data?.user_id === user.id)
        .map(normalizeOfflineComplaint);

      const pendingFiltered = pending.filter((complaint) => {
        const matchesStatus =
          !filters?.status ||
          filters.status === "all" ||
          filters.status === "pending" ||
          filters.status === "pending_sync";
        const matchesCategory =
          !filters?.category || filters.category === "all" || complaint.category === filters.category;
        return matchesStatus && matchesCategory;
      });

      if (!navigator.onLine) return pendingFiltered;

      try {
        const data = await complaintsApi.list(filters);
        const serverComplaints = [...(data as Complaint[])].sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        ).map((complaint) => ({
          ...complaint,
          status: normalizeServerStatus(complaint.status),
        }));
        return [...pendingFiltered, ...serverComplaints];
      } catch {
        // Keep local visibility if API list is temporarily unavailable.
        return pendingFiltered;
      }
    },
    enabled: !!user,
    staleTime: 60_000,
    refetchOnMount: true,
  });
};

export const useComplaintDetail = (complaintId: string | undefined) => {
  const { user } = useAuth();

  return useQuery({
    queryKey: ["complaint-detail", user?.id, complaintId],
    queryFn: async () => {
      if (!complaintId) return null;

      if (complaintId.startsWith("local-")) {
        const localId = complaintId.replace("local-", "");
        const pending = await getPendingComplaints();
        const match = pending.find((item) => item.localId === localId);
        return match ? normalizeOfflineComplaint(match) : null;
      }

      const complaint = await complaintsApi.getComplaint(complaintId);
      return {
        ...complaint,
        status: normalizeServerStatus(complaint.status),
      } as Complaint;
    },
    enabled: !!user && !!complaintId,
    staleTime: 120_000,
    refetchOnMount: true,
  });
};

export const useCreateComplaint = () => {
  const { user } = useAuth();

  return useMutation({
    mutationFn: async (data: CreateComplaintData) => {
      if (!user) throw new Error("User not authenticated");

      if (!navigator.onLine) {
        if (!data.already_queued) {
          await savePendingComplaint({
            ...data,
            user_id: user.id,
          });
        }
        return { ok: true };
      }

      return await complaintsApi.create({
        ...data,
        category: "Uncategorized",
        priority: "medium",
        is_anonymous: data.is_anonymous ?? true,
        attachments: data.attachment_keys ?? [],
        evidence_types: data.evidence_types ?? [],
        user_id: user.id,
        status: "submitted",
      });
    },
  });
};

export async function syncOfflineComplaints(userId: string) {
  if (!navigator.onLine) return;

  const pending: PendingComplaint[] = await getPendingComplaints();
  const mine = pending.filter((p) => p?.data?.user_id === userId);

  for (const p of mine) {
    try {
      const attachmentKeys = Array.isArray(p.data.attachment_keys) ? [...p.data.attachment_keys] : [];
      const queued = Array.isArray(p.data.queued_attachments) ? p.data.queued_attachments : [];

      if (queued.length > 0) {
        for (const item of queued) {
          const contentType = item.type || "application/octet-stream";
          const uploadMeta = await complaintsApi.createUploadUrl({
            fileName: item.name,
            contentType,
            fileSize: item.size,
          });
          const fileBlob =
            item.file instanceof Blob ? item.file : new Blob([item.file], { type: contentType });
          await complaintsApi.uploadToS3(uploadMeta.uploadUrl, fileBlob, contentType);
          attachmentKeys.push(uploadMeta.key);
        }
      }

      const payload = {
        title: p.data.title,
        description: p.data.description,
        category: "Uncategorized",
        priority: "medium",
        is_anonymous: p.data.is_anonymous ?? true,
        attachments: attachmentKeys,
        evidence_types: p.data.evidence_types ?? [],
        user_id: userId,
        status: "submitted",
      };

      await complaintsApi.create(payload);
      await deletePendingComplaint(p.localId);
    } catch (error) {
      console.error("Failed to sync pending complaint", p.localId, error);
      // Keep for retry.
    }
  }
}
