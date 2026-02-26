import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { complaintsApi } from "@/integrations/aws/client";
import { useAuth } from "@/contexts/AuthContext";
import {
  savePendingComplaint,
  getPendingComplaints,
  deletePendingComplaint,
  type PendingComplaint,
  type QueuedAttachment,
} from "@/offline/db";

export interface Complaint {
  id: string;
  user_id: string;
  title: string;
  description: string;
  category?: string;
  priority?: string;
  status: string;
  attachments?: string[];
  created_at: string;
  updated_at?: string;
}

export interface CreateComplaintData {
  title: string;
  description: string;
  attachment_keys?: string[];
  queued_attachments?: QueuedAttachment[];
  already_queued?: boolean;
}

export const useComplaints = () => {
  const { user } = useAuth();

  return useQuery({
    queryKey: ["complaints", user?.id],
    queryFn: async () => {
      if (!user) return [];

      const pending = (await getPendingComplaints())
        .filter((p) => p?.data?.user_id === user.id)
        .map((p) => {
          return {
            id: `local-${p.localId}`,
            user_id: p.data.user_id,
            title: p.data.title,
            description: p.data.description,
            category: p.data.category ?? "Uncategorized",
            priority: p.data.priority ?? "medium",
            status: "pending_sync",
            created_at: new Date(p.createdAt).toISOString(),
          } as Complaint;
        });

      if (!navigator.onLine) return pending;

      try {
        const data = await complaintsApi.list();
        const serverComplaints = [...(data as Complaint[])].sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );
        return [...pending, ...serverComplaints];
      } catch {
        // Keep local visibility if API list is temporarily unavailable.
        return pending;
      }
    },
    enabled: !!user,
  });
};

export const useCreateComplaint = () => {
  const queryClient = useQueryClient();
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
        attachments: data.attachment_keys ?? [],
        user_id: user.id,
        status: "pending",
      });
    },
    onSuccess: () => {
      if (user) {
        queryClient.invalidateQueries({ queryKey: ["complaints", user.id] });
      } else {
        queryClient.invalidateQueries({ queryKey: ["complaints"] });
      }
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
          });
          const fileBlob =
            item.file instanceof Blob ? item.file : new Blob([item.file], { type: contentType });
          await complaintsApi.uploadToS3(uploadMeta.uploadUrl, fileBlob, contentType);
          attachmentKeys.push(uploadMeta.key);
        }
      }

      const payload = {
        ...p.data,
        category: p.data.category ?? "Uncategorized",
        priority: p.data.priority ?? "medium",
        attachments: attachmentKeys,
        user_id: userId,
        status: "pending",
      };

      await complaintsApi.create(payload);
      await deletePendingComplaint(p.localId);
    } catch (error) {
      console.error("Failed to sync pending complaint", p.localId, error);
      // Keep for retry.
    }
  }
}
