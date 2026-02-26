import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { complaintsApi } from "@/integrations/aws/client";
import { useAuth } from "@/contexts/AuthContext";
import {
  savePendingComplaint,
  getPendingComplaints,
  deletePendingComplaint,
  type PendingComplaint,
} from "@/offline/db";

export interface Complaint {
  id: string;
  user_id: string;
  title: string;
  description: string;
  category: string;
  priority: string;
  status: string;
  created_at: string;
  updated_at?: string;
}

export interface CreateComplaintData {
  title: string;
  description: string;
  category: string;
  priority: string;
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
            category: p.data.category,
            priority: p.data.priority,
            status: "pending_sync",
            created_at: new Date(p.createdAt).toISOString(),
          } as Complaint;
        });

      if (!navigator.onLine) return pending;

      const data = await complaintsApi.list();
      const serverComplaints = [...(data as Complaint[])].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      return [...pending, ...serverComplaints];
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
        await savePendingComplaint({
          ...data,
          user_id: user.id,
        });
        return { ok: true };
      }

      return await complaintsApi.create({
        ...data,
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
    const payload = {
      ...p.data,
      user_id: userId,
      status: "pending",
    };

    try {
      await complaintsApi.create(payload);
      await deletePendingComplaint(p.localId);
    } catch {
      // Keep for retry.
    }
  }
}
