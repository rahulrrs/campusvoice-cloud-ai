import { useEffect } from "react";
import { getPendingComplaints, deletePendingComplaint } from "./db";
import { complaintsApi } from "@/integrations/aws/client";

async function syncOnce(userId: string) {
  if (!navigator.onLine) return;

  const pending = await getPendingComplaints();

  for (const item of pending) {
    if (item.data?.user_id && item.data.user_id !== userId) {
      continue;
    }

    try {
      await complaintsApi.create(item.data);
      await deletePendingComplaint(item.localId);
    } catch {
      // Keep the item for retry.
    }
  }
}

export function useOfflineSync(userId: string | null) {
  useEffect(() => {
    if (!userId) return;

    const onOnline = () => {
      void syncOnce(userId);
    };

    window.addEventListener("online", onOnline);

    if (navigator.onLine) {
      void syncOnce(userId);
    }

    return () => window.removeEventListener("online", onOnline);
  }, [userId]);
}
