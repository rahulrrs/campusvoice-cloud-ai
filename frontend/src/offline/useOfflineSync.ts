import { useEffect } from "react";
import { syncOfflineComplaints } from "@/hooks/useComplaints";

async function syncOnce(userId: string) {
  await syncOfflineComplaints(userId);
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
