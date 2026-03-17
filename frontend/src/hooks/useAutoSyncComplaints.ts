import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { syncOfflineComplaints } from "@/hooks/useComplaints";

export function useAutoSyncComplaints() {
  const { user } = useAuth();
  const userId = user?.id;
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!userId) return;

    const run = async () => {
      await syncOfflineComplaints(userId);
      queryClient.invalidateQueries({ queryKey: ["complaints", userId] });
    };

    window.addEventListener("online", run);
    run();

    return () => window.removeEventListener("online", run);
  }, [queryClient, userId]);
}
