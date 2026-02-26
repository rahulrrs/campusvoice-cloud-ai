import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { syncOfflineComplaints } from "@/hooks/useComplaints";

export function useAutoSyncComplaints() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!user) return;

    const run = async () => {
      await syncOfflineComplaints(user.id);
      queryClient.invalidateQueries({ queryKey: ["complaints", user.id] });
    };

    window.addEventListener("online", run);
    run();

    return () => window.removeEventListener("online", run);
  }, [user?.id, queryClient]);
}