import { useQuery } from "@tanstack/react-query";
import { complaintsApi } from "@/integrations/aws/client";
import { useAuth } from "@/contexts/AuthContext";

export const useAccessProfile = () => {
  const { user } = useAuth();

  return useQuery({
    queryKey: ["access-profile", user?.id],
    queryFn: complaintsApi.getAccessProfile,
    enabled: !!user,
    staleTime: 60_000,
    refetchOnMount: false,
    retry: 1,
  });
};
