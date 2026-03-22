import { useEffect, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bell, CheckCheck, ChevronRight, Clock3, ExternalLink, Filter } from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { complaintsApi } from "@/integrations/aws/client";
import { useToast } from "@/hooks/use-toast";
import { useAccessProfile } from "@/hooks/useAccessProfile";

const Notifications = () => {
  const { user } = useAuth();
  const { data: accessProfile } = useAccessProfile();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const isAdmin = Boolean(accessProfile?.is_admin);

  const notificationsQuery = useQuery({
    queryKey: ["notifications", user?.id],
    queryFn: complaintsApi.getNotifications,
    enabled: !!user,
    staleTime: 120_000,
    refetchOnMount: false,
  });

  useEffect(() => {
    if (!user) {
      navigate("/auth");
    }
  }, [navigate, user]);

  const markReadMutation = useMutation({
    mutationFn: complaintsApi.markNotificationsRead,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["notifications"] }),
        queryClient.invalidateQueries({ queryKey: ["complaints"] }),
        queryClient.invalidateQueries({ queryKey: ["admin-complaints"] }),
      ]);
    },
    onError: (error: Error) => {
      toast({
        title: "Could not mark notifications as read",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const allItems = useMemo(
    () => (notificationsQuery.data?.groups ?? []).flatMap((group) => group.items),
    [notificationsQuery.data?.groups]
  );
  const totalGroups = notificationsQuery.data?.groups?.length ?? 0;
  const latestTimestamp = useMemo(() => {
    const values = allItems
      .map((item) => item.timestamp)
      .filter((value): value is string => Boolean(value))
      .map((value) => new Date(value).getTime())
      .filter((value) => !Number.isNaN(value));
    if (values.length === 0) return null;
    return new Date(Math.max(...values)).toLocaleString();
  }, [allItems]);

  const markItemRead = async (complaintId: string) => {
    await markReadMutation.mutateAsync({ complaint_id: complaintId });
    toast({
      title: "Notification cleared",
      description: "This complaint notification was marked as read.",
    });
  };

  const markAllRead = async () => {
    await markReadMutation.mutateAsync({ mark_all: true });
    toast({
      title: "All notifications cleared",
      description: "Your unread notifications were marked as read.",
    });
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1">
        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-10">
            <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
              <div className="max-w-3xl">
                <h1 className="text-3xl font-bold text-foreground">Notifications</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Stay on top of complaint updates, team action items, and recently resolved cases without jumping between screens.
                </p>
                <div className="mt-4 flex flex-wrap gap-2 text-xs">
                  <span className="inline-flex items-center gap-2 rounded-full border bg-background px-3 py-1.5 text-muted-foreground">
                    <Filter className="h-3.5 w-3.5" />
                    {isAdmin ? "Admin queue activity" : "My complaint activity"}
                  </span>
                  {latestTimestamp ? (
                    <span className="inline-flex items-center gap-2 rounded-full border bg-background px-3 py-1.5 text-muted-foreground">
                      <Clock3 className="h-3.5 w-3.5" />
                      Latest activity {latestTimestamp}
                    </span>
                  ) : null}
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border bg-background px-5 py-4 shadow-sm">
                  <p className="text-sm text-muted-foreground">Unread items</p>
                  <p className="mt-1 text-3xl font-semibold text-foreground">{notificationsQuery.data?.total ?? 0}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Across {totalGroups} notification group{totalGroups === 1 ? "" : "s"}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  className="h-auto min-h-[92px] flex-col items-start justify-center gap-2 rounded-2xl px-5 py-4 text-left"
                  onClick={() => void markAllRead()}
                  disabled={markReadMutation.isPending || !notificationsQuery.data?.total}
                >
                  <span className="inline-flex items-center gap-2 text-sm font-semibold text-foreground">
                    <CheckCheck className="h-4 w-4" />
                    Mark All As Read
                  </span>
                  <span className="text-xs text-muted-foreground">
                    Clear all current notification badges in one step.
                  </span>
                </Button>
              </div>
            </div>
          </div>
        </section>

        <section className="container mx-auto px-4 py-10">
          {notificationsQuery.isLoading ? (
            <div className="flex items-center justify-center py-16">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
            </div>
          ) : notificationsQuery.isError ? (
            <Card className="mx-auto max-w-2xl">
              <CardContent className="space-y-4 py-10 text-center">
                <p className="text-lg font-semibold text-foreground">Could not load notifications</p>
                <p className="text-sm text-muted-foreground">{(notificationsQuery.error as Error).message}</p>
                <Button type="button" variant="outline" onClick={() => void notificationsQuery.refetch()}>
                  Try Again
                </Button>
              </CardContent>
            </Card>
          ) : !notificationsQuery.data?.groups?.length ? (
            <Card className="mx-auto max-w-2xl">
              <CardContent className="py-12 text-center">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <Bell className="h-5 w-5" />
                </div>
                <p className="mt-4 text-lg font-semibold text-foreground">You are all caught up</p>
                <p className="mt-2 text-sm text-muted-foreground">
                  New complaint activity will show up here when there is something that needs your attention.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6">
              {notificationsQuery.data.groups.map((group) => (
                <Card key={group.key} className="shadow-card">
                  <CardHeader className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div>
                      <CardTitle className="text-lg">{group.label}</CardTitle>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {group.count} item{group.count === 1 ? "" : "s"} waiting in this section
                      </p>
                    </div>
                    <div className="inline-flex items-center gap-2 rounded-full border bg-background px-3 py-1.5 text-xs font-medium text-muted-foreground">
                      <Bell className="h-3.5 w-3.5 text-primary" />
                      {group.count} unread
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {group.items.map((item, index) => {
                      const target = isAdmin ? `/admin#complaint-${item.complaint_id}` : `/complaints/${item.complaint_id}`;
                      return (
                        <div
                          key={`${group.key}-${item.complaint_id}`}
                          className="rounded-2xl border bg-background p-4 transition hover:border-primary/30 hover:bg-primary/5"
                        >
                          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-primary/10 px-2 font-semibold text-primary">
                                  {index + 1}
                                </span>
                                <span>{item.department ?? "General queue"}</span>
                                {item.timestamp ? (
                                  <>
                                    <span className="text-border">•</span>
                                    <span className="inline-flex items-center gap-1">
                                      <Clock3 className="h-3.5 w-3.5" />
                                      {new Date(item.timestamp).toLocaleString()}
                                    </span>
                                  </>
                                ) : null}
                              </div>

                              <div className="mt-3 flex items-start justify-between gap-3">
                                <div className="min-w-0">
                                  <p className="truncate text-lg font-semibold text-foreground">{item.title}</p>
                                  {item.preview ? (
                                    <p className="mt-1 text-sm leading-6 text-muted-foreground">{item.preview}</p>
                                  ) : (
                                    <p className="mt-1 text-sm text-muted-foreground">
                                      Open this complaint to review the latest activity and next step.
                                    </p>
                                  )}
                                </div>

                                <Link
                                  to={target}
                                  className="hidden rounded-lg p-1 text-muted-foreground transition hover:bg-accent hover:text-foreground sm:block"
                                >
                                  <ExternalLink className="h-4 w-4" />
                                </Link>
                              </div>

                              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                                <span className="rounded-full border px-2.5 py-1 text-muted-foreground">
                                  {item.category}
                                </span>
                                <span className="rounded-full border px-2.5 py-1 text-muted-foreground capitalize">
                                  {item.status.replace(/_/g, " ")}
                                </span>
                                {item.priority ? (
                                  <span className="rounded-full border px-2.5 py-1 text-muted-foreground capitalize">
                                    {item.priority}
                                  </span>
                                ) : null}
                              </div>
                            </div>

                            <div className="flex shrink-0 flex-wrap gap-2 lg:flex-col lg:items-end">
                              <Link to={target}>
                                <Button type="button" size="sm" variant="outline" className="gap-2">
                                  View case
                                  <ChevronRight className="h-4 w-4" />
                                </Button>
                              </Link>
                              <Button
                                type="button"
                                size="sm"
                                variant="ghost"
                                onClick={() => void markItemRead(item.complaint_id)}
                                disabled={markReadMutation.isPending}
                              >
                                Dismiss
                              </Button>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Notifications;
