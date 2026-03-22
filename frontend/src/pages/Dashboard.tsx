import { useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ClipboardList,
  Clock,
  Filter,
  Plus,
  Search,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import ComplaintCard from "@/components/complaints/ComplaintCard";
import StatsCard from "@/components/complaints/StatsCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAuth } from "@/contexts/AuthContext";
import { useComplaints, type Complaint } from "@/hooks/useComplaints";
import { useAccessProfile } from "@/hooks/useAccessProfile";

type StatusFilter = "all" | "pending" | "in-progress" | "resolved";

const Dashboard = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [categoryFilter, setCategoryFilter] = useState("all");

  const { user, loading: authLoading } = useAuth();
  const { data: accessProfile } = useAccessProfile();
  const navigate = useNavigate();
  const isAdmin = Boolean(accessProfile?.is_admin);
  const isSuperAdmin = Boolean(accessProfile?.is_super_admin);

  const deferredSearchQuery = useDeferredValue(searchQuery);
  const { data: complaints = [], isLoading } = useComplaints();

  useEffect(() => {
    if (!navigator.onLine) return;
    if (!authLoading && !user) navigate("/auth");
    if (!authLoading && isSuperAdmin) navigate("/super-admin");
    if (!authLoading && isAdmin && !isSuperAdmin) navigate("/admin");
  }, [user, authLoading, isAdmin, isSuperAdmin, navigate]);

  const isOffline = !navigator.onLine;

  const categories = useMemo(() => {
    const unique = new Set<string>();
    complaints.forEach((complaint) => {
      if (complaint.category) {
        unique.add(complaint.category);
      }
    });
    return Array.from(unique).sort((a, b) => a.localeCompare(b));
  }, [complaints]);

  const filteredComplaints = useMemo(() => {
    const query = deferredSearchQuery.trim().toLowerCase();

    return complaints.filter((complaint) => {
      const matchesStatus =
        statusFilter === "all" ||
        (statusFilter === "pending" &&
          (complaint.status === "submitted" || complaint.status === "pending" || complaint.status === "pending_sync")) ||
        (statusFilter === "in-progress" && complaint.status === "in-progress") ||
        (statusFilter === "resolved" && complaint.status === "resolved");
      const matchesCategory =
        categoryFilter === "all" || (complaint.category ?? "Uncategorized") === categoryFilter;
      if (!matchesStatus || !matchesCategory) {
        return false;
      }

      if (!query) {
        return true;
      }

      const title = complaint.title.toLowerCase();
      const description = complaint.description.toLowerCase();
      const category = (complaint.category ?? "").toLowerCase();
      return title.includes(query) || description.includes(query) || category.includes(query);
    });
  }, [categoryFilter, complaints, deferredSearchQuery, statusFilter]);

  const stats = useMemo(() => {
    const total = complaints.length;
    const pending = complaints.filter((item) => item.status === "submitted" || item.status === "pending").length;
    const inProgress = complaints.filter((item) => item.status === "in-progress").length;
    const resolved = complaints.filter((item) => item.status === "resolved").length;
    return { total, pending, inProgress, resolved };
  }, [complaints]);
  const hasActiveFilters = Boolean(searchQuery || categoryFilter !== "all" || statusFilter !== "all");

  if (navigator.onLine && (authLoading || isLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1">
        {isOffline && (
          <div className="border-b bg-yellow-50">
            <div className="container mx-auto px-4 py-3 text-sm text-yellow-800">
              You are offline. New complaints will be stored safely and synced when your connection returns.
            </div>
          </div>
        )}

        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-8">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <h1 className="text-3xl font-bold text-foreground">My Complaints</h1>
                <p className="mt-1 text-muted-foreground">
                  Track progress, open full complaint details, and stay updated on every stage.
                </p>
              </div>

              <Link to="/submit">
                <Button variant="hero" size="lg">
                  <Plus className="mr-2 h-4 w-4" />
                  New Complaint
                </Button>
              </Link>
            </div>
          </div>
        </section>

        <section className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatsCard
              title="Total"
              value={stats.total}
              icon={ClipboardList}
              variant="primary"
              active={statusFilter === "all"}
              onClick={() => setStatusFilter("all")}
            />
            <StatsCard
              title="Pending"
              value={stats.pending}
              icon={Clock}
              variant="pending"
              active={statusFilter === "pending"}
              onClick={() => setStatusFilter("pending")}
            />
            <StatsCard
              title="In Progress"
              value={stats.inProgress}
              icon={AlertCircle}
              variant="primary"
              active={statusFilter === "in-progress"}
              onClick={() => setStatusFilter("in-progress")}
            />
            <StatsCard
              title="Resolved"
              value={stats.resolved}
              icon={CheckCircle2}
              variant="success"
              active={statusFilter === "resolved"}
              onClick={() => setStatusFilter("resolved")}
            />
          </div>
        </section>

        <section className="container mx-auto px-4 pb-8">
          <div className="rounded-2xl border bg-card p-4 shadow-card">
            <div className="flex flex-col gap-4 md:flex-row">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search complaints..."
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  className="pl-10"
                />
              </div>

              <div className="flex gap-3">
                <Select value={statusFilter} onValueChange={(value) => setStatusFilter(value as StatusFilter)}>
                  <SelectTrigger className="w-[170px]">
                    <Filter className="mr-2 h-4 w-4" />
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="in-progress">In Progress</SelectItem>
                    <SelectItem value="resolved">Resolved</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                  <SelectTrigger className="w-[190px]">
                    <SelectValue placeholder="Category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    {categories.map((category) => (
                      <SelectItem key={category} value={category}>
                        {category}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        </section>

        <section className="container mx-auto px-4 pb-12">
          {filteredComplaints.length === 0 ? (
            <div className="rounded-2xl border bg-card px-6 py-16 text-center shadow-card">
              <ClipboardList className="mx-auto mb-4 h-16 w-16 text-muted-foreground/30" />
              <h3 className="text-lg font-semibold text-foreground">No complaints found</h3>
              <p className="mt-2 text-muted-foreground">
                {hasActiveFilters
                  ? "Try adjusting the search or filters."
                  : "You have not submitted any complaints yet."}
              </p>
              {hasActiveFilters ? (
                <Button
                  variant="outline"
                  className="mt-6"
                  onClick={() => {
                    setSearchQuery("");
                    setStatusFilter("all");
                    setCategoryFilter("all");
                  }}
                >
                  Clear Filters
                </Button>
              ) : (
                <Link to="/submit" className="mt-6 inline-flex">
                  <Button variant="hero">Submit Your First Complaint</Button>
                </Link>
              )}
            </div>
          ) : (
            <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
              {filteredComplaints.map((complaint) => (
                <div key={complaint.id}>
                  <ComplaintCard
                    id={complaint.id}
                    title={complaint.title}
                    description={complaint.description}
                    category={complaint.category ?? "Uncategorized"}
                    status={complaint.status as Complaint["status"] & ("submitted" | "pending" | "in-progress" | "resolved" | "pending_sync")}
                    date={new Date(complaint.submitted_at ?? complaint.created_at).toLocaleDateString()}
                    hasUnreadUpdates={Boolean(complaint.has_unread_updates_for_user)}
                    onClick={() => navigate(`/complaints/${complaint.id}`)}
                  />
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Dashboard;
