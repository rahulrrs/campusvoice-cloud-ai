// src/pages/Dashboard.tsx
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ClipboardList,
  Clock,
  CheckCircle2,
  AlertCircle,
  Search,
  Filter,
  Plus,
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

import { categories } from "@/data/mockComplaints";
import { useAuth } from "@/contexts/AuthContext";
import { useComplaints, type Complaint } from "@/hooks/useComplaints";

type StatusFilter = "all" | "pending" | "in-progress" | "resolved" | "rejected" | "pending_sync";

const Dashboard = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  const { user, loading: authLoading } = useAuth();
  const { data: complaints = [], isLoading } = useComplaints();
  const navigate = useNavigate();
  const adminEmails = (import.meta.env.VITE_ADMIN_EMAILS ?? "")
    .split(",")
    .map((v: string) => v.trim().toLowerCase())
    .filter((v: string) => v.length > 0);
  const isAdmin = !!user?.email && adminEmails.includes(user.email.toLowerCase());

  // ✅ Auto-sync queued complaints when internet returns
  // ✅ Online: enforce login. Offline: allow viewing (session may exist locally)
  useEffect(() => {
    if (!navigator.onLine) return;
    if (!authLoading && !user) navigate("/auth");
    if (!authLoading && isAdmin) navigate("/admin");
  }, [user, authLoading, isAdmin, navigate]);

  const isOffline = !navigator.onLine;

  const filteredComplaints = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();

    return (complaints as Complaint[]).filter((complaint) => {
      const title = (complaint.title || "").toLowerCase();
      const desc = (complaint.description || "").toLowerCase();

      const matchesSearch = !q || title.includes(q) || desc.includes(q);
      const matchesStatus =
        statusFilter === "all" ? true : complaint.status === statusFilter;
      const matchesCategory =
        categoryFilter === "all" ? true : complaint.category === categoryFilter;

      return matchesSearch && matchesStatus && matchesCategory;
    });
  }, [complaints, searchQuery, statusFilter, categoryFilter]);

  const stats = useMemo(() => {
    const list = complaints as Complaint[];

    const total = list.length;
    const pending = list.filter((c) => c.status === "pending").length;
    const inProgress = list.filter((c) => c.status === "in-progress").length;
    const resolved = list.filter((c) => c.status === "resolved").length;
    const pendingSync = list.filter((c) => c.status === "pending_sync").length;

    return { total, pending, inProgress, resolved, pendingSync };
  }, [complaints]);

  if (navigator.onLine && (authLoading || isLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1">
        {/* Offline banner */}
        {isOffline && (
          <div className="border-b bg-yellow-50">
            <div className="container mx-auto px-4 py-3 text-sm text-yellow-800">
              You are offline. You can still create complaints — they will be saved locally and
              automatically submitted when the internet returns.
            </div>
          </div>
        )}

        {/* Header Section */}
        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-8">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <h1 className="text-3xl font-bold text-foreground">My Complaints</h1>
                <p className="text-muted-foreground mt-1">
                  Track and manage all your submitted complaints
                </p>
                {stats.pendingSync > 0 && (
                  <p className="text-sm text-yellow-700 mt-2">
                    Pending sync: <span className="font-semibold">{stats.pendingSync}</span>
                  </p>
                )}
              </div>

              <Link to="/submit">
                <Button variant="hero" size="lg">
                  <Plus className="h-4 w-4 mr-2" />
                  New Complaint
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* Stats Grid */}
        <section className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatsCard
              title="Total Complaints"
              value={stats.total}
              icon={ClipboardList}
              variant="primary"
            />
            <StatsCard title="Pending" value={stats.pending} icon={Clock} variant="pending" />
            <StatsCard
              title="In Progress"
              value={stats.inProgress}
              icon={AlertCircle}
              variant="primary"
            />
            <StatsCard
              title="Resolved"
              value={stats.resolved}
              icon={CheckCircle2}
              variant="success"
            />
          </div>
        </section>

        {/* Filters and Search */}
        <section className="container mx-auto px-4 pb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card rounded-xl border shadow-card p-4"
          >
            <div className="flex flex-col md:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search complaints..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>

              <div className="flex gap-3">
                <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as StatusFilter)}>
                  <SelectTrigger className="w-[160px]">
                    <Filter className="h-4 w-4 mr-2" />
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="in-progress">In Progress</SelectItem>
                    <SelectItem value="resolved">Resolved</SelectItem>
                    <SelectItem value="rejected">Rejected</SelectItem>
                    <SelectItem value="pending_sync">Pending Sync</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                  <SelectTrigger className="w-[160px]">
                    <SelectValue placeholder="Category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    {categories.map((cat) => (
                      <SelectItem key={cat} value={cat}>
                        {cat}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Complaints List */}
        <section className="container mx-auto px-4 pb-12">
          {filteredComplaints.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-16"
            >
              <ClipboardList className="h-16 w-16 text-muted-foreground/30 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-foreground mb-2">No complaints found</h3>
              <p className="text-muted-foreground mb-6">
                {searchQuery || statusFilter !== "all" || categoryFilter !== "all"
                  ? "Try adjusting your filters"
                  : "You haven't submitted any complaints yet"}
              </p>
              <Link to="/submit">
                <Button variant="hero">Submit Your First Complaint</Button>
              </Link>
            </motion.div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredComplaints.map((complaint, index) => (
                <motion.div
                  key={complaint.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <ComplaintCard
                    id={complaint.id}
                    title={complaint.title}
                    description={complaint.description}
                    category={complaint.category ?? "Uncategorized"}
                    status={
                      complaint.status as
                        | "pending"
                        | "in-progress"
                        | "resolved"
                        | "rejected"
                        | "pending_sync"
                    }
                    date={new Date(complaint.created_at).toLocaleDateString()}
                  />
                </motion.div>
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
