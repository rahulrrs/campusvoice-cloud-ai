import { lazy, Suspense, useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { useOfflineSync } from "@/offline/useOfflineSync";
import {
  loadAdminLoginPage,
  loadAdminPage,
  loadAdminAccessPage,
  loadAuthPage,
  loadComplaintDetailPage,
  loadDashboardPage,
  loadFaqPage,
  loadIndexPage,
  loadNotFoundPage,
  loadNotificationsPage,
  loadSuperAdminPage,
  loadSubmitComplaintPage,
} from "@/lib/routeLoaders";

const Index = lazy(loadIndexPage);
const Dashboard = lazy(loadDashboardPage);
const SubmitComplaint = lazy(loadSubmitComplaintPage);
const ComplaintDetail = lazy(loadComplaintDetailPage);
const Auth = lazy(loadAuthPage);
const AdminLogin = lazy(loadAdminLoginPage);
const Admin = lazy(loadAdminPage);
const AdminAccess = lazy(loadAdminAccessPage);
const SuperAdmin = lazy(loadSuperAdminPage);
const FAQ = lazy(loadFaqPage);
const Notifications = lazy(loadNotificationsPage);
const NotFound = lazy(loadNotFoundPage);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      retry: 1,
      staleTime: 120_000,
      gcTime: 10 * 60_000,
    },
  },
});

function RouteViewport() {
  const location = useLocation();

  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-background">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      }
    >
      <Routes location={location}>
        <Route path="/" element={<Index />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/admin-login" element={<AdminLogin />} />
        <Route path="/admin-access" element={<AdminAccess />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/submit" element={<SubmitComplaint />} />
        <Route path="/complaints/:id" element={<ComplaintDetail />} />
        <Route path="/faq" element={<FAQ />} />
        <Route path="/notifications" element={<Notifications />} />
        <Route path="/admin" element={<Admin />} />
        <Route path="/super-admin" element={<SuperAdmin />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
}

function AppRoutes() {
  const { user } = useAuth();
  const location = useLocation();
  useOfflineSync(user?.id ?? null);

  useEffect(() => {
    const preload = () => {
      void loadIndexPage();
      void loadFaqPage();

      if (!user) {
        void loadAuthPage();
        void loadAdminLoginPage();
        void loadAdminAccessPage();
        return;
      }

      switch (location.pathname) {
        case "/":
          void loadDashboardPage();
          void loadSubmitComplaintPage();
          void loadNotificationsPage();
          break;
        case "/dashboard":
          void loadComplaintDetailPage();
          void loadSubmitComplaintPage();
          void loadNotificationsPage();
          break;
        case "/submit":
          void loadDashboardPage();
          void loadNotificationsPage();
          break;
        case "/notifications":
          void loadDashboardPage();
          void loadComplaintDetailPage();
          break;
        case "/admin":
          void loadSuperAdminPage();
          void loadNotificationsPage();
          void loadDashboardPage();
          break;
        case "/super-admin":
          void loadAdminPage();
          void loadNotificationsPage();
          break;
        default:
          void loadDashboardPage();
          void loadNotificationsPage();
          break;
      }
    };

    if (typeof window === "undefined") return;
    if ("requestIdleCallback" in window) {
      const idleId = window.requestIdleCallback(preload, { timeout: 900 });
      return () => window.cancelIdleCallback(idleId);
    }

    const timeoutId = globalThis.setTimeout(preload, 180);
    return () => globalThis.clearTimeout(timeoutId);
  }, [location.pathname, user]);

  return (
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <RouteViewport />
    </TooltipProvider>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true,
          }}
        >
          <AppRoutes />
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
