import { useState } from "react";
import { motion } from "framer-motion";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Bell, LayoutDashboard, LogOut, Menu, MessageSquare, Plus, ShieldCheck, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { useAccessProfile } from "@/hooks/useAccessProfile";
import HeaderAssistant from "@/components/layout/HeaderAssistant";
import { complaintsApi } from "@/integrations/aws/client";
import { preloadRouteForPath } from "@/lib/routeLoaders";
import { cn } from "@/lib/utils";

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, signOut } = useAuth();
  const { data: accessProfile } = useAccessProfile();
  const { toast } = useToast();
  const isAdmin = Boolean(accessProfile?.is_admin);

  const notificationsQuery = useQuery({
    queryKey: ["notifications", user?.id],
    queryFn: complaintsApi.getNotifications,
    enabled: !!user,
    staleTime: 120_000,
    refetchOnMount: false,
  });
  const unreadCount = notificationsQuery.data?.total ?? 0;

  const navItems = isAdmin
    ? [
        { path: "/", label: "Home", icon: null },
        { path: "/notifications", label: "Notifications", icon: Bell, count: unreadCount },
        { path: "/admin", label: "Admin", icon: ShieldCheck },
      ]
    : [
        { path: "/", label: "Home", icon: null },
        { path: "/dashboard", label: "My Complaints", icon: LayoutDashboard },
        { path: "/notifications", label: "Notifications", icon: Bell, count: unreadCount },
        { path: "/submit", label: "Submit Complaint", icon: Plus },
      ];

  const isActive = (path: string) => location.pathname === path;

  const handleSignOut = async () => {
    await signOut();
    toast({
      title: "Signed out",
      description: "You have been signed out successfully.",
    });
    navigate("/");
  };

  return (
    <header className="sticky top-0 z-50 w-full px-3 pt-3 md:px-4">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between rounded-[24px] border border-border/70 bg-background/95 px-4 shadow-sm">
        <Link to="/" className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl gradient-primary shadow-sm">
            <MessageSquare className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <span className="heading-display text-xl font-bold text-foreground">CampusVoice</span>
            <p className="hidden text-xs text-muted-foreground md:block">Student grievance desk</p>
          </div>
        </Link>

        <nav className="hidden items-center gap-2 md:flex">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onMouseEnter={() => preloadRouteForPath(item.path)}
              onFocus={() => preloadRouteForPath(item.path)}
              className={cn(
                "relative rounded-full px-4 py-2 text-sm font-medium transition-colors",
                isActive(item.path)
                  ? "gradient-primary text-white shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {isActive(item.path) && (
                <motion.span
                  layoutId="header-active-tab"
                  className="absolute inset-0 rounded-full gradient-primary shadow-sm"
                  transition={{ type: "spring", stiffness: 520, damping: 38, mass: 0.55 }}
                />
              )}
              <span className="relative z-10">{item.label}</span>
              {item.count ? (
                <span className="relative z-10 ml-2 inline-flex min-w-5 items-center justify-center rounded-full bg-primary px-1.5 py-0.5 text-[11px] font-semibold text-primary-foreground">
                  {item.count > 99 ? "99+" : item.count}
                </span>
              ) : null}
            </Link>
          ))}
        </nav>

        <div className="hidden items-center gap-3 md:flex">
          <HeaderAssistant />
          {user ? (
            <>
              <span className="max-w-[220px] truncate text-sm text-muted-foreground">{user.email}</span>
              <Button variant="outline" size="sm" className="rounded-full" onClick={handleSignOut}>
                <LogOut className="mr-2 h-4 w-4" />
                Sign Out
              </Button>
            </>
          ) : (
            <>
              <Link to="/auth">
                <Button variant="ghost" size="sm" className="rounded-full">
                  Sign In
                </Button>
              </Link>
              <Link to="/admin-login">
                <Button variant="outline" size="sm" className="rounded-full">
                  Admin Login
                </Button>
              </Link>
              <Link to="/auth">
                <Button variant="hero" size="sm" className="rounded-full">
                  Get Started
                </Button>
              </Link>
            </>
          )}
        </div>

        <button
          className="rounded-full p-2 hover:bg-accent md:hidden"
          onClick={() => setIsMenuOpen((open) => !open)}
        >
          {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {isMenuOpen && (
        <div className="mx-auto mt-2 max-w-7xl rounded-[24px] border border-border/70 bg-background/95 shadow-sm md:hidden">
          <nav className="flex flex-col gap-2 px-4 py-4">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                onMouseEnter={() => preloadRouteForPath(item.path)}
                onFocus={() => preloadRouteForPath(item.path)}
                onClick={() => setIsMenuOpen(false)}
                className={cn(
                  "flex items-center gap-2 rounded-2xl px-4 py-3 text-sm font-medium transition-colors",
                  isActive(item.path)
                    ? "gradient-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground"
                )}
              >
                {item.icon && <item.icon className="h-4 w-4" />}
                {item.label}
                {item.count ? (
                  <span className="ml-auto inline-flex min-w-5 items-center justify-center rounded-full bg-primary px-1.5 py-0.5 text-[11px] font-semibold text-primary-foreground">
                    {item.count > 99 ? "99+" : item.count}
                  </span>
                ) : null}
              </Link>
            ))}

            <div className="mt-2 flex flex-col gap-2 border-t pt-4">
              <div className="px-1">
                <HeaderAssistant />
              </div>
              {user ? (
                <>
                  <p className="px-4 text-sm text-muted-foreground">{user.email}</p>
                  <Button variant="outline" className="w-full rounded-2xl" onClick={handleSignOut}>
                    <LogOut className="mr-2 h-4 w-4" />
                    Sign Out
                  </Button>
                </>
              ) : (
                <>
                  <Link to="/auth" onClick={() => setIsMenuOpen(false)}>
                    <Button variant="ghost" className="w-full rounded-2xl">
                      Sign In
                    </Button>
                  </Link>
                  <Link to="/admin-login" onClick={() => setIsMenuOpen(false)}>
                    <Button variant="outline" className="w-full rounded-2xl">
                      Admin Login
                    </Button>
                  </Link>
                  <Link to="/auth" onClick={() => setIsMenuOpen(false)}>
                    <Button variant="hero" className="w-full rounded-2xl">
                      Get Started
                    </Button>
                  </Link>
                </>
              )}
            </div>
          </nav>
        </div>
      )}
    </header>
  );
};

export default Header;
