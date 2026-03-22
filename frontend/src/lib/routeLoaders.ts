export const loadIndexPage = () => import("../pages/Index");
export const loadDashboardPage = () => import("../pages/Dashboard");
export const loadSubmitComplaintPage = () => import("../pages/SubmitComplaint");
export const loadComplaintDetailPage = () => import("../pages/ComplaintDetail");
export const loadAuthPage = () => import("../pages/Auth");
export const loadAdminLoginPage = () => import("../pages/AdminLogin");
export const loadAdminPage = () => import("../pages/Admin");
export const loadAdminAccessPage = () => import("../pages/AdminAccess");
export const loadSuperAdminPage = () => import("../pages/SuperAdmin");
export const loadFaqPage = () => import("../pages/FAQ");
export const loadNotificationsPage = () => import("../pages/Notifications");
export const loadNotFoundPage = () => import("../pages/NotFound");

export const preloadRouteForPath = (path: string) => {
  switch (path) {
    case "/":
      void loadIndexPage();
      break;
    case "/dashboard":
      void loadDashboardPage();
      break;
    case "/submit":
      void loadSubmitComplaintPage();
      break;
    case "/auth":
      void loadAuthPage();
      break;
    case "/admin-login":
      void loadAdminLoginPage();
      break;
    case "/admin":
      void loadAdminPage();
      break;
    case "/admin-access":
      void loadAdminAccessPage();
      break;
    case "/super-admin":
      void loadSuperAdminPage();
      break;
    case "/faq":
      void loadFaqPage();
      break;
    case "/notifications":
      void loadNotificationsPage();
      break;
    default:
      break;
  }
};
