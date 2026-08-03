import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";
import { Spinner } from "@/components/ui/spinner";

const Dashboard = lazy(() => import("@/pages/dashboard"));
const PrivacyPolicy = lazy(() => import("@/pages/privacy-policy"));
const TermsOfService = lazy(() => import("@/pages/terms-of-service"));
const Chat = lazy(() => import("@/pages/FullQuoteChat"));
const Providers = lazy(() => import("@/pages/providers"));
const PrintQuote = lazy(() => import("@/pages/PrintQuote"));
const AuthPage = lazy(() => import("@/pages/AuthPage"));
const ProfilePage = lazy(() => import("@/pages/ProfilePage"));

function PageFallback() {
  return (
    <div
      role="status"
      aria-label="Loading page"
      className="container mx-auto flex min-h-40 max-w-7xl items-center justify-center px-4 py-12 sm:px-6 lg:px-8"
    >
      <Spinner className="h-6 w-6 text-text-tertiary" />
    </div>
  );
}

const WithLayout = ({ children }) => (
  <>
    <Header />
    {children}
    <Footer />
  </>
);

const AppRouter = () => {
  return (
    <Suspense fallback={<PageFallback />}>
      <Routes>
        <Route path="/" element={<WithLayout><Dashboard /></WithLayout>} />
        <Route path="/privacy-policy" element={<WithLayout><PrivacyPolicy /></WithLayout>} />
        <Route path="/terms-of-service" element={<WithLayout><TermsOfService /></WithLayout>} />
        <Route path="/chat" element={<WithLayout><Chat /></WithLayout>} />
        <Route path="/providers" element={<WithLayout><Providers /></WithLayout>} />
        <Route path="/print" element={<PrintQuote />} />

        {/* Auth — no header/footer */}
        <Route path="/auth" element={<AuthPage />} />

        {/* Profile — with header/footer */}
        <Route path="/profile" element={<WithLayout><ProfilePage /></WithLayout>} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
};

export default AppRouter;
