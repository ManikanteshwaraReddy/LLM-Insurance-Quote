import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";
import Dashboard from "@/pages/dashboard";
import PrivacyPolicy from "@/pages/privacy-policy";
import TermsOfService from "@/pages/terms-of-service";
import Chat from "@/pages/FullQuoteChat";
import Providers from "@/pages/providers";
import PrintQuote from "@/pages/PrintQuote";
import AuthPage from "@/pages/AuthPage";

const AppRouter = () => {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <>
            <Header />
            <Dashboard />
            <Footer />
          </>
        }
      />
      <Route
        path="/privacy-policy"
        element={
          <>
            <Header />
            <PrivacyPolicy />
            <Footer />
          </>
        }
      />
      <Route
        path="/terms-of-service"
        element={
          <>
            <Header />
            <TermsOfService />
            <Footer />
          </>
        }
      />
      <Route
        path="/chat"
        element={
          <>
            <Header />
            <Chat />
            <Footer />
          </>
        }
      />
      <Route
        path="/providers"
        element={
          <>
            <Header />
            <Providers />
            <Footer />
          </>
        }
      />
      <Route path="/print" element={<PrintQuote />} />
      <Route
        path="/auth"
        element={
          <>
            {/* AuthPage typically doesn't have the main Header/Footer */}
            {/* <Header /> */}
            <AuthPage />
            {/* <Footer /> */}
          </>
        }
      />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
};

export default AppRouter;
