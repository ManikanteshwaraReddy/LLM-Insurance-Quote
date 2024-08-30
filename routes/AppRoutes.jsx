import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import NavBar from "../components/NavBar/NavBar";
import Dashboard from "../components/Dashboard/Dashboard";
import Help from "../components/Help/Help";
import Explore from "../components/Explore/Explore";
import QuoteGeneration from "../components/QuoteGeneration/QuoteGeneration";
import PrintQuote from "../components/PrintQuote/PrintQuote";

const AppRoutes = () => {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <>
            <NavBar
              items={[
                { title: "Explore", path: "/explore", imgpath: "/Explore.svg" },
                { title: "Help", path: "/help", imgpath: "/Help.svg" },
              ]}
            />
            <Dashboard />
          </>
        }
      />
      <Route
        path="/explore"
        element={
          <>
            <NavBar
              items={[
                { title: "Home", path: "/", imgpath: "/Home.svg" },
                { title: "Help", path: "/help", imgpath: "/Help.svg" }]}
            />
            <Explore />
          </>
        }
      />
      <Route path="/help" element={<Help />} />
      <Route
        path="/generate-quote"
        element={
          <>
            <NavBar
              items={[
                { title: "Home", path: "/", imgpath: "/Home.svg" },
                { title: "Help", path: "/help", imgpath: "/Help.svg" }]}
            />
            <QuoteGeneration />
          </>
        }
      />
      <Route path="/print-quote" element={<PrintQuote />} />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
};

export default AppRoutes;
