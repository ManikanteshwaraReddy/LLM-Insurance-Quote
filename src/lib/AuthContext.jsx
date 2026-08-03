/**
 * AuthContext — global authentication state.
 *
 * Provides:
 *   useAuth()  → { user, isLoading, isAuthenticated, login, register, logout }
 *
 * On app load it attempts to silently refresh the access token from the stored
 * refresh token so the user stays logged in across page reloads.
 *
 * Usage:
 *   Wrap your app in <AuthProvider> (done in main.jsx).
 *   Then in any component:
 *     const { user, login, logout, isAuthenticated } = useAuth();
 */
import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import {
  login as apiLogin,
  register as apiRegister,
  logout as apiLogout,
  getMe,
  refreshAccessToken,
  tokenStore,
} from "@/lib/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true); // true until initial auth check done

  // ── On mount: attempt silent refresh if a refresh token is stored ──────────
  useEffect(() => {
    async function initAuth() {
      const storedRefresh = tokenStore.getRefresh();
      if (!storedRefresh) {
        setIsLoading(false);
        return;
      }

      try {
        const ok = await refreshAccessToken();
        if (ok) {
          const { user: me } = await getMe();
          setUser(me);
        }
      } catch {
        tokenStore.clearAll();
      } finally {
        setIsLoading(false);
      }
    }

    initAuth();
  }, []);

  // ── Login ──────────────────────────────────────────────────────────────────
  const login = useCallback(async (email, password) => {
    const data = await apiLogin(email, password);
    setUser(data.user);
    return data;
  }, []);

  // ── Register ───────────────────────────────────────────────────────────────
  const register = useCallback(async (fullName, email, password) => {
    const data = await apiRegister(fullName, email, password);
    setUser(data.user);
    return data;
  }, []);

  // ── Logout ─────────────────────────────────────────────────────────────────
  const logout = useCallback(async () => {
    await apiLogout();
    setUser(null);
  }, []);

  const value = {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth() must be used inside <AuthProvider>");
  }
  return ctx;
}
