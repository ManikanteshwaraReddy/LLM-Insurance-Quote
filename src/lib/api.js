/**
 * Centralised API client for SmartHealthQuote backend.
 *
 * Base URL is read from the VITE_API_BASE_URL env variable so it works
 * in both local dev and production without code changes.
 *
 * Auth conventions:
 *   - Access token is stored in memory (authContext) and sent as Bearer header.
 *   - Refresh token is stored in localStorage for page-reload persistence.
 *   - On 401, apiRequest() attempts a silent token refresh once, then gives up.
 */

const BASE = (
  import.meta.env.VITE_API_BASE_URL || "https://smarthealthquote-backend.onrender.com"
).replace(/\/$/, "");

// ─── Token storage ────────────────────────────────────────────────────────────
// Access token lives in memory only (XSS mitigation).
// Refresh token lives in localStorage (survives page reload).

let _accessToken = null;

export const tokenStore = {
  getAccess: () => _accessToken,
  setAccess: (t) => { _accessToken = t; },
  clearAccess: () => { _accessToken = null; },

  getRefresh: () => localStorage.getItem("refreshToken"),
  setRefresh: (t) => localStorage.setItem("refreshToken", t),
  clearRefresh: () => localStorage.removeItem("refreshToken"),

  clearAll: () => {
    _accessToken = null;
    localStorage.removeItem("refreshToken");
  },
};

// ─── Core request helper ──────────────────────────────────────────────────────

async function apiRequest(path, options = {}, retry = true) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  const token = tokenStore.getAccess();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers,
  });

  // Silent token refresh on 401
  if (res.status === 401 && retry) {
    const refreshed = await _silentRefresh();
    if (refreshed) {
      return apiRequest(path, options, false); // one retry
    }
    // Refresh failed — caller must handle
  }

  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const message =
      data?.error?.message ||
      data?.error ||
      data?.details ||
      `Request failed (${res.status})`;
    const err = new Error(message);
    err.status = res.status;
    err.code = data?.error?.code;
    throw err;
  }

  return data;
}

// Attempt a silent token refresh using the stored refresh token.
// Returns true if successful, false otherwise.
async function _silentRefresh() {
  const refreshToken = tokenStore.getRefresh();
  if (!refreshToken) return false;

  try {
    const res = await fetch(`${BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshToken }),
    });

    if (!res.ok) {
      tokenStore.clearAll();
      return false;
    }

    const data = await res.json();
    const tokens = data?.data;
    if (tokens?.accessToken) {
      tokenStore.setAccess(tokens.accessToken);
      if (tokens.refreshToken) tokenStore.setRefresh(tokens.refreshToken);
      return true;
    }
  } catch {
    // network error — give up
  }

  tokenStore.clearAll();
  return false;
}

// ─── Auth endpoints ───────────────────────────────────────────────────────────

/**
 * Register a new account.
 * @param {string} fullName
 * @param {string} email
 * @param {string} password
 * @returns {{ user, tokens }}
 */
export async function register(fullName, email, password) {
  const res = await apiRequest("/auth/register", {
    method: "POST",
    body: JSON.stringify({ fullName, email, password }),
  });
  _storeTokens(res.data?.tokens);
  return res.data;
}

/**
 * Login with email + password.
 * @returns {{ user, tokens }}
 */
export async function login(email, password) {
  const res = await apiRequest("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  _storeTokens(res.data?.tokens);
  return res.data;
}

/**
 * Logout — revokes refresh token on the backend.
 */
export async function logout() {
  try {
    await apiRequest("/auth/logout", { method: "POST" });
  } finally {
    tokenStore.clearAll();
  }
}

/**
 * Fetch the authenticated user's profile.
 * @returns {{ user }}
 */
export async function getMe() {
  const res = await apiRequest("/auth/me");
  return res.data;
}

/**
 * Manually refresh the access token.
 * Returns true on success.
 */
export async function refreshAccessToken() {
  return _silentRefresh();
}

// ─── Quote endpoint ───────────────────────────────────────────────────────────

/**
 * Generate a health insurance quote.
 * Works whether the user is logged in or not.
 */
export async function getQuote(payload) {
  const res = await apiRequest("/api/quote", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  // Quote endpoint returns raw data (not wrapped in { success, data })
  // for backwards compat — handle both shapes.
  return res.data ?? res;
}

// ─── Profile endpoints ────────────────────────────────────────────────────────

/**
 * Update the authenticated user's profile fields.
 * @param {{ fullName?, phone?, gender?, address? }} fields
 */
export async function updateProfile(fields) {
  const res = await apiRequest("/auth/profile", {
    method: "PATCH",
    body: JSON.stringify(fields),
  });
  return res.data;
}

/**
 * Change the authenticated user's password.
 * @param {string} currentPassword
 * @param {string} newPassword
 */
export async function changePassword(currentPassword, newPassword) {
  const res = await apiRequest("/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ currentPassword, newPassword }),
  });
  return res.data;
}

// ─── Health ───────────────────────────────────────────────────────────────────

export async function healthCheck() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

function _storeTokens(tokens) {
  if (!tokens) return;
  if (tokens.accessToken) tokenStore.setAccess(tokens.accessToken);
  if (tokens.refreshToken) tokenStore.setRefresh(tokens.refreshToken);
}
