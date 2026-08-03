const DEFAULT_API_BASE_URL = "http://localhost:8000";

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, "");

export async function getQuote(payload) {
  const response = await fetch(`${apiBaseUrl}/api/quote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data?.details || data?.error || "Unable to generate a quote.");
  }

  return data;
}
