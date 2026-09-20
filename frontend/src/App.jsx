import { useEffect, useState } from "react";

const API_BASE_URL = "http://localhost:8000";

/**
 * NexusAI home page — Phase 1. No search, no ingestion, no model calls yet; this page only
 * proves the frontend can reach the API, so later phases build on a stack already known to work.
 */
export default function App() {
  const [apiStatus, setApiStatus] = useState("checking...");

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => setApiStatus(`connected (${data.status})`))
      .catch((err) => setApiStatus(`unreachable (${err.message})`));
    return () => controller.abort();
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "3rem", maxWidth: 640 }}>
      <h1>NexusAI</h1>
      <p>Local-first multimodal search — Phase 1 foundation.</p>
      <p>
        API status: <strong>{apiStatus}</strong>
      </p>
    </main>
  );
}
