import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 1: no proxy to the API is configured yet — the home page calls the API's full
// http://localhost:8000 URL directly (see src/App.jsx). A dev-server proxy can be added once
// there are more than one or two calls worth simplifying.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Required so the dev server accepts connections from outside the Docker container
    // (0.0.0.0), not just localhost inside it.
    host: true,
  },
});
