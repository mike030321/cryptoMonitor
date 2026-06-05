import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

// In production, rewrite /api/* calls to the Render api-server.
// In dev, the Vite proxy handles it (no change needed).
const apiOrigin = import.meta.env.VITE_API_ORIGIN as string | undefined;
if (apiOrigin) {
  const _fetch = window.fetch.bind(window);
  window.fetch = (input, init?) => {
    if (typeof input === "string" && input.startsWith("/api/")) {
      input = apiOrigin + input;
    } else if (input instanceof Request && new URL(input.url, location.href).pathname.startsWith("/api/")) {
      const u = new URL(input.url, location.href);
      input = new Request(apiOrigin + u.pathname + u.search, input);
    }
    return _fetch(input, init);
  };
}

createRoot(document.getElementById("root")!).render(<App />);
