import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "src"),
    },
    dedupe: ["react", "react-dom"],
  },
  test: {
    environment: "jsdom",
    globals: false,
    include: ["src/**/__tests__/**/*.{test,spec}.{ts,tsx}"],
    setupFiles: ["src/test-setup.ts"],
  },
});
