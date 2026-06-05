import app from "./app";
import { logger } from "./lib/logger";
import { runMigrations } from "./lib/migrate";
import { hydrateBindings as hydrateMetaBrainBindings } from "./lib/meta-brain";
import { initializeAgents, startMonitoring } from "./lib/monitor";
import { loadFingerprintBuffers } from "./lib/regime-detector";
import { loadAutoApplyTightenOverride, loadTuningStateFromDb } from "./lib/tuning-tracker";
import { startMarketSignalsPoller } from "./lib/market-signals-poller";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

// Task #518 — graceful shutdown so workflow restarts release the
// listening socket before the new process tries to bind it. The
// audit observed intermittent EADDRINUSE crashes when the SIGTERM
// path left the http.Server in a half-closed state. Closing the
// listener inside the signal handler — and unref()'ing the close
// timeout so the process actually exits — is the documented Node.js
// fix.
let shuttingDown = false;
function installSignalHandlers(serverHandle: { close: (cb?: (err?: Error) => void) => void }): void {
  for (const sig of ["SIGTERM", "SIGINT"] as const) {
    process.on(sig, () => {
      if (shuttingDown) return;
      shuttingDown = true;
      logger.info({ signal: sig }, "Received shutdown signal, closing HTTP server");
      // Force-exit guard: if no in-flight request lets us drain in 8s,
      // exit anyway so the workflow can rebind the port.
      const forceExit = setTimeout(() => {
        logger.warn({ signal: sig }, "Forcing exit after 8s shutdown grace period");
        process.exit(0);
      }, 8_000);
      forceExit.unref?.();
      serverHandle.close((err) => {
        if (err) logger.warn({ err }, "Error during HTTP server close");
        clearTimeout(forceExit);
        process.exit(0);
      });
    });
  }
}

async function start(): Promise<void> {
  // Task #347 — apply the database-level safety net for `price_history`
  // (and any future migrations) BEFORE we open the listening socket. We
  // refuse to start with a half-applied schema so a constraint failure
  // here is fatal and visible in the workflow logs, not silently
  // deferred until the first write.
  await runMigrations();

  // Restore the persisted auto-apply tighten override BEFORE we start
  // accepting traffic so dashboard requests never observe the env default
  // during the small window between listen() and the async restore.
  try {
    await loadAutoApplyTightenOverride();
  } catch (preInitErr) {
    logger.error(
      { err: preInitErr },
      "Failed to restore persisted auto-apply tighten override; continuing with env default",
    );
  }

  const server = app.listen(port, async (err) => {
    if (err) {
      logger.error({ err }, "Error listening on port");
      process.exit(1);
    }

    logger.info({ port }, "Server listening");

    try {
      await loadFingerprintBuffers();
      await loadTuningStateFromDb();
      // Task #381 — restore the persisted trade→tick binding map BEFORE
      // any close path can fire, so record_outcome can close the loop on
      // trades opened in a previous boot.
      try {
        await hydrateMetaBrainBindings();
      } catch (hydrateErr) {
        logger.warn(
          { err: String(hydrateErr) },
          "meta-brain trade→tick hydration failed; continuing with empty map",
        );
      }
      await initializeAgents();
      logger.info("AI agents initialized");
      startMonitoring();
      logger.info("Crypto monitoring engine started (30s cycles)");
      // Task #271 — write per-coin snapshots of funding rate, open
      // interest, liquidations proxy, and order-book spread to the
      // `market_signals` table on a 60s cadence. Plus BTC/ETH mid prices
      // so the ml-engine can derive the lead-return features for every
      // other coin. The trainer reads these snapshots in `db.py` and
      // joins them onto each candle bucket inside `labels.py`.
      startMarketSignalsPoller();
      logger.info("Market signals poller started (60s cadence)");
    } catch (initErr) {
      logger.error({ err: initErr }, "Failed to initialize monitoring engine");
    }
  });

  server.on("error", (err: NodeJS.ErrnoException) => {
    if (err.code === "EADDRINUSE") {
      logger.error(
        { port, code: err.code },
        "Port already in use — a previous workflow process did not release the socket. Exiting so the workflow supervisor can retry.",
      );
      process.exit(1);
    }
    logger.error({ err }, "HTTP server error");
  });

  installSignalHandlers(server);
}

void start();
