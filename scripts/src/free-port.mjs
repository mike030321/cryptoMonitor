#!/usr/bin/env node
/**
 * Task #518 — release a TCP port before a workflow tries to bind it.
 *
 * Replit's workflow supervisor sometimes restarts a service before the
 * previous process has fully released its listening socket, causing the
 * new process to crash with EADDRINUSE before any logging happens. This
 * helper runs as a pre-start step:
 *
 *   1. Try to connect to localhost:PORT — if the connection refuses,
 *      the port is free and we exit immediately.
 *   2. If something is listening, find the *single* PID that owns the
 *      LISTEN socket on this port by parsing /proc/net/tcp{,6} → inode,
 *      then walking /proc/<pid>/fd/* to map inode → pid. SIGTERM only
 *      that PID (and its children), then SIGKILL after 6s.
 *   3. Re-check that the port is free, retrying for up to 6 seconds.
 *
 * Critically: we *never* fall back to a broad pgrep over command-line
 * patterns. An earlier draft did, and it could kill a healthy
 * api-server while restarting ml-engine (or vice versa). If we cannot
 * positively identify the owner of the port, we exit cleanly without
 * killing anything and let the downstream service surface its own
 * EADDRINUSE error.
 *
 * Usage (in package.json):
 *   "predev":   "node ../../scripts/src/free-port.mjs",
 *   "prestart": "node ../../scripts/src/free-port.mjs"
 *
 * The helper exits 0 even if it cannot free the port — the downstream
 * service will then report EADDRINUSE itself (api-server has a
 * dedicated handler; ml-engine relies on uvicorn's own startup error).
 * Hard-failing here would block recovery paths where the port is
 * genuinely held by an unrelated process the supervisor will retry.
 */
import { readdirSync, readFileSync, readlinkSync, statSync } from "node:fs";
import net from "node:net";

const port = Number(process.env.PORT ?? process.argv[2] ?? "0");
if (!Number.isFinite(port) || port <= 0 || port > 65535) {
  console.log(`[free-port] no PORT given or PORT=${port} invalid — skipping`);
  process.exit(0);
}

function isPortFree(p) {
  return new Promise((resolve) => {
    const sock = net.createConnection({ host: "127.0.0.1", port: p });
    const done = (free) => {
      try { sock.destroy(); } catch {}
      resolve(free);
    };
    sock.once("connect", () => done(false));
    sock.once("error", () => done(true));
    setTimeout(() => done(true), 800);
  });
}

/**
 * Scan /proc/net/tcp and /proc/net/tcp6 for sockets in LISTEN state
 * (st=0A) bound to the given port and return the set of socket inodes.
 *
 * Format of each line (whitespace separated):
 *   sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode ...
 * `local_address` is "HEX_IP:HEX_PORT" (the PORT is uppercase hex).
 */
function findListenInodes(targetPort) {
  const inodes = new Set();
  const targetHex = targetPort.toString(16).toUpperCase().padStart(4, "0");
  for (const path of ["/proc/net/tcp", "/proc/net/tcp6"]) {
    let raw;
    try {
      raw = readFileSync(path, "utf8");
    } catch {
      continue;
    }
    const lines = raw.split("\n").slice(1);
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length < 10) continue;
      const local = parts[1];
      const state = parts[3];
      const inode = parts[9];
      if (state !== "0A") continue; // 0A == TCP_LISTEN
      const colon = local.lastIndexOf(":");
      if (colon < 0) continue;
      if (local.slice(colon + 1) !== targetHex) continue;
      const n = Number(inode);
      if (Number.isFinite(n) && n > 0) inodes.add(n);
    }
  }
  return inodes;
}

/**
 * Walk /proc/<pid>/fd/* and return the set of PIDs that hold any of
 * the given socket inodes open. We deliberately ignore EACCES /
 * ENOENT errors — the runtime user can't see other users' fd
 * directories, and short-lived processes can disappear mid-scan.
 */
function findOwningPids(targetInodes) {
  const owners = new Set();
  if (targetInodes.size === 0) return owners;
  let pidEntries;
  try {
    pidEntries = readdirSync("/proc");
  } catch {
    return owners;
  }
  for (const entry of pidEntries) {
    const pid = Number(entry);
    if (!Number.isFinite(pid) || pid <= 0) continue;
    if (pid === process.pid) continue;
    const fdDir = `/proc/${pid}/fd`;
    let fds;
    try {
      fds = readdirSync(fdDir);
    } catch {
      continue;
    }
    for (const fd of fds) {
      let target;
      try {
        target = readlinkSync(`${fdDir}/${fd}`);
      } catch {
        continue;
      }
      const m = /^socket:\[(\d+)\]$/.exec(target);
      if (!m) continue;
      const inode = Number(m[1]);
      if (targetInodes.has(inode)) {
        owners.add(pid);
        break; // one match is enough; move to next pid
      }
    }
  }
  return owners;
}

function describePid(pid) {
  try {
    const cmdline = readFileSync(`/proc/${pid}/cmdline`, "utf8")
      .replace(/\0+$/, "")
      .replace(/\0/g, " ");
    return cmdline || `pid ${pid}`;
  } catch {
    return `pid ${pid}`;
  }
}

function pidStillAlive(pid) {
  try {
    statSync(`/proc/${pid}`);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  if (await isPortFree(port)) {
    console.log(`[free-port] port ${port} already free`);
    return;
  }

  const inodes = findListenInodes(port);
  if (inodes.size === 0) {
    console.warn(
      `[free-port] port ${port} is in use but no LISTEN socket found in /proc/net/tcp{,6} — letting downstream service report its own EADDRINUSE`,
    );
    return;
  }

  const owners = findOwningPids(inodes);
  if (owners.size === 0) {
    console.warn(
      `[free-port] port ${port} owner PID could not be resolved (likely owned by a different user) — letting downstream service report its own EADDRINUSE`,
    );
    return;
  }

  for (const pid of owners) {
    try {
      process.kill(pid, "SIGTERM");
      console.warn(`[free-port] SIGTERM ${pid} (${describePid(pid)})`);
    } catch (err) {
      console.warn(`[free-port] SIGTERM ${pid} failed: ${err?.message ?? err}`);
    }
  }

  for (let i = 0; i < 6; i += 1) {
    await new Promise((r) => setTimeout(r, 1000));
    if (await isPortFree(port)) {
      console.log(`[free-port] port ${port} freed after ${i + 1}s`);
      return;
    }
  }

  console.warn(`[free-port] port ${port} still busy after 6s — sending SIGKILL`);
  for (const pid of owners) {
    if (!pidStillAlive(pid)) continue;
    try {
      process.kill(pid, "SIGKILL");
      console.warn(`[free-port] SIGKILL ${pid}`);
    } catch (err) {
      console.warn(`[free-port] SIGKILL ${pid} failed: ${err?.message ?? err}`);
    }
  }

  await new Promise((r) => setTimeout(r, 1500));
  if (await isPortFree(port)) {
    console.log(`[free-port] port ${port} freed after SIGKILL`);
  } else {
    console.warn(
      `[free-port] port ${port} STILL busy — letting downstream service report its own EADDRINUSE`,
    );
  }
}

main().catch((err) => {
  console.warn(`[free-port] unexpected error (continuing): ${err?.message ?? err}`);
  process.exit(0);
});
