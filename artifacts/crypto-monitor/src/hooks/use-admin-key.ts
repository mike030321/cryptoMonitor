import { useEffect, useRef, useState, useCallback } from "react";

const SESSION_STORAGE_PREFIX = "crypto-monitor:admin-key:";

const cache = new Map<string, string>();
const lastRejected = new Map<string, string>();
const rejectionCounter = new Map<string, number>();
const requestCounter = new Map<string, number>();
const listeners = new Map<string, Set<() => void>>();
const hydrated = new Set<string>();

function getSessionStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function hydrateFromSession(keyName: string) {
  if (hydrated.has(keyName)) return;
  hydrated.add(keyName);
  const storage = getSessionStorage();
  if (!storage) return;
  try {
    const stored = storage.getItem(SESSION_STORAGE_PREFIX + keyName);
    if (stored) cache.set(keyName, stored);
  } catch {
    // ignore storage failures (private mode, quota, etc.)
  }
}

function persistToSession(keyName: string, value: string | null) {
  const storage = getSessionStorage();
  if (!storage) return;
  try {
    if (value === null) {
      storage.removeItem(SESSION_STORAGE_PREFIX + keyName);
    } else {
      storage.setItem(SESSION_STORAGE_PREFIX + keyName, value);
    }
  } catch {
    // ignore storage failures
  }
}

function notify(keyName: string) {
  const set = listeners.get(keyName);
  if (!set) return;
  for (const l of set) l();
}

function setCachedKey(keyName: string, value: string | null) {
  if (value === null) {
    cache.delete(keyName);
  } else {
    cache.set(keyName, value);
    lastRejected.delete(keyName);
    rejectionCounter.set(keyName, 0);
    requestCounter.set(keyName, 0);
  }
  persistToSession(keyName, value);
  notify(keyName);
}

function recordRejection(keyName: string, attemptedValue: string) {
  cache.delete(keyName);
  persistToSession(keyName, null);
  lastRejected.set(keyName, attemptedValue);
  rejectionCounter.set(keyName, (rejectionCounter.get(keyName) ?? 0) + 1);
  notify(keyName);
}

function recordKeyRequest(keyName: string) {
  requestCounter.set(keyName, (requestCounter.get(keyName) ?? 0) + 1);
  notify(keyName);
}

export interface UseAdminKey {
  /** Env-var name this hook is operating against (e.g. ADMIN_API_KEY). */
  keyName: string;
  /** True when a key is cached for this tab. */
  hasKey: boolean;
  /** The last value the server rejected, so the inline field can pre-fill it. */
  lastRejected: string | null;
  /** Increments each time the server rejects a key — used to refocus the field. */
  rejectionAttempt: number;
  /** Increments each time an action needed a key but none was cached. */
  keyRequestAttempt: number;
  /** Cache the key entered via the inline panel field. */
  setKey: (value: string) => void;
  /**
   * Returns the cached key (or null when none is cached). Unlike the previous
   * implementation this NEVER opens a `window.prompt` — operators paste keys
   * into the inline panel field instead.
   */
  ensureKey: (action: string) => string | null;
  clearKey: () => void;
  adminFetch: (input: RequestInfo, init: RequestInit & { action: string }) => Promise<Response | null>;
}

export interface UseAdminKeyOptions {
  /** Environment variable name shown in the inline field. Defaults to ADMIN_API_KEY. */
  keyName?: string;
  /**
   * Called when an admin request returns 401/403 and the cached key is cleared.
   * Use this to surface a clear "key rejected" toast to the operator; the
   * inline field will also re-focus pre-filled with the rejected value.
   */
  onRejected?: (keyName: string, attempt: number) => void;
}

export function useAdminKey(options: UseAdminKeyOptions = {}): UseAdminKey {
  const keyName = options.keyName ?? "ADMIN_API_KEY";
  const [, setTick] = useState(0);
  const onRejectedRef = useRef(options.onRejected);
  onRejectedRef.current = options.onRejected;

  useEffect(() => {
    hydrateFromSession(keyName);
    const listener = () => setTick((t) => t + 1);
    let set = listeners.get(keyName);
    if (!set) {
      set = new Set();
      listeners.set(keyName, set);
    }
    set.add(listener);
    setTick((t) => t + 1);
    return () => {
      set?.delete(listener);
    };
  }, [keyName]);

  const setKey = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;
      setCachedKey(keyName, trimmed);
    },
    [keyName],
  );

  const ensureKey = useCallback(
    (_action: string): string | null => {
      const cached = cache.get(keyName);
      if (cached) return cached;
      // Bump the request counter so any inline AdminKeyField listening to
      // this key can refocus and prompt the operator to paste it.
      recordKeyRequest(keyName);
      return null;
    },
    [keyName],
  );

  const clearKey = useCallback(() => {
    setCachedKey(keyName, null);
    lastRejected.delete(keyName);
    rejectionCounter.set(keyName, 0);
    requestCounter.set(keyName, 0);
    notify(keyName);
  }, [keyName]);

  const adminFetch = useCallback(
    async (input: RequestInfo, init: RequestInit & { action: string }): Promise<Response | null> => {
      const { action, headers, ...rest } = init;
      const key = ensureKey(action);
      if (!key) return null;
      const res = await fetch(input, {
        ...rest,
        headers: { ...(headers as Record<string, string> | undefined), "x-admin-key": key },
      });
      if (res.status === 401 || res.status === 403) {
        const attempt = (rejectionCounter.get(keyName) ?? 0) + 1;
        recordRejection(keyName, key);
        onRejectedRef.current?.(keyName, attempt);
        return null;
      }
      if (lastRejected.has(keyName) || (rejectionCounter.get(keyName) ?? 0) > 0) {
        lastRejected.delete(keyName);
        rejectionCounter.set(keyName, 0);
        notify(keyName);
      }
      return res;
    },
    [ensureKey, keyName],
  );

  return {
    keyName,
    hasKey: cache.has(keyName),
    lastRejected: lastRejected.get(keyName) ?? null,
    rejectionAttempt: rejectionCounter.get(keyName) ?? 0,
    keyRequestAttempt: requestCounter.get(keyName) ?? 0,
    setKey,
    ensureKey,
    clearKey,
    adminFetch,
  };
}
