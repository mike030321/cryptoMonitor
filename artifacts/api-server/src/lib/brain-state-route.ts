/**
 * Task #406 — handler factory for `POST /api/crypto/brain/state`.
 *
 * The route logic lives here (rather than inlined in routes/crypto/index.ts)
 * so it can be exercised end-to-end by integration tests with the
 * verification-history check and the brain-flag writer stubbed out. The
 * production wiring in routes/crypto/index.ts calls this factory with all
 * defaults; tests pass in deterministic doubles. Admin-key auth stays at
 * the route layer so this handler stays focused on the
 * gate-then-toggle policy.
 */
import type { Request, Response } from "express";
import { hasPromotedSlice, type PromotionGateVerdict } from "./brain-promotion-gate";
import { setBrainState, type BrainSource } from "./brain-flag";

interface BrainStateLike {
  enabled: boolean;
  source: BrainSource;
  lastChangedAt: string;
}
import {
  getBrainRevertLog,
  getAutoRevertCounter,
  type BrainRevertEvent,
} from "./brain-auto-revert";

export interface BrainStateHandlerDeps {
  hasPromotedSlice?: () => Promise<PromotionGateVerdict>;
  setBrainState?: (enabled: boolean, source: "manual") => Promise<BrainStateLike>;
  getBrainRevertLog?: () => Promise<BrainRevertEvent[]>;
  getAutoRevertCounter?: () => number;
}

export function createBrainStatePostHandler(deps: BrainStateHandlerDeps = {}) {
  const _hasPromotedSlice = deps.hasPromotedSlice ?? (() => hasPromotedSlice());
  const _setBrainState = deps.setBrainState ?? setBrainState;
  const _getBrainRevertLog = deps.getBrainRevertLog ?? getBrainRevertLog;
  const _getAutoRevertCounter = deps.getAutoRevertCounter ?? getAutoRevertCounter;
  return async function handleBrainStatePost(req: Request, res: Response): Promise<void> {
    const body = (req.body ?? {}) as { enabled?: unknown };
    if (typeof body.enabled !== "boolean") {
      res.status(400).json({ error: "body must include { enabled: boolean }" });
      return;
    }
    try {
      if (body.enabled === true) {
        const verdict = await _hasPromotedSlice();
        if (!verdict.ok) {
          res.status(409).json({
            error: "promotion_gate_blocked",
            message:
              "Refusing to enable quant_brain_enabled: latest verification-history " +
              "record contains no promoted slice. Run a retrain that produces at " +
              "least one slice with promoted:true before flipping the brain on.",
            gate: verdict,
          });
          return;
        }
      }
      const state = await _setBrainState(body.enabled, "manual");
      const revertLog = await _getBrainRevertLog();
      res.json({
        ...state,
        autoRevert: {
          consecutiveDriftCycles: _getAutoRevertCounter(),
          recentEvents: revertLog.slice(0, 10),
        },
      });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
    }
  };
}
