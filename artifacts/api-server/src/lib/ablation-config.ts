import { logger } from "./logger";

export interface AblationConfig {
  dualConsensus: boolean;
  fingerprintMatching: boolean;
  contagionDetection: boolean;
  confidenceCalibration: boolean;
  regimeDetection: boolean;
  agentSpecialization: boolean;
}

const defaultConfig: AblationConfig = {
  dualConsensus: true,
  fingerprintMatching: true,
  contagionDetection: true,
  confidenceCalibration: true,
  regimeDetection: true,
  agentSpecialization: true,
};

let currentConfig: AblationConfig = { ...defaultConfig };

export function getAblationConfig(): AblationConfig {
  return { ...currentConfig };
}

export function updateAblationConfig(updates: Partial<AblationConfig>): AblationConfig {
  const changed: string[] = [];
  for (const [key, value] of Object.entries(updates)) {
    if (key in currentConfig && typeof value === "boolean") {
      const k = key as keyof AblationConfig;
      if (currentConfig[k] !== value) {
        changed.push(`${key}: ${currentConfig[k]} → ${value}`);
        currentConfig[k] = value;
      }
    }
  }
  if (changed.length > 0) {
    logger.info({ changes: changed }, "Ablation config updated");
  }
  return { ...currentConfig };
}

export function resetAblationConfig(): AblationConfig {
  currentConfig = { ...defaultConfig };
  logger.info("Ablation config reset to defaults (all features enabled)");
  return { ...currentConfig };
}
