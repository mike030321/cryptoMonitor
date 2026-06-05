/**
 * Task #468 — strategy-profile registry barrel.
 */

export {
  agentProfileSchema,
  agentProfileListSchema,
  agentFamilySchema,
  agentRegimeSchema,
  AGENT_STATUSES,
  type AgentProfile,
  type AgentStatus,
  type AgentFamily,
  type AgentRegime,
} from "./schema";

export {
  getAgentProfile,
  tryGetAgentProfile,
  listProfiles,
  listProfileIds,
  listExecutingProfileIds,
  profileAllowsRegime,
} from "./registry";

export {
  mapLegacyNameToProfileId,
  mapLegacyNameToSubId,
  listKnownLegacyNamesForTests,
} from "./compat";

export { syncAgentProfileIds } from "./migration";

export { seedExecutorAgents } from "./seed";

export {
  loadAgentRegistryCache,
  getCachedProfileForAgentId,
  tryGetCachedEntry,
  getCacheStats,
  installSighupReload,
  AgentNotExecutableError,
  _resetCacheForTests,
  _seedCacheForTests,
} from "./cache";

export {
  evaluateRetirementCandidates,
  getRetirementSnapshot,
  startRetirementLoop,
  _resetRetirementForTests,
  type RetirementCandidate,
} from "./retirement";
