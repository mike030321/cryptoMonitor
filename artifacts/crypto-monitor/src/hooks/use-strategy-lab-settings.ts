import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

export interface DcaSettings {
  drawdownTriggerPct: number;
  resumeLookbackDays: number;
  cycleDeployUsd: number;
  buyIntervalHours: number;
}

export interface StrategySettingsResponse {
  settings: DcaSettings;
  defaults: DcaSettings;
}

export function useStrategyLabSettings() {
  return useQuery<StrategySettingsResponse>({
    queryKey: ["strategy-lab-settings"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/strategy-lab/settings`);
      if (!res.ok) throw new Error("Failed to load strategy lab settings");
      return res.json();
    },
  });
}

export function useUpdateStrategyLabSettings() {
  const qc = useQueryClient();
  return useMutation<StrategySettingsResponse, Error, Partial<DcaSettings>>({
    mutationFn: async (input) => {
      const res = await fetch(`${apiBase}/crypto/strategy-lab/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
      if (!res.ok) throw new Error("Failed to save strategy lab settings");
      return res.json();
    },
    onSuccess: (data) => {
      qc.setQueryData(["strategy-lab-settings"], data);
    },
  });
}
