export function formatCurrency(value: number): string {
  const abs = Math.abs(value);
  // Sub-penny coins (SHIB, FLOKI, PEPE, BONK) need many decimals to be readable.
  // Auto-scale so the user always sees significant digits instead of "$0.0000".
  let minFrac: number;
  let maxFrac: number;
  if (abs === 0) {
    minFrac = 2; maxFrac = 2;
  } else if (abs >= 1) {
    minFrac = 2; maxFrac = 2;
  } else if (abs >= 0.01) {
    minFrac = 4; maxFrac = 4;
  } else if (abs >= 0.0001) {
    minFrac = 6; maxFrac = 6;
  } else {
    minFrac = 8; maxFrac = 10;
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: minFrac,
    maximumFractionDigits: maxFrac,
  }).format(value);
}

export function formatCompactCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: "compact",
    compactDisplay: "short",
  }).format(value);
}

export function formatPercentage(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value / 100);
}

export function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const seconds = Math.round((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return `${seconds}s ago`;
  
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}
