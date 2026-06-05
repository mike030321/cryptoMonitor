// Shared utility functions used by both contact.js (HTML landing pages)
// and the React app (src/lib/lead-utils.ts mirrors this exactly).

export function getCookie(name) {
  const match = document.cookie.match("(^|;)\\s*" + name + "\\s*=\\s*([^;]+)");
  return match ? decodeURIComponent(match[2]) : null;
}

export function getBrowser() {
  const ua = navigator.userAgent;
  if (ua.includes("Chrome") && !ua.includes("Edg")) return "Chrome";
  if (ua.includes("Firefox")) return "Firefox";
  if (ua.includes("Safari") && !ua.includes("Chrome")) return "Safari";
  if (ua.includes("Edg")) return "Edge";
  if (ua.includes("Opera") || ua.includes("OPR")) return "Opera";
  return "Unknown";
}

export function getDevice() {
  const ua = navigator.userAgent;
  if (/iPad/i.test(ua) || (/Android/i.test(ua) && !/Mobile/i.test(ua))) return "Tablet";
  if (/Android|iPhone|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua)) return "Phone";
  return "Desktop";
}

export async function fetchWithTimeout(url, options = {}, timeout = 10000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") throw new Error("Request timed out");
    throw error;
  }
}

let cachedIpData = null;

export async function getIpAndCountry() {
  if (cachedIpData) return cachedIpData;

  try {
    const res = await fetchWithTimeout("https://ipinfo.io/json?token=a91cf61e10058a", {}, 5000);
    if (res.ok) {
      const d = await res.json();
      cachedIpData = { userIp: d.ip ?? "", userCountry: d.country ?? "Unknown" };
      return cachedIpData;
    }
  } catch { /* fallthrough to backup */ }

  try {
    const res = await fetchWithTimeout("https://ipapi.co/json/", {}, 5000);
    if (res.ok) {
      const d = await res.json();
      cachedIpData = { userIp: d.ip ?? "", userCountry: d.country_code ?? "Unknown" };
      return cachedIpData;
    }
  } catch { /* fallthrough */ }

  return { userIp: "", userCountry: "Unknown" };
}

// Returns { valid: boolean, error?: string }
export function validateName(value, fieldName) {
  if (!value.trim()) return { valid: false, error: `${fieldName} is required` };
  if (value.trim().length < 2) return { valid: false, error: `${fieldName} must be at least 2 characters` };
  if (/\d/.test(value)) return { valid: false, error: `${fieldName} cannot contain numbers` };
  if (!/^[a-zA-ZÀ-ÿ\s'-]+$/.test(value)) return { valid: false, error: `${fieldName} can only contain letters` };
  return { valid: true };
}
