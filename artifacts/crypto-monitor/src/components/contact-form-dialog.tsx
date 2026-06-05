import { useState, useRef, useEffect } from "react";
import { Mail } from "lucide-react";
import { PhoneInput, type PhoneInputRefType } from "react-international-phone";
import "react-international-phone/style.css";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import {
  getBrowser,
  getDevice,
  getCookie,
  getIpAndCountry,
  validateName,
} from "@/lib/lead-utils";
import { isKnownPixel } from "@/lib/pixel-tokens-client";

// In dev: Vite proxies /phpServices → XAMPP localhost
// In production: points directly to SmarterASP PHP host
const MAIL_ENDPOINT = import.meta.env.VITE_MAIL_ENDPOINT ?? "/phpServices/mail.php";

// ── validation ────────────────────────────────────────────────────────────────

function validateEmail(value: string): string | null {
  if (!value.trim()) return "Email is required";
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) return "Enter a valid email address";
  return null;
}

function validatePhone(value: string): string | null {
  const digits = value.replace(/\D/g, "");
  if (!value || digits.length === 0) return "Phone number is required";
  if (digits.length < 7) return "Phone number is too short";
  return null;
}

// Adapter: lead-utils validateName returns { valid, error? } — unwrap to string | null
function checkName(value: string, field: string): string | null {
  const r = validateName(value, field);
  return r.valid ? null : (r.error ?? null);
}

// ── component ─────────────────────────────────────────────────────────────────

interface FormState {
  name: string;
  lastname: string;
  email: string;
  phone: string;
}

interface FormErrors {
  name?: string;
  lastname?: string;
  email?: string;
  phone?: string;
}

// CSS variable overrides to theme the library for the app's dark palette
const phoneInputTheme: React.CSSProperties = {
  ["--react-international-phone-height" as string]: "36px",
  ["--react-international-phone-font-size" as string]: "14px",
  ["--react-international-phone-border-radius" as string]: "6px",
  ["--react-international-phone-background-color" as string]: "transparent",
  ["--react-international-phone-text-color" as string]: "hsl(var(--foreground))",
  ["--react-international-phone-placeholder-color" as string]: "hsl(var(--muted-foreground))",
  ["--react-international-phone-border-color" as string]: "hsl(var(--border))",
  ["--react-international-phone-country-selector-background-color" as string]: "transparent",
  ["--react-international-phone-country-selector-background-color-hover" as string]:
    "hsl(var(--muted) / 0.5)",
  ["--react-international-phone-country-selector-arrow-color" as string]:
    "hsl(var(--muted-foreground))",
  ["--react-international-phone-dropdown-item-background-color" as string]: "hsl(var(--popover))",
  ["--react-international-phone-dropdown-item-text-color" as string]: "hsl(var(--popover-foreground))",
  ["--react-international-phone-dropdown-item-dial-code-color" as string]:
    "hsl(var(--muted-foreground))",
  ["--react-international-phone-selected-dropdown-item-background-color" as string]:
    "hsl(var(--muted))",
  ["--react-international-phone-selected-dropdown-item-text-color" as string]:
    "hsl(var(--foreground))",
  ["--react-international-phone-dropdown-shadow" as string]:
    "0 8px 32px rgba(0, 0, 0, 0.6)",
  width: "100%",
};

export interface ContactFormDialogProps {
  onCloseSidebar?: () => void;
  /** Controlled open state — when provided the dialog is fully controlled externally */
  controlledOpen?: boolean;
  onControlledOpenChange?: (open: boolean) => void;
  /** Optional caption shown below the title */
  captionText?: string;
}

export function ContactFormDialog({
  onCloseSidebar,
  controlledOpen,
  onControlledOpenChange,
  captionText,
}: ContactFormDialogProps) {
  const { toast } = useToast();
  const isControlled = controlledOpen !== undefined;
  const [internalOpen, setInternalOpen] = useState(false);
  const open = isControlled ? controlledOpen : internalOpen;
  const setOpen = (val: boolean) => {
    if (isControlled) {
      onControlledOpenChange?.(val);
    } else {
      setInternalOpen(val);
    }
  };
  const [loading, setLoading] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [form, setForm] = useState<FormState>({ name: "", lastname: "", email: "", phone: "" });
  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Partial<Record<keyof FormState, boolean>>>({});

  const formRef = useRef<HTMLFormElement>(null);
  const phoneInputRef = useRef<PhoneInputRefType>(null);

  // Auto-detect country from IP when the dialog opens
  useEffect(() => {
    if (!open) return;
    getIpAndCountry().then(({ userCountry }) => {
      if (userCountry && userCountry !== "Unknown") {
        phoneInputRef.current?.setCountry(userCountry.toLowerCase());
      }
    });
  }, [open]);

  function handleChange(field: keyof FormState, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
    if (touched[field]) runValidation(field, value);
  }

  function handleBlur(field: keyof FormState) {
    setTouched((prev) => ({ ...prev, [field]: true }));
    runValidation(field, form[field]);
  }

  function runValidation(field: keyof FormState, value: string): string | null {
    let error: string | null = null;
    if (field === "name")     error = checkName(value, "First name");
    if (field === "lastname") error = checkName(value, "Last name");
    if (field === "email")    error = validateEmail(value);
    if (field === "phone")    error = validatePhone(value);
    setErrors((prev) => ({ ...prev, [field]: error ?? undefined }));
    return error;
  }

  function validateAll(): boolean {
    const errs: FormErrors = {
      name:     checkName(form.name, "First name") ?? undefined,
      lastname: checkName(form.lastname, "Last name") ?? undefined,
      email:    validateEmail(form.email) ?? undefined,
      phone:    validatePhone(form.phone) ?? undefined,
    };
    setErrors(errs);
    setTouched({ name: true, lastname: true, email: true, phone: true });
    return !Object.values(errs).some(Boolean);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!validateAll()) return;

    setLoading(true);
    setSubmitError(null);

    try {
      const { userIp, userCountry } = await getIpAndCountry();
      const params = new URLSearchParams(window.location.search);
      const p = (key: string) => params.get(key) || getCookie(key) || "";

      const formData = new FormData();
      formData.append("name",        form.name.trim());
      formData.append("lastname",    form.lastname.trim());
      formData.append("email",       form.email.trim());
      formData.append("phone",       form.phone.trim());
      formData.append("ip",          userIp);
      formData.append("country",     userCountry);
      formData.append("browser",     getBrowser());
      formData.append("device",      getDevice());
      formData.append("user_agent",  navigator.userAgent);
      formData.append("subid",       p("subid"));
      const rawPixel = params.get("pixel") || getCookie("pixel") || "";
      formData.append("pixel_id",    isKnownPixel(rawPixel) ? rawPixel : "");
      formData.append("campaign_id", p("campaign_id"));
      formData.append("adset_id",    p("adset_id"));
      formData.append("ad_id",       p("ad_id"));
      formData.append("creo_id",     p("creo_id"));
      formData.append("flow",        p("flow"));
      formData.append("fb_account",  p("fb_account"));
      formData.append("fbc",         params.get("fbc") || getCookie("fbc") || getCookie("_fbc") || "");
      formData.append("fbp",         params.get("fbp") || getCookie("fbp") || getCookie("_fbp") || "");

      const response = await fetch(MAIL_ENDPOINT, { method: "POST", body: formData });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const result = await response.json();
      if (result.status === "success") {
        setOpen(false);
        toast({
          title: "Message sent!",
          description: "We'll be in touch shortly.",
        });
        if (result.redirectUrl?.trim()) {
          setTimeout(() => { window.location.href = result.redirectUrl; }, 1000);
        }
      } else {
        setSubmitError(result.message || "Submission failed. Please try again.");
      }
    } catch {
      setSubmitError("Network error. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  function handleOpenChange(val: boolean) {
    setOpen(val);
    if (!val) {
      setForm({ name: "", lastname: "", email: "", phone: "" });
      setErrors({});
      setTouched({});
      setSubmitError(null);
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      {/* Render the sidebar trigger only when NOT in controlled/global mode */}
      {!isControlled && (
        <DialogTrigger asChild>
          <button
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all duration-200 group"
            onClick={() => onCloseSidebar?.()}  style={{ cursor: 'pointer' }}
          >
            <span className="w-8 h-8 rounded-lg flex items-center justify-center bg-white/5 group-hover:bg-white/10 transition-all">
              <Mail className="w-4 h-4" />
            </span>
            Contact Us
          </button>
        </DialogTrigger>
      )}

      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold">Contact Us</DialogTitle>
          {captionText && (
            <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
              {captionText}
            </p>
          )}
        </DialogHeader>

        <form ref={formRef} onSubmit={handleSubmit} noValidate className="grid gap-4 pt-2">
            {/* Name row */}
            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-1.5">
                <Label htmlFor="cf-firstname">First name</Label>
                <Input
                  id="cf-firstname"
                  placeholder="John"
                  value={form.name}
                  onChange={(e) => handleChange("name", e.target.value)}
                  onBlur={() => handleBlur("name")}
                  className={errors.name ? "border-destructive focus-visible:ring-destructive" : ""}
                />
                <p className="h-4 text-xs text-destructive">{errors.name ?? ""}</p>
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="cf-lastname">Last name</Label>
                <Input
                  id="cf-lastname"
                  placeholder="Doe"
                  value={form.lastname}
                  onChange={(e) => handleChange("lastname", e.target.value)}
                  onBlur={() => handleBlur("lastname")}
                  className={errors.lastname ? "border-destructive focus-visible:ring-destructive" : ""}
                />
                <p className="h-4 text-xs text-destructive">{errors.lastname ?? ""}</p>
              </div>
            </div>

            {/* Email */}
            <div className="grid gap-1.5">
              <Label htmlFor="cf-email">Email</Label>
              <Input
                id="cf-email"
                type="email"
                placeholder="john@example.com"
                value={form.email}
                onChange={(e) => handleChange("email", e.target.value)}
                onBlur={() => handleBlur("email")}
                className={errors.email ? "border-destructive focus-visible:ring-destructive" : ""}
              />
              <p className="h-4 text-xs text-destructive">{errors.email ?? ""}</p>
            </div>

            {/* Phone with flag dropdown */}
            <div className="grid gap-1.5">
              <Label>Phone number</Label>
              <div style={phoneInputTheme}>
                <PhoneInput
                  ref={phoneInputRef}
                  defaultCountry="us"
                  value={form.phone}
                  onChange={(phone) => handleChange("phone", phone)}
                  onBlur={() => handleBlur("phone")}
                  placeholder="000 000 0000"
                  inputClassName="w-full !bg-transparent !text-foreground placeholder:!text-muted-foreground"
                  countrySelectorStyleProps={{
                    dropdownStyleProps: {
                      style: { zIndex: 9999 },
                    },
                  }}
                />
              </div>
              <p className="h-4 text-xs text-destructive">{errors.phone ?? ""}</p>
            </div>

            {submitError && (
              <p className="text-xs text-destructive bg-destructive/10 px-3 py-2 rounded-lg">
                {submitError}
              </p>
            )}

            <Button type="submit" className="mt-1" disabled={loading}>
              {loading ? "Sending…" : "Send message"}
            </Button>
          </form>
      </DialogContent>
    </Dialog>
  );
}
