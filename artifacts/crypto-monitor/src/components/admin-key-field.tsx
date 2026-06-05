import { useEffect, useRef, useState } from "react";
import { AlertCircle, Eye, EyeOff, KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { UseAdminKey } from "@/hooks/use-admin-key";

interface AdminKeyFieldProps {
  admin: UseAdminKey;
  label?: string;
  helpText?: string;
  className?: string;
  autoFocus?: boolean;
  testIdPrefix?: string;
}

export function AdminKeyField({
  admin,
  label,
  helpText,
  className,
  autoFocus = false,
  testIdPrefix = "admin-key",
}: AdminKeyFieldProps) {
  const { keyName, lastRejected, rejectionAttempt, keyRequestAttempt, setKey } = admin;
  const [value, setValue] = useState(lastRejected ?? "");
  const [show, setShow] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (rejectionAttempt > 0) {
      setValue(lastRejected ?? "");
      const el = inputRef.current;
      if (el) {
        el.focus();
        el.select();
      }
    }
  }, [rejectionAttempt, lastRejected]);

  useEffect(() => {
    if (keyRequestAttempt > 0) {
      const el = inputRef.current;
      if (el) {
        el.focus();
        el.select();
      }
    }
  }, [keyRequestAttempt]);

  useEffect(() => {
    if (autoFocus) inputRef.current?.focus();
  }, [autoFocus]);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    setKey(trimmed);
  };

  const rejected = rejectionAttempt > 0;
  const inputId = `admin-key-input-${keyName}`;

  return (
    <div
      className={cn(
        "rounded-lg border p-3",
        rejected
          ? "border-rose-500/40 bg-rose-500/[0.04]"
          : "border-amber-500/30 bg-amber-500/[0.04]",
        className,
      )}
      data-testid={`${testIdPrefix}-field`}
    >
      <div className="flex items-center gap-2">
        <KeyRound
          className={cn("w-3.5 h-3.5", rejected ? "text-rose-300" : "text-amber-300")}
        />
        <Label
          htmlFor={inputId}
          className="text-[11px] font-mono uppercase tracking-wider text-muted-foreground"
        >
          {label ?? `Admin key (${keyName})`}
        </Label>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="mt-2 flex flex-wrap items-center gap-2"
      >
        <div className="relative flex-1 min-w-[180px]">
          <Input
            id={inputId}
            ref={inputRef}
            type={show ? "text" : "password"}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Paste admin key"
            autoComplete="off"
            spellCheck={false}
            data-testid={`${testIdPrefix}-input`}
            className="h-9 pr-9 font-mono text-sm"
          />
          <button
            type="button"
            onClick={() => setShow((s) => !s)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            aria-label={show ? "Hide admin key" : "Show admin key"}
            data-testid={`${testIdPrefix}-toggle-visibility`}
          >
            {show ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
        <Button
          type="submit"
          size="sm"
          disabled={!value.trim()}
          data-testid={`${testIdPrefix}-save`}
        >
          Save key
        </Button>
      </form>
      {rejected ? (
        <p
          className="mt-2 text-[11px] font-mono text-rose-300 flex items-center gap-1.5"
          data-testid={`${testIdPrefix}-rejected-hint`}
        >
          <AlertCircle className="w-3 h-3" />
          Key rejected — fix the typo and re-save. Last attempt is pre-filled.
        </p>
      ) : helpText ? (
        <p className="mt-2 text-[11px] font-mono text-muted-foreground">{helpText}</p>
      ) : null}
    </div>
  );
}
