/* eslint-disable react-refresh/only-export-components */
import { Check, X } from "lucide-react";
import { cn } from "@/lib/utils";

export const PASSWORD_REQUIREMENTS = [
  { id: "length", label: "At least 8 characters", test: (p) => p.length >= 8 },
  { id: "upper", label: "One uppercase letter", test: (p) => /[A-Z]/.test(p) },
  { id: "lower", label: "One lowercase letter", test: (p) => /[a-z]/.test(p) },
  { id: "digit", label: "One number", test: (p) => /\d/.test(p) },
  {
    id: "special",
    label: "One special character",
    test: (p) => /[^a-zA-Z0-9\s]/.test(p),
  },
];

function PasswordChecklist({ password, className }) {
  return (
    <ul className={cn("mt-2.5 space-y-1.5", className)}>
      {PASSWORD_REQUIREMENTS.map((req) => {
        const met = req.test(password);
        return (
          <li
            key={req.id}
            className={cn(
              "flex items-center gap-2 text-caption transition-colors",
              met ? "text-success" : "text-text-tertiary"
            )}
          >
            <span
              className={cn(
                "flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full",
                met ? "bg-success/10" : "bg-muted"
              )}
            >
              {met ? (
                <Check className="h-2.5 w-2.5" aria-hidden="true" />
              ) : (
                <X className="h-2.5 w-2.5" aria-hidden="true" />
              )}
            </span>
            {req.label}
          </li>
        );
      })}
    </ul>
  );
}

export { PasswordChecklist };
