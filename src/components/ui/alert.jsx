import { AlertCircle, AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

const alertStyles = {
  error: { className: "border-error/30 bg-error/10 text-error", Icon: AlertCircle },
  success: { className: "border-success/30 bg-success/10 text-success", Icon: CheckCircle2 },
  warning: { className: "border-warning/30 bg-warning/10 text-warning", Icon: AlertTriangle },
  info: { className: "border-primary/20 bg-primary/5 text-text-primary", Icon: Info },
};

function Alert({ type = "info", title, message, children, onClose, className }) {
  const { className: styles, Icon } = alertStyles[type] || alertStyles.info;

  if (!message && !children && !title) return null;

  return (
    <div
      role={type === "error" ? "alert" : "status"}
      className={cn(
        "flex items-start gap-2.5 rounded-md border px-4 py-3 text-small",
        styles,
        className
      )}
    >
      <Icon className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1 space-y-1">
        {title && <p className="font-semibold">{title}</p>}
        {message && <p>{message}</p>}
        {children}
      </div>
      {onClose && (
        <button
          type="button"
          onClick={onClose}
          aria-label="Dismiss"
          className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md opacity-60 transition-opacity hover:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

export { Alert };
