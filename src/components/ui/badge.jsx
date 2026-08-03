/* eslint-disable react-refresh/only-export-components */
import { cn } from "@/lib/utils";

const badgeVariants = {
  neutral: "border border-border bg-muted/60 text-text-secondary font-medium",
  success: "border border-success/20 bg-success/10 text-success font-medium",
  warning: "border border-warning/20 bg-warning/10 text-warning font-medium",
  error: "border border-error/20 bg-error/10 text-error font-medium",
  primary: "border border-primary/20 bg-primary/10 text-primary font-medium",
  outline: "border border-border/80 bg-card text-text-secondary font-medium",
};

function Badge({ variant = "neutral", className, dot = false, children, ...props }) {
  return (
    <span
      data-slot="badge"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-caption transition-colors duration-150",
        badgeVariants[variant],
        className
      )}
      {...props}
    >
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current shrink-0" aria-hidden="true" />}
      {children}
    </span>
  );
}

export { Badge, badgeVariants };
