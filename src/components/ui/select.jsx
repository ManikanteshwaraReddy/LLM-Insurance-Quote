import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

function Select({ className, children, ...props }) {
  return (
    <div className="relative">
      <select
        data-slot="select"
        className={cn(
          "h-10 w-full cursor-pointer appearance-none rounded-md border border-input bg-card pl-3 pr-9 text-body text-text-primary shadow-elev-1",
          "transition-[color,box-shadow] outline-none",
          "focus-visible:border-focus focus-visible:ring-2 focus-visible:ring-focus/25",
          "aria-invalid:border-destructive aria-invalid:ring-destructive/25",
          "disabled:cursor-not-allowed disabled:opacity-50 md:text-small",
          "[&>option]:bg-card [&>option]:text-text-primary",
          className
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown
        className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-tertiary"
        aria-hidden="true"
      />
    </div>
  );
}

export { Select };
