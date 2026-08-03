import * as React from "react";
import { cn } from "@/lib/utils";

function Input({ className, type, ...props }) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "flex h-10 w-full min-w-0 rounded-md border border-input bg-card px-3.5 py-2 text-small text-text-primary placeholder:text-text-tertiary shadow-elev-1 transition-all duration-150 ease-out outline-none focus-visible:border-focus focus-visible:ring-2 focus-visible:ring-focus/30 aria-invalid:border-error aria-invalid:ring-error/20 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  );
}

export { Input };
