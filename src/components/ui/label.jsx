import { cn } from "@/lib/utils";

function Label({ className, required = false, children, ...props }) {
  return (
    <label
      data-slot="label"
      className={cn("mb-1.5 block text-label font-medium text-text-primary", className)}
      {...props}
    >
      {children}
      {required && <span className="text-error"> *</span>}
    </label>
  );
}

export { Label };
