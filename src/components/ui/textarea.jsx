import { cn } from "@/lib/utils";

function Textarea({ className, ...props }) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "flex min-h-20 w-full rounded-md border border-input bg-card px-3 py-2 text-body text-text-primary shadow-elev-1",
        "placeholder:text-text-tertiary transition-[color,box-shadow] outline-none",
        "focus-visible:border-focus focus-visible:ring-2 focus-visible:ring-focus/25",
        "aria-invalid:border-destructive aria-invalid:ring-destructive/25",
        "disabled:cursor-not-allowed disabled:opacity-50 md:text-small",
        className
      )}
      {...props}
    />
  );
}

export { Textarea };
