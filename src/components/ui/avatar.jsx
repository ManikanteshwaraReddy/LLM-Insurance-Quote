import { cn } from "@/lib/utils";

const avatarSizes = {
  sm: "h-9 w-9 text-small",
  md: "h-12 w-12 text-h3",
  lg: "h-20 w-20 text-h1",
};

function Avatar({ className, size = "md", children, ...props }) {
  return (
    <div
      data-slot="avatar"
      aria-hidden="true"
      className={cn(
        "flex flex-shrink-0 select-none items-center justify-center rounded-full bg-primary font-semibold text-primary-foreground",
        avatarSizes[size],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export { Avatar };
