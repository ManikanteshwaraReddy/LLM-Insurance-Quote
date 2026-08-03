import { cn } from "@/lib/utils";

function PageHeader({ title, description, action, align = "center", className }) {
  return (
    <div
      data-slot="page-header"
      className={cn("mb-8 md:mb-12", align === "center" ? "text-center" : "text-left", className)}
    >
      <h1 className="text-h1 text-text-primary">{title}</h1>
      {description && (
        <p
          className={cn(
            "mt-2.5 text-body-large text-text-secondary",
            align === "center" && "mx-auto max-w-2xl"
          )}
        >
          {description}
        </p>
      )}
      {action && (
        <div className={cn("mt-6", align === "center" && "flex justify-center")}>{action}</div>
      )}
    </div>
  );
}

export { PageHeader };
