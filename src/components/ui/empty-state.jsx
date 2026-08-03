import { cn } from "@/lib/utils";

function EmptyState({ icon: Icon, title, description, action, className }) {
  return (
    <div
      data-slot="empty-state"
      className={cn("flex flex-col items-center justify-center px-6 py-12 text-center", className)}
    >
      {Icon && (
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-muted text-text-tertiary">
          <Icon className="h-7 w-7" aria-hidden="true" />
        </div>
      )}
      <h3 className="text-h3 text-text-primary">{title}</h3>
      {description && (
        <p className="mt-1.5 max-w-sm text-small text-text-secondary">{description}</p>
      )}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export { EmptyState };
