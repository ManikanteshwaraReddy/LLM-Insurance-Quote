import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-md text-small font-medium transition-all duration-150 ease-out outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 disabled:pointer-events-none disabled:opacity-40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg]:size-4 cursor-pointer",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow-elev-1 hover:bg-primary-hover active:bg-primary-active",
        destructive:
          "bg-destructive text-destructive-foreground shadow-elev-1 hover:bg-destructive/90 active:bg-destructive",
        outline:
          "border border-border bg-card text-text-primary shadow-elev-1 hover:bg-muted hover:text-text-primary hover:border-border/80 active:bg-muted",
        secondary:
          "bg-secondary text-secondary-foreground shadow-elev-1 hover:bg-secondary-hover active:bg-secondary-active",
        ghost:
          "text-text-primary hover:bg-muted hover:text-text-primary active:bg-muted/80",
        link:
          "text-primary underline-offset-4 hover:underline p-0 h-auto font-normal",
      },
      size: {
        default: "h-10 px-4 has-[>svg]:px-3.5",
        sm: "h-8.5 gap-1.5 px-3 text-caption font-semibold has-[>svg]:px-2.5",
        lg: "h-11 px-5 text-body font-semibold has-[>svg]:px-4",
        icon: "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

function Button({ className, variant, size, asChild = false, ...props }) {
  const Comp = asChild ? Slot : "button";

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };
