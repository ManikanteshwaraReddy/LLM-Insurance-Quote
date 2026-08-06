import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/ui/page-header";
import { ExternalLink } from "lucide-react";
import { providers } from "@/data/providersData";

const Providers = () => {
  return (
    <div className="container mx-auto max-w-5xl px-4 py-12 sm:px-6 md:py-16">
      <PageHeader
        title="Insurance Providers Directory"
        description="Explore registered health insurance carriers. Compare coverage details and visit official websites."
      />

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {providers.map((provider) => (
          <div
            key={provider.id}
            className="flex flex-col justify-between rounded-xl border border-border bg-card p-6"
          >
            <div>
              <h3 className="text-h3 font-semibold text-text-primary mb-2">
                {provider.title}
              </h3>
              <p className="text-small text-text-secondary leading-relaxed">
                {provider.description}
              </p>
            </div>

            <div className="mt-6 pt-4 border-t border-border">
              <a
                href={provider.website}
                target="_blank"
                rel="noopener noreferrer"
                className="w-full block"
              >
                <Button variant="outline" className="w-full gap-2 text-small">
                  <span>Visit Website</span>
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                </Button>
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Providers;
