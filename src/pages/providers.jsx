import { useState } from "react";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/ui/page-header";
import { Input } from "@/components/ui/input";
import { ExternalLink, Search } from "lucide-react";
import { providers } from "@/data/providersData";

const Providers = () => {
  const [search, setSearch] = useState("");

  const filteredProviders = providers.filter((p) =>
    p.title.toLowerCase().includes(search.toLowerCase()) ||
    p.description.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="container mx-auto max-w-5xl px-4 py-12 sm:px-6 md:py-16">
      <PageHeader
        title="Insurance Providers Directory"
        description="Explore registered health insurance carriers. Compare coverage details and visit official websites."
      />

      {/* Filter / Search Bar */}
      <div className="mb-8 max-w-md relative">
        <Input
          type="text"
          placeholder="Search carriers..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-10"
        />
        <Search className="absolute left-3 top-3 h-4 w-4 text-text-tertiary" />
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredProviders.map((provider) => (
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
