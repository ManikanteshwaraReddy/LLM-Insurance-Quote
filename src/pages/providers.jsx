import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";
import { providers } from "@/data/providersData";

const Providers = () => {
  return (
    <div className="container mx-auto max-w-7xl px-4 py-12">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold text-primary mb-4">
          Insurance Providers Directory
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Browse our network of trusted insurance providers. Use your
          personalized quote to negotiate better rates.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {providers.map((provider) => (
          <Card
            key={provider.id}
            className="border-border hover:shadow-md transition-shadow"
          >
            <CardHeader className="pb-2">
              <div className="h-16 flex items-center justify-center mb-2">
                <provider.icon className="w-10 h-10 text-secondary" />
              </div>
              <CardTitle className="text-xl text-primary">
                {provider.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-muted-foreground min-h-[80px]">
                {provider.description}
              </CardDescription>
            </CardContent>
            <CardFooter className="flex flex-col sm:flex-row gap-2">
              <a
                href={provider.website}
                target="_blank"
                rel="noopener noreferrer"
                className="w-full sm:w-auto"
              >
                <Button
                  variant="outline"
                  className="w-full sm:w-auto border-border text-foreground hover:bg-muted"
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Visit Website
                </Button>
              </a>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default Providers;
