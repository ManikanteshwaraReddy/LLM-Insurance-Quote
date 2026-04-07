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
        <h1 className="text-3xl font-bold text-blue-900 mb-4 dark:text-blue-400">
          Insurance Providers Directory
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto dark:text-gray-400">
          Browse our network of trusted insurance providers. Use your
          personalized quote to negotiate better rates.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {providers.map((provider) => (
          <Card
            key={provider.id}
            className="border-gray-200 hover:shadow-md transition-shadow dark:border-gray-800"
          >
            <CardHeader className="pb-2">
              <div className="h-16 flex items-center justify-center mb-2">
              <provider.icon className="w-10 h-10 text-blue-600" />
              </div>
              <CardTitle className="text-xl text-blue-900 dark:text-blue-400">
                {provider.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-gray-700 min-h-[80px] dark:text-gray-300">
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
                  className="w-full sm:w-auto border-blue-300 text-blue-700 hover:bg-blue-50 dark:border-blue-800 dark:text-blue-400 dark:hover:bg-blue-950"
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
