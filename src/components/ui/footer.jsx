import React from "react";
import { NavLink } from "react-router";

const Footer = () => {
  return (
    <footer className="bg-muted/30 border-t border-border pt-8 pb-4">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="md:text-lg font-semibold text-foreground mb-2 md:mb-4">
              SmartHealthQuote
            </h3>
            <p className="text-sm md:text-xl text-muted-foreground">
              Get insurance quotes tailored to your health history, instantly!
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4 col-span-1 md:col-span-2">
            <div>
              <h3 className="md:text-lg font-semibold text-foreground mb-2 md:mb-4">
                Quick Links
              </h3>
              <ul className="space-y-2">
                <li>
                  <NavLink
                    to="/"
                    className="text-sm md:text-base text-muted-foreground hover:text-primary transition-colors"
                  >
                    Home
                  </NavLink>
                </li>
                <li>
                  <NavLink
                    to="/chat"
                    className="text-sm md:text-base text-muted-foreground hover:text-primary transition-colors"
                  >
                    Get a Quote
                  </NavLink>
                </li>
                <li>
                  <NavLink
                    to="/providers"
                    className="text-sm md:text-base text-muted-foreground hover:text-primary transition-colors"
                  >
                    Insurance Providers
                  </NavLink>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="md:text-lg font-semibold text-foreground mb-2 md:mb-4">
                Legal
              </h3>
              <ul className="space-y-2">
                <li>
                  <NavLink
                    to="/privacy-policy"
                    className="text-sm md:text-base text-muted-foreground hover:text-primary transition-colors"
                  >
                    Privacy Policy
                  </NavLink>
                </li>
                <li>
                  <NavLink
                    to="/terms-of-service"
                    className="text-sm md:text-base text-muted-foreground hover:text-primary transition-colors"
                  >
                    Terms of Service
                  </NavLink>
                </li>
              </ul>
            </div>
          </div>
        </div>
        <div className="mt-4 pt-4 md:mt-8 md:pt-4 border-t border-border">
          <p className="text-sm md:text-lg text-muted-foreground text-center">
            Copyrights © SmartHealthQuote. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
