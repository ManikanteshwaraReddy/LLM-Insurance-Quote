import React from "react";
import { NavLink } from "react-router";

const Footer = () => {
  return (
    <footer className="bg-gray-50 border-t border-gray-200 pt-8 pb-4 dark:bg-gray-900 dark:border-gray-800">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="md:text-lg font-semibold text-gray-900 mb-2 md:mb-4 dark:text-white">
              SmartHealthQuote
            </h3>
            <p className="text-sm md:text-xl text-gray-600 dark:text-gray-400">
              Get insurance quotes tailored to your health history, instantly!
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4 col-span-1 md:col-span-2">
            <div>
              <h3 className="md:text-lg font-semibold text-gray-900 mb-2 md:mb-4 dark:text-white">
                Quick Links
              </h3>
              <ul className="space-y-2">
                <li>
                  <NavLink
                    to="/"
                    className="text-sm md:text-base text-gray-600 hover:text-blue-700 dark:text-gray-400 dark:hover:text-blue-400"
                  >
                    Home
                  </NavLink>
                </li>
                <li>
                  <NavLink
                    to="/chat"
                    className="text-sm md:text-base text-gray-600 hover:text-blue-700 dark:text-gray-400 dark:hover:text-blue-400"
                  >
                    Get a Quote
                  </NavLink>
                </li>
                <li>
                  <NavLink
                    to="/providers"
                    className="text-sm md:text-base text-gray-600 hover:text-blue-700 dark:text-gray-400 dark:hover:text-blue-400"
                  >
                    Insurance Providers
                  </NavLink>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="md:text-lg font-semibold text-gray-900 mb-2 md:mb-4 dark:text-white">
                Legal
              </h3>
              <ul className="space-y-2">
                <li>
                  <NavLink
                    to="/privacy-policy"
                    className="text-sm md:text-base text-gray-600 hover:text-blue-700 dark:text-gray-400 dark:hover:text-blue-400"
                  >
                    Privacy Policy
                  </NavLink>
                </li>
                <li>
                  <NavLink
                    to="/terms-of-service"
                    className="text-sm md:text-base text-gray-600 hover:text-blue-700 dark:text-gray-400 dark:hover:text-blue-400"
                  >
                    Terms of Service
                  </NavLink>
                </li>
              </ul>
            </div>
          </div>
        </div>
        <div className="mt-4 pt-4 md:mt-8 md:pt-4 border-t border-gray-200 dark:border-gray-800">
          <p className="text-sm md:text-lg text-gray-600 text-center dark:text-gray-400">
            Copyrights © SmartHealthQuote. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
