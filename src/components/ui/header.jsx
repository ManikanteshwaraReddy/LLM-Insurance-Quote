import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Menu, X, LogIn } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Header() {
  const [isOpen, setIsOpen] = useState(false);

  const navLinkClasses = ({ isActive }) =>
    `text-sm font-medium transition-colors hover:text-blue-600 ${
      isActive
        ? "text-blue-600 dark:text-blue-400"
        : "text-gray-700 dark:text-gray-300"
    }`;

  return (
    <header className="border-b border-gray-200 bg-white dark:bg-gray-950">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <img src="/logo.svg" alt="Logo" className="h-10 md:h-16 md:w-16" />
            <NavLink to="/" className="flex items-center">
              <span className="font-bold text-blue-700 dark:text-blue-400 md:text-2xl">
                SmartHealthQuote
              </span>
            </NavLink>
          </div>

          <nav className="hidden md:flex items-center space-x-8">
            <NavLink to="/" className={navLinkClasses}>
              Home
            </NavLink>
            <NavLink to="/providers" className={navLinkClasses}>
              Providers
            </NavLink>
            <NavLink to="/chat">
              {({ isActive }) => (
                <Button
                  className={`${isActive
                      ? "bg-green-700"
                      : "bg-green-600 hover:bg-green-700"
                    } text-white`}
                >
                  Start Chat
                </Button>
              )}
            </NavLink>
            <NavLink to="/auth">
              <Button variant="outline" size="sm">
                <LogIn className="mr-2 h-4 w-4" />
                Login / Sign Up
              </Button>
            </NavLink>
          </nav>

          <div className="md:hidden">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsOpen(!isOpen)}
              aria-label="Toggle menu"
            >
              {isOpen ? (
                <X className="h-5 w-5" />
              ) : (
                <Menu className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>

        {isOpen && (
          <nav className="md:hidden mt-4 space-y-2 border-t border-gray-200 dark:border-gray-800 pt-2">
            <div className="border-b text-center p-2 align-middle">
              <NavLink
                to="/"
                className={navLinkClasses}
                onClick={() => setIsOpen(false)}
              >
                Home
              </NavLink>
            </div>
            <div className="border-b text-center p-2 align-middle">
              <NavLink
                to="/providers"
                className={navLinkClasses}
                onClick={() => setIsOpen(false)}
              >
                Providers
              </NavLink>
            </div>
            <div className="border-b text-center p-2 align-middle">
              <NavLink to="/chat" onClick={() => setIsOpen(false)}>
                {({ isActive }) => (
                  <Button
                    className={`w-full ${
                      isActive
                        ? "bg-green-700"
                        : "bg-green-600 hover:bg-green-700"
                    } text-white`}
                  >
                    Start Chat
                  </Button>
                )}
              </NavLink>
            </div>
            <div className="text-center p-2 align-middle">
              <NavLink to="/auth" onClick={() => setIsOpen(false)}>
                 <Button variant="outline" className="w-full">
                    <LogIn className="mr-2 h-4 w-4" />
                    Login / Sign Up
                  </Button>
              </NavLink>
            </div>
          </nav>
        )}
      </div>
    </header>
  );
}
