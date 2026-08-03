import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Menu, X, LogIn } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Header() {
  const [isOpen, setIsOpen] = useState(false);

  const navLinkClasses = ({ isActive }) =>
    `text-sm font-medium transition-colors hover:text-primary ${
      isActive
        ? "text-primary"
        : "text-muted-foreground"
    }`;

  return (
    <header className="border-b border-border bg-card">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <img src="/logo.svg" alt="Logo" className="h-10 md:h-16 md:w-16" />
            <NavLink to="/" className="flex items-center">
              <span className="font-bold text-primary md:text-2xl">
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
                      ? "bg-secondary/90"
                      : "bg-secondary hover:bg-secondary/80"
                    } text-primary-foreground`}
                >
                  Start Chat
                </Button>
              )}
            </NavLink>
            <NavLink to="/auth">
              <Button variant="outline" size="sm" className="border-border">
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
      </div>

      {isOpen && (
        <div className="md:hidden border-t border-border bg-card px-4 py-4 space-y-3">
          <nav className="flex flex-col space-y-3">
            <NavLink
              to="/"
              className={navLinkClasses}
              onClick={() => setIsOpen(false)}
            >
              Home
            </NavLink>
            <NavLink
              to="/providers"
              className={navLinkClasses}
              onClick={() => setIsOpen(false)}
            >
              Providers
            </NavLink>
            <NavLink to="/chat" onClick={() => setIsOpen(false)}>
              {({ isActive }) => (
                <Button
                  className={`w-full ${
                    isActive
                      ? "bg-secondary/90"
                      : "bg-secondary hover:bg-secondary/80"
                  } text-primary-foreground`}
                >
                  Start Chat
                </Button>
              )}
            </NavLink>
            <NavLink to="/auth" onClick={() => setIsOpen(false)}>
               <Button variant="outline" className="w-full border-border">
                  <LogIn className="mr-2 h-4 w-4" />
                  Login / Sign Up
                </Button>
            </NavLink>
          </nav>
        </div>
      )}
    </header>
  );
}
