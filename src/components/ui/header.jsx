import { useEffect, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { LogIn, LogOut, Menu, User, X, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/AuthContext";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Home" },
  { to: "/providers", label: "Insurance Providers" },
];

export default function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  // Close mobile menu on Escape
  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isOpen]);

  const handleLogout = async () => {
    setIsOpen(false);
    await logout();
    navigate("/");
  };

  const desktopNavLinkClass = ({ isActive }) =>
    cn(
      "rounded-lg px-3.5 py-2 text-small font-medium transition-all duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus",
      isActive
        ? "bg-muted text-text-primary font-semibold"
        : "text-text-secondary hover:bg-muted/60 hover:text-text-primary"
    );

  const mobileNavLinkClass = ({ isActive }) =>
    cn(
      "rounded-lg px-4 py-2.5 text-small font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus",
      isActive
        ? "bg-muted text-text-primary font-semibold"
        : "text-text-secondary hover:bg-muted/60 hover:text-text-primary"
    );

  return (
    <header className="sticky top-0 z-40 border-b border-border/80 bg-card/95 backdrop-blur-md">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-header items-center justify-between gap-4">
          {/* Brand Logo */}
          <NavLink
            to="/"
            aria-label="SmartHealthQuote home"
            className="flex min-w-0 items-center gap-3 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-card p-1 border border-border/60 shadow-elev-1 shrink-0">
              <img
                src="/logo.svg"
                alt="SmartHealthQuote Logo"
                className="h-full w-full object-contain"
              />
            </div>
            <span className="truncate text-body font-bold tracking-tight text-text-primary md:text-h3">
              SmartHealthQuote
            </span>
          </NavLink>

          {/* Desktop Navigation */}
          <nav className="hidden items-center gap-1.5 md:flex" aria-label="Main navigation">
            {NAV_ITEMS.map((item) => (
              <NavLink key={item.to} to={item.to} className={desktopNavLinkClass}>
                {item.label}
              </NavLink>
            ))}

            <NavLink to="/chat" className="ml-2">
              <Button size="sm" className="gap-1.5 text-white font-semibold shadow-elev-1">
                <span>Get Quote</span>
              </Button>
            </NavLink>

            <div className="mx-2 h-5 w-px bg-border/60" aria-hidden="true" />

            <DesktopAuth
              isAuthenticated={isAuthenticated}
              user={user}
              onLogout={handleLogout}
            />
          </nav>

          {/* Mobile Menu Controls */}
          <div className="flex items-center gap-2 md:hidden">
            <NavLink to="/chat">
              <Button size="sm" className="text-white text-caption font-semibold">
                Get Quote
              </Button>
            </NavLink>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsOpen((o) => !o)}
              aria-label={isOpen ? "Close menu" : "Open menu"}
              aria-expanded={isOpen}
              aria-controls="mobile-menu"
            >
              {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {isOpen && (
        <div id="mobile-menu" className="border-t border-border bg-card px-4 pb-6 pt-4 md:hidden">
          <nav className="flex flex-col gap-1.5" aria-label="Mobile navigation">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={mobileNavLinkClass}
                onClick={() => setIsOpen(false)}
              >
                {item.label}
              </NavLink>
            ))}
            <div className="my-2 border-t border-border/60" />
            <MobileAuth
              isAuthenticated={isAuthenticated}
              user={user}
              onLogout={handleLogout}
              onClose={() => setIsOpen(false)}
            />
          </nav>
        </div>
      )}
    </header>
  );
}

function DesktopAuth({ isAuthenticated, user, onLogout }) {
  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-2">
        <NavLink
          to="/profile"
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-small font-medium text-text-secondary transition-colors duration-150 ease-out hover:bg-muted hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <User className="h-4 w-4 text-text-tertiary" aria-hidden="true" />
          <span>{user?.fullName?.split(" ")[0]}</span>
        </NavLink>
        <Button variant="outline" size="sm" onClick={onLogout} className="gap-1.5 text-caption">
          <LogOut className="h-3.5 w-3.5" aria-hidden="true" />
          Sign Out
        </Button>
      </div>
    );
  }
  return (
    <NavLink to="/auth">
      <Button variant="outline" size="sm" className="gap-1.5">
        <LogIn className="h-3.5 w-3.5" aria-hidden="true" />
        Sign In
      </Button>
    </NavLink>
  );
}

function MobileAuth({ isAuthenticated, user, onLogout, onClose }) {
  if (isAuthenticated) {
    return (
      <>
        <NavLink
          to="/profile"
          onClick={onClose}
          className="flex items-center gap-2 rounded-lg px-4 py-2.5 text-small font-medium text-text-primary hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <User className="h-4 w-4 text-text-tertiary" aria-hidden="true" />
          <span>{user?.fullName}</span>
        </NavLink>
        <Button variant="outline" className="mt-2 w-full gap-2" onClick={onLogout}>
          <LogOut className="h-4 w-4" aria-hidden="true" />
          Sign Out
        </Button>
      </>
    );
  }
  return (
    <NavLink to="/auth" onClick={onClose} className="block">
      <Button variant="outline" className="w-full gap-2">
        <LogIn className="h-4 w-4" aria-hidden="true" />
        Sign In
      </Button>
    </NavLink>
  );
}
