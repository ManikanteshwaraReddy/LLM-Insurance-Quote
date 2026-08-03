import { NavLink } from "react-router-dom";

const linkClass =
  "text-small text-text-secondary transition-colors duration-150 ease-out hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus rounded-sm";

const Footer = () => {
  return (
    <footer className="border-t border-border bg-card">
      <div className="container mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-12">
          {/* Brand Col */}
          <div className="md:col-span-5 space-y-2">
            <div className="flex items-center gap-2.5">
              <img
                src="/logo.svg"
                alt="SmartHealthQuote Logo"
                className="h-7 w-7 object-contain"
              />
              <span className="text-h3 font-bold tracking-tight text-text-primary">
                SmartHealthQuote
              </span>
            </div>
            <p className="max-w-sm text-small text-text-secondary leading-relaxed">
              Health insurance policy estimation and carrier comparison tool.
            </p>
          </div>

          {/* Links Grid */}
          <div className="grid grid-cols-2 gap-8 md:col-span-7 sm:grid-cols-3">
            <div>
              <h4 className="mb-2 text-caption text-text-tertiary font-semibold uppercase tracking-wider">
                Product
              </h4>
              <ul className="space-y-2">
                <li>
                  <NavLink to="/" className={linkClass}>
                    Home
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/chat" className={linkClass}>
                    Get a Quote
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/providers" className={linkClass}>
                    Carriers
                  </NavLink>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="mb-2 text-caption text-text-tertiary font-semibold uppercase tracking-wider">
                Account
              </h4>
              <ul className="space-y-2">
                <li>
                  <NavLink to="/auth" className={linkClass}>
                    Sign In
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/profile" className={linkClass}>
                    Profile
                  </NavLink>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="mb-2 text-caption text-text-tertiary font-semibold uppercase tracking-wider">
                Legal
              </h4>
              <ul className="space-y-2">
                <li>
                  <NavLink to="/privacy-policy" className={linkClass}>
                    Privacy Policy
                  </NavLink>
                </li>
                <li>
                  <NavLink to="/terms-of-service" className={linkClass}>
                    Terms of Service
                  </NavLink>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-8 border-t border-border/60 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-caption text-text-tertiary">
            © {new Date().getFullYear()} SmartHealthQuote. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
