import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { useAuth } from "@/lib/AuthContext";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PasswordInput } from "@/components/ui/password-input";
import {
  PasswordChecklist,
  PASSWORD_REQUIREMENTS,
} from "@/components/ui/password-checklist";
import { cn } from "@/lib/utils";

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [globalError, setGlobalError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showChecklist, setShowChecklist] = useState(false);

  const { login, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = location.state?.from?.pathname || "/";

  const switchMode = () => {
    setIsLogin((v) => !v);
    setGlobalError("");
    setFieldErrors({});
    setEmail("");
    setPassword("");
    setShowChecklist(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setGlobalError("");
    setFieldErrors({});
    setIsLoading(true);

    const fd = new FormData(e.target);
    const emailVal = (fd.get("email") || email || "").trim();
    const pwdVal = fd.get("password") || password || "";
    const nameVal = (fd.get("fullName") || "").trim();
    const confirmPwdVal = fd.get("confirm-password") || "";

    const errs = {};
    if (!isLogin && !nameVal) errs.fullName = "Full name is required.";
    if (!emailVal) errs.email = "Email is required.";
    if (!pwdVal) errs.password = "Password is required.";
    if (!isLogin && pwdVal !== confirmPwdVal)
      errs.confirmPassword = "Passwords do not match.";
    if (!isLogin && pwdVal && PASSWORD_REQUIREMENTS.some((r) => !r.test(pwdVal))) {
      errs.password = "Password does not meet all requirements.";
    }

    if (Object.keys(errs).length > 0) {
      setFieldErrors(errs);
      setIsLoading(false);
      return;
    }

    try {
      if (isLogin) {
        await login(emailVal, pwdVal);
      } else {
        await register(nameVal, emailVal, pwdVal);
      }
      navigate(from, { replace: true });
    } catch (err) {
      const code = err.code;
      if (code === "EMAIL_TAKEN")
        setFieldErrors({ email: "An account with this email already exists." });
      else if (code === "ACCOUNT_LOCKED")
        setGlobalError(
          "Account temporarily locked. Too many failed attempts. Try again in 15 minutes."
        );
      else if (code === "ACCOUNT_SUSPENDED")
        setGlobalError("Your account has been suspended. Please contact support.");
      else setGlobalError(err.message || "Something went wrong. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const errorId = (key) => (fieldErrors[key] ? `${key}-error` : undefined);

  return (
    <div className="flex min-h-dvh items-center justify-center bg-background px-4 py-12">
      <div className="w-full max-w-md">
        {/* Header / Brand */}
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-card border border-border p-3">
            <img
              src="/logo.svg"
              alt="SmartHealthQuote Logo"
              className="h-full w-full object-contain"
            />
          </div>
          <h1 className="text-h1 font-bold tracking-tight text-text-primary">
            SmartHealthQuote
          </h1>
          <p className="mt-1 text-small text-text-secondary">
            {isLogin ? "Sign in to your account" : "Create your free account"}
          </p>
        </div>

        {/* Card */}
        <div className="rounded-xl border border-border bg-card p-6 sm:p-8">
          {/* Mode Switcher */}
          <div className="mb-6 grid grid-cols-2 gap-1 rounded-lg bg-muted p-1">
            {[
              {
                key: "login",
                label: "Sign In",
                active: isLogin,
                onClick: () => !isLogin && switchMode(),
              },
              {
                key: "signup",
                label: "Sign Up",
                active: !isLogin,
                onClick: () => isLogin && switchMode(),
              },
            ].map((tab) => (
              <button
                key={tab.key}
                type="button"
                aria-pressed={tab.active}
                onClick={tab.onClick}
                className={cn(
                  "h-9 rounded-md text-small font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                  tab.active
                    ? "bg-card text-text-primary shadow-elev-1 font-semibold"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {globalError && (
            <Alert
              type="error"
              message={globalError}
              onClose={() => setGlobalError("")}
              className="mb-5"
            />
          )}

          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            {/* Full Name (Register Only) */}
            {!isLogin && (
              <div>
                <Label htmlFor="fullName" required>
                  Full Name
                </Label>
                <Input
                  id="fullName"
                  name="fullName"
                  type="text"
                  autoComplete="name"
                  placeholder="Jane Doe"
                  disabled={isLoading}
                  aria-invalid={!!fieldErrors.fullName || undefined}
                  aria-describedby={errorId("fullName")}
                />
                {fieldErrors.fullName && (
                  <p id="fullName-error" className="mt-1.5 text-caption text-error">
                    {fieldErrors.fullName}
                  </p>
                )}
              </div>
            )}

            {/* Email Address */}
            <div>
              <Label htmlFor="email" required>
                Email address
              </Label>
              <Input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
                aria-invalid={!!fieldErrors.email || undefined}
                aria-describedby={errorId("email")}
              />
              {fieldErrors.email && (
                <p id="email-error" className="mt-1.5 text-caption text-error">
                  {fieldErrors.email}
                </p>
              )}
            </div>

            {/* Password */}
            <div>
              <Label htmlFor="password" required>
                Password
              </Label>
              <PasswordInput
                id="password"
                name="password"
                autoComplete={isLogin ? "current-password" : "new-password"}
                placeholder="••••••••"
                disabled={isLoading}
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  if (!isLogin) setShowChecklist(true);
                }}
                aria-invalid={!!fieldErrors.password || undefined}
                aria-describedby={errorId("password")}
              />
              {fieldErrors.password && (
                <p id="password-error" className="mt-1.5 text-caption text-error">
                  {fieldErrors.password}
                </p>
              )}
              {!isLogin && showChecklist && <PasswordChecklist password={password} />}
            </div>

            {/* Confirm Password (Register Only) */}
            {!isLogin && (
              <div>
                <Label htmlFor="confirm-password" required>
                  Confirm Password
                </Label>
                <PasswordInput
                  id="confirm-password"
                  name="confirm-password"
                  autoComplete="new-password"
                  placeholder="••••••••"
                  disabled={isLoading}
                  aria-invalid={!!fieldErrors.confirmPassword || undefined}
                  aria-describedby={errorId("confirmPassword")}
                />
                {fieldErrors.confirmPassword && (
                  <p
                    id="confirmPassword-error"
                    className="mt-1.5 text-caption text-error"
                  >
                    {fieldErrors.confirmPassword}
                  </p>
                )}
              </div>
            )}

            {/* Submit */}
            <Button
              type="submit"
              size="lg"
              className="mt-2 w-full gap-2 font-semibold"
              disabled={isLoading}
            >
              {isLoading ? (
                <span>{isLogin ? "Signing in…" : "Creating account…"}</span>
              ) : (
                <>
                  <span>{isLogin ? "Sign In" : "Create Account"}</span>
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
