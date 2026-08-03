import { useEffect } from "react";
import { NavLink } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, FileText } from "lucide-react";

const TermsOfService = () => {
  useEffect(() => {
    document.title = "Terms of Service — SmartHealthQuote";
  }, []);

  return (
    <div className="container mx-auto max-w-4xl px-4 py-12 md:py-16">
      <div className="mb-6">
        <NavLink to="/">
          <Button variant="ghost" size="sm" className="gap-1.5 text-text-secondary hover:text-text-primary">
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Back to Home
          </Button>
        </NavLink>
      </div>

      <Card className="p-6 md:p-10 border border-border/80 bg-card shadow-elev-1">
        <div className="flex items-center gap-2 mb-2 text-caption font-semibold text-primary uppercase tracking-wider">
          <FileText className="h-4 w-4" /> Legal & Terms
        </div>
        <h1 className="text-h1 font-bold text-text-primary tracking-tight mb-2">
          Terms of Service
        </h1>
        <p className="text-small text-text-tertiary mb-8 pb-4 border-b border-border/60">
          Last Updated: May 8, 2025
        </p>

        <div className="space-y-8 leading-relaxed text-body text-text-primary">
          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              1. Agreement to Terms
            </h2>
            <p className="text-text-secondary">
              By accessing or using SmartHealthQuote, you agree to be bound by these Terms of Service and our Privacy Policy. If you do not agree with any part of these terms, you may not use our service.
            </p>
          </section>

          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              2. Nature of Service
            </h2>
            <p className="text-text-secondary mb-3">
              SmartHealthQuote provides informational health insurance policy estimates and carrier matching services. SmartHealthQuote is an independent calculation tool and does not issue insurance policies directly.
            </p>
            <p className="text-text-secondary">
              Final policy issuance, premium rates, and coverage terms remain subject to medical underwriting and approval by IRDAI-registered insurance providers.
            </p>
          </section>

          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              3. User Responsibilities
            </h2>
            <p className="text-text-secondary">
              You agree to provide accurate, current, and complete information during the quote assessment process. Inaccurate information may affect the validity of estimated quotes.
            </p>
          </section>

          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              4. Limitation of Liability
            </h2>
            <p className="text-text-secondary">
              In no event shall SmartHealthQuote or its affiliates be liable for any indirect, incidental, or consequential damages resulting from the use or inability to use our quote calculation service.
            </p>
          </section>
        </div>
      </Card>
    </div>
  );
};

export default TermsOfService;
