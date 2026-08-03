import { useEffect } from "react";
import { NavLink } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, ShieldCheck } from "lucide-react";

const PrivacyPolicy = () => {
  useEffect(() => {
    document.title = "Privacy Policy — SmartHealthQuote";
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
          <ShieldCheck className="h-4 w-4" /> Data Protection & Privacy
        </div>
        <h1 className="text-h1 font-bold text-text-primary tracking-tight mb-2">
          Privacy Policy
        </h1>
        <p className="text-small text-text-tertiary mb-8 pb-4 border-b border-border/60">
          Last Updated: May 8, 2025
        </p>

        <div className="space-y-8 leading-relaxed text-body text-text-primary">
          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              1. Introduction
            </h2>
            <p className="mb-3 text-text-secondary">
              Welcome to SmartHealthQuote. We respect your privacy and are
              committed to protecting your personal data. This privacy policy
              will inform you about how we look after your personal data when
              you visit our website and tell you about your privacy rights and
              how the law protects you.
            </p>
            <p className="text-text-secondary">
              This privacy policy aims to give you information on how
              SmartHealthQuote collects and processes your personal data
              through your use of this website, including any data you may
              provide when using our health quote assistant.
            </p>
          </section>

          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              2. Data We Collect
            </h2>
            <p className="mb-3 text-text-secondary">
              We may collect, use, store and transfer different kinds of
              personal data about you which we have grouped together as follows:
            </p>
            <ul className="list-disc space-y-2 pl-6 text-text-secondary">
              <li><strong>Identity Data:</strong> Name, age, gender, date of birth.</li>
              <li><strong>Contact Data:</strong> Email address, telephone number, city, and state.</li>
              <li><strong>Health & Medical Data:</strong> Pre-existing medical conditions, past medical history, family medical history, BMI.</li>
              <li><strong>Coverage Preference Data:</strong> Requested sum insured, policy term, and family coverage choices.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              3. How We Use Your Data
            </h2>
            <p className="mb-3 text-text-secondary">
              We will only use your personal data when the law allows us to. Most commonly, we will use your personal data to generate personalized health insurance estimates and match you with insurance carrier options.
            </p>
          </section>

          <section>
            <h2 className="text-h2 font-semibold text-text-primary mb-3">
              4. Data Security
            </h2>
            <p className="text-text-secondary">
              We have put in place appropriate security measures to prevent your personal data from being accidentally lost, used, or accessed in an unauthorized way, altered, or disclosed. All transmissions are encrypted using 256-bit SSL technology.
            </p>
          </section>
        </div>
      </Card>
    </div>
  );
};

export default PrivacyPolicy;
