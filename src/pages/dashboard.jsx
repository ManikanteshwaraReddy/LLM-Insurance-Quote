import { NavLink } from "react-router-dom";
import { ArrowRight, Clock, MessageSquare, ShieldCheck, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { providers } from "@/data/providersData";

const STEPS = [
  {
    step: "1",
    title: "Answer Health Questions",
    description: "Provide details about your age, health history, lifestyle, and coverage preferences through a guided questionnaire.",
  },
  {
    step: "2",
    title: "Calculate Estimated Premiums",
    description: "Our system evaluates your health inputs and computes customized policy estimates across available coverage tiers.",
  },
  {
    step: "3",
    title: "Compare & Connect",
    description: "Review your policy breakdown, choose a payment schedule, and connect with registered insurance providers.",
  },
];

const Dashboard = () => {
  return (
    <div className="w-full bg-background">
      {/* ── Hero Section ─────────────────────────────────────────────────── */}
      <section className="border-b border-border bg-card py-16 md:py-24">
        <div className="container mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="grid items-center gap-10 md:grid-cols-2">
            <div className="space-y-6">
              <h1 className="text-display font-bold text-text-primary tracking-tight">
                Calculate health insurance quotes based on your health history.
              </h1>

              <p className="text-body-large text-text-secondary leading-relaxed">
                Complete a brief health assessment to receive personalized policy estimates and compare coverage options from leading insurance carriers.
              </p>

              <div className="flex flex-col sm:flex-row flex-wrap sm:items-center gap-4 pt-2">
                <NavLink to="/chat" className="w-full sm:w-auto">
                  <Button size="lg" className="gap-2 font-semibold text-white w-full sm:w-auto">
                    <span className="text-white">Start Health Assessment</span>
                    <ArrowRight className="h-4 w-4 text-white" aria-hidden="true" />
                  </Button>
                </NavLink>

                <NavLink to="/providers" className="w-full sm:w-auto">
                  <Button size="lg" variant="outline" className="w-full sm:w-auto">
                    Browse Insurance Providers
                  </Button>
                </NavLink>
              </div>

              <div className="pt-2 flex flex-wrap items-center gap-6 text-caption text-text-tertiary">
                <span className="flex items-center gap-1.5">
                  <ShieldCheck className="h-4 w-4 text-text-secondary" /> Confidential & Private
                </span>
                <span className="flex items-center gap-1.5">
                  <Clock className="h-4 w-4 text-text-secondary" /> 5-Minute Guided Process
                </span>
                <span className="flex items-center gap-1.5">
                  <MessageSquare className="h-4 w-4 text-text-secondary" /> Direct Policy Estimates
                </span>
              </div>
            </div>

            <div className="flex justify-center md:justify-end">
              <img
                src="/landing-image.png"
                alt="Health insurance illustration"
                className="h-auto w-full max-w-md object-contain"
              />
            </div>
          </div>
        </div>
      </section>

      {/* ── How It Works ─────────────────────────────────────────────────── */}
      <section className="py-16 md:py-20 border-b border-border">
        <div className="container mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <div className="mb-12 text-center">
            <h2 className="text-h1 font-bold text-text-primary tracking-tight">How It Works</h2>
            <p className="mt-1 text-body text-text-secondary">
              Three straightforward steps to estimate and compare your health coverage.
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            {STEPS.map((s) => (
              <div key={s.step} className="rounded-xl border border-border/80 bg-card p-6">
                <span className="inline-block text-label font-bold text-text-tertiary mb-2">Step {s.step}</span>
                <h3 className="text-h3 font-semibold text-text-primary mb-2">{s.title}</h3>
                <p className="text-small text-text-secondary leading-relaxed">{s.description}</p>
              </div>
            ))}
          </div>

          <div className="mt-10 text-center">
            <NavLink to="/chat">
              <Button className="gap-2">
                <span>Begin Assessment</span>
                <ArrowRight className="h-4 w-4" />
              </Button>
            </NavLink>
          </div>
        </div>
      </section>

      {/* ── Insurance Carriers Overview ────────────────────────────────────── */}
      <section className="py-16 bg-card">
        <div className="container mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 gap-4">
            <div>
              <h2 className="text-h2 font-bold text-text-primary tracking-tight">Partner Insurance Carriers</h2>
              <p className="mt-1 text-small text-text-secondary">
                Registered health insurance providers offering policy options.
              </p>
            </div>
            <NavLink to="/providers" className="w-full sm:w-auto">
              <Button variant="outline" size="sm" className="w-full sm:w-auto">
                View All Carriers
              </Button>
            </NavLink>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {providers.slice(0, 4).map((provider) => {
              const IconComponent = provider.icon;
              return (
                <div key={provider.id} className="rounded-lg border border-border p-4 bg-background flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      {IconComponent && <IconComponent className="h-4 w-4 text-primary shrink-0" />}
                      <h3 className="font-semibold text-text-primary text-small line-clamp-1" title={provider.title}>
                        {provider.title}
                      </h3>
                    </div>
                    <p className="text-caption text-text-secondary line-clamp-2 mb-3">
                      {provider.description}
                    </p>
                  </div>
                  <a
                    href={provider.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-caption text-primary hover:underline font-medium"
                  >
                    <span>Visit Website</span>
                    <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              );
            })}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
