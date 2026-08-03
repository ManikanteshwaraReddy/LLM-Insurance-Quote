import { NavLink } from "react-router-dom";
import { ArrowRight, Clock, MessageSquare, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";

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

const CARRIERS = [
  { name: "HDFC ERGO Health", category: "Comprehensive & Super Top-up" },
  { name: "Star Health Insurance", category: "Family Floater & Senior Care" },
  { name: "Care Health Insurance", category: "Critical Illness & Global Care" },
  { name: "Niva Bupa Health", category: "Individual & Maternity Cover" },
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

              <div className="flex flex-wrap items-center gap-4 pt-2">
                <NavLink to="/chat">
                  <Button size="lg" className="gap-2 font-semibold text-white">
                    <span className="text-white">Start Health Assessment</span>
                    <ArrowRight className="h-4 w-4 text-white" aria-hidden="true" />
                  </Button>
                </NavLink>

                <NavLink to="/providers">
                  <Button size="lg" variant="outline">
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
            <NavLink to="/providers">
              <Button variant="outline" size="sm">
                View All Carriers
              </Button>
            </NavLink>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {CARRIERS.map((carrier) => (
              <div key={carrier.name} className="rounded-lg border border-border p-4 bg-background">
                <h3 className="font-semibold text-text-primary text-small mb-1">{carrier.name}</h3>
                <p className="text-caption text-text-secondary">{carrier.category}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
