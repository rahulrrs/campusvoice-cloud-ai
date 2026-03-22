import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BellRing,
  CheckCircle2,
  Clock3,
  MessageSquarePlus,
  Shield,
  Sparkles,
  Waypoints,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";

const pillars = [
  {
    title: "Speak up without friction",
    description: "Submit a complaint quickly, attach evidence, and choose anonymous reporting when needed.",
    icon: MessageSquarePlus,
  },
  {
    title: "Track every movement",
    description: "See status changes, replies, and resolution progress instead of wondering what happened next.",
    icon: Waypoints,
  },
  {
    title: "Escalate the right cases",
    description: "Urgent issues surface faster so safety, harassment, and service breakdowns are easier to act on.",
    icon: Shield,
  },
];

const stats = [
  { label: "Submission paths", value: "Text + files + voice" },
  { label: "Tracking states", value: "4 structured stages" },
  { label: "Reporting mode", value: "Anonymous by default" },
];

const Index = () => {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />

      <main className="space-y-16 pb-8 pt-6 md:space-y-20">
        <section className="section-shell">
          <div className="hero-frame mesh-grid px-6 py-10 md:px-10 md:py-12">
            <div className="absolute inset-0 gradient-hero opacity-90" />
            <div className="relative grid gap-10 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div className="inline-flex items-center gap-2 rounded-full border border-slate-900/10 bg-white/80 px-4 py-2 text-sm text-slate-700 shadow-card">
                  <Sparkles className="h-4 w-4 text-primary" />
                  Complaint handling that feels clear, modern, and accountable
                </div>

                <div className="space-y-4">
                  <h1 className="heading-display max-w-3xl text-4xl font-bold leading-[0.96] md:text-6xl">
                    A clearer way for students to report, track, and resolve campus issues.
                  </h1>
                  <p className="max-w-2xl text-base leading-7 text-muted-foreground md:text-lg">
                    One place to submit complaints, follow progress, and receive updates without confusion.
                  </p>
                </div>

                <div className="flex flex-col gap-3 sm:flex-row">
                  <Button asChild variant="hero" size="xl" className="rounded-full px-8">
                    <Link to="/submit">
                      Submit a Complaint
                      <ArrowRight className="h-5 w-5" />
                    </Link>
                  </Button>
                  <Button asChild variant="outline" size="xl" className="rounded-full bg-white/85 px-8">
                    <Link to="/dashboard">
                      Track My Complaints
                    </Link>
                  </Button>
                </div>

                <div className="grid gap-3 md:grid-cols-3">
                  {stats.map((item) => (
                    <div key={item.label} className="surface-soft px-4 py-4">
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">{item.label}</p>
                      <p className="mt-2 text-base font-semibold text-foreground">{item.value}</p>
                    </div>
                  ))}
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="surface-panel relative overflow-hidden p-7"
              >
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(125,211,252,0.22),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(251,191,36,0.14),transparent_24%)]" />
                <div className="relative space-y-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Live system snapshot</p>
                      <h2 className="mt-2 heading-display text-2xl font-semibold text-white">What students get</h2>
                    </div>
                    <div className="rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs text-slate-200">
                      Real-time workflow
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-sky-400/15 text-sky-200">
                          <Clock3 className="h-5 w-5" />
                        </div>
                        <div>
                          <p className="font-medium text-white">Structured complaint timeline</p>
                          <p className="text-sm text-slate-300">Submitted, pending, in progress, resolved.</p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-400/15 text-emerald-200">
                          <BellRing className="h-5 w-5" />
                        </div>
                        <div>
                          <p className="font-medium text-white">Clear notifications</p>
                          <p className="text-sm text-slate-300">Unread updates, assignment changes, and resolution alerts.</p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-amber-300/15 text-amber-100">
                          <CheckCircle2 className="h-5 w-5" />
                        </div>
                        <div>
                          <p className="font-medium text-white">Anonymous reporting</p>
                          <p className="text-sm text-slate-300">Safer reporting by default, with identity reveal left optional.</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        <section className="section-shell">
          <div className="grid gap-5 lg:grid-cols-3">
            {pillars.map((pillar, index) => (
              <motion.div
                key={pillar.title}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="surface-card p-7"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                  <pillar.icon className="h-6 w-6" />
                </div>
                <h2 className="mt-5 heading-display text-2xl font-semibold">{pillar.title}</h2>
                <p className="mt-3 text-sm leading-7 text-muted-foreground">{pillar.description}</p>
              </motion.div>
            ))}
          </div>
        </section>

        <section className="section-shell">
          <div className="surface-panel px-6 py-8 md:px-10">
            <div className="grid gap-8 lg:grid-cols-[1fr_auto] lg:items-center">
              <div className="space-y-3">
                <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-200">Ready to start</p>
                <h2 className="heading-display text-3xl font-bold text-white md:text-4xl">
                  Give students one place to be heard.
                </h2>
                <p className="max-w-2xl text-base leading-8 text-slate-300">
                  Submit a complaint, track the status, and follow updates in one clear workflow.
                </p>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row lg:flex-col">
                <Button asChild variant="hero" size="lg" className="rounded-full bg-white text-slate-950 hover:bg-white/90">
                  <Link to="/submit">Submit a Complaint</Link>
                </Button>
                <Button asChild variant="secondary" size="lg" className="rounded-full border border-white/15 bg-white/10 text-white hover:bg-white/15">
                  <Link to="/faq">Read FAQs</Link>
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Index;
