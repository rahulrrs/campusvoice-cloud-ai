import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { 
  MessageSquarePlus, 
  Search, 
  Shield, 
  Zap, 
  Users, 
  CheckCircle2,
  ArrowRight,
  ClipboardList
} from "lucide-react";
import { Button } from "@/components/ui/button";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";

const features = [
  {
    icon: MessageSquarePlus,
    title: "Easy Submission",
    description: "Submit complaints in minutes with our intuitive form. Attach files and track progress easily.",
  },
  {
    icon: Search,
    title: "Real-time Tracking",
    description: "Monitor your complaint status 24/7. Get instant notifications on every update.",
  },
  {
    icon: Shield,
    title: "Secure & Private",
    description: "Your data is encrypted and protected. Anonymous submission options available.",
  },
  {
    icon: Zap,
    title: "Fast Resolution",
    description: "Our streamlined process ensures quick response times and efficient resolution.",
  },
];

const stats = [
  { value: "2,500+", label: "Complaints Resolved" },
  { value: "98%", label: "Satisfaction Rate" },
  { value: "24hr", label: "Avg Response Time" },
  { value: "15+", label: "Departments" },
];

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden gradient-hero">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-transparent to-transparent" />
        <div className="container mx-auto px-4 py-20 md:py-32 relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-3xl mx-auto text-center"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6"
            >
              <Users className="h-4 w-4" />
              Trusted by 10,000+ Students
            </motion.div>
            
            <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6 leading-tight">
              Your Voice Matters.{" "}
              <span className="text-primary">We Listen.</span>
            </h1>
            
            <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              The modern platform for students to submit, track, and resolve campus complaints efficiently. Make your campus better, one voice at a time.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/submit">
                <Button variant="hero" size="xl" className="w-full sm:w-auto">
                  <MessageSquarePlus className="h-5 w-5 mr-2" />
                  Submit a Complaint
                </Button>
              </Link>
              <Link to="/dashboard">
                <Button variant="outline" size="xl" className="w-full sm:w-auto">
                  <ClipboardList className="h-5 w-5 mr-2" />
                  Track My Complaints
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-b bg-card">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-primary mb-2">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Why Choose CampusVoice?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Built specifically for students, our platform makes it easy to raise concerns and get them resolved.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="group p-6 rounded-2xl bg-card border shadow-card hover:shadow-elevated transition-all duration-300"
              >
                <div className="h-12 w-12 rounded-xl gradient-primary flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <feature.icon className="h-6 w-6 text-primary-foreground" />
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-secondary/50">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              How It Works
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Three simple steps to get your concerns addressed
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {[
              { step: "01", title: "Submit", desc: "Fill out the complaint form with details about your concern" },
              { step: "02", title: "Track", desc: "Monitor your complaint status in real-time on your dashboard" },
              { step: "03", title: "Resolve", desc: "Get notified when your complaint is addressed and resolved" },
            ].map((item, index) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.15 }}
                className="relative text-center"
              >
                <div className="text-6xl font-bold text-primary/10 mb-4">
                  {item.step}
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-2">
                  {item.title}
                </h3>
                <p className="text-muted-foreground">{item.desc}</p>
                {index < 2 && (
                  <ArrowRight className="hidden md:block absolute top-12 -right-4 h-6 w-6 text-primary/30" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto rounded-3xl gradient-primary p-8 md:p-12 text-center relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-white/10 via-transparent to-transparent" />
            <div className="relative">
              <CheckCircle2 className="h-12 w-12 text-primary-foreground mx-auto mb-6" />
              <h2 className="text-2xl md:text-4xl font-bold text-primary-foreground mb-4">
                Ready to Be Heard?
              </h2>
              <p className="text-lg text-primary-foreground/80 mb-8 max-w-xl mx-auto">
                Join thousands of students who have successfully resolved their concerns through CampusVoice.
              </p>
              <Link to="/submit">
                <Button 
                  size="xl" 
                  className="bg-primary-foreground text-primary hover:bg-primary-foreground/90"
                >
                  Submit Your First Complaint
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
