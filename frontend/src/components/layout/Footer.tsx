import { Link } from "react-router-dom";
import { Mail, MapPin, MessageSquare, Phone } from "lucide-react";

const Footer = () => {
  return (
    <footer className="mt-12 px-4 pb-6 pt-4">
      <div className="container mx-auto rounded-[32px] border border-white/80 bg-white/82 px-6 py-12 shadow-card backdrop-blur-md">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-[1.4fr_0.9fr_1fr_1fr]">
          <div className="space-y-5">
            <Link to="/" className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl gradient-primary shadow-card">
                <MessageSquare className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <span className="heading-display text-xl font-bold text-foreground">CampusVoice</span>
                <p className="text-xs text-muted-foreground">Campus complaints, minus the confusion.</p>
              </div>
            </Link>
            <p className="max-w-md text-sm leading-6 text-muted-foreground">
              Empowering students to voice concerns, track progress, and get support from the right campus team.
            </p>
            <div className="surface-soft inline-flex items-center gap-3 px-4 py-3 text-sm text-slate-700">
              <div className="h-2.5 w-2.5 rounded-full bg-primary shadow-[0_0_0_6px_rgba(59,130,246,0.14)]" />
              Anonymous reporting, live tracking, and admin escalation in one place.
            </div>
          </div>

          <div>
            <h4 className="mb-4 heading-display text-base font-semibold text-foreground">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/submit" className="text-sm text-muted-foreground transition-colors hover:text-primary">
                  Submit Complaint
                </Link>
              </li>
              <li>
                <Link to="/dashboard" className="text-sm text-muted-foreground transition-colors hover:text-primary">
                  Track Status
                </Link>
              </li>
              <li>
                <Link to="/notifications" className="text-sm text-muted-foreground transition-colors hover:text-primary">
                  Notifications
                </Link>
              </li>
              <li>
                <Link to="/faq" className="text-sm text-muted-foreground transition-colors hover:text-primary">
                  FAQs
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="mb-4 heading-display text-base font-semibold text-foreground">Complaint Areas</h4>
            <ul className="space-y-2">
              <li className="text-sm text-muted-foreground">Academic & Attendance</li>
              <li className="text-sm text-muted-foreground">Hostel, Infrastructure & Library</li>
              <li className="text-sm text-muted-foreground">Fees, Records & Placement</li>
              <li className="text-sm text-muted-foreground">Safety, Harassment & IT Services</li>
            </ul>
          </div>

          <div>
            <h4 className="mb-4 heading-display text-base font-semibold text-foreground">Contact Us</h4>
            <ul className="space-y-3">
              <li className="flex items-center gap-2 text-sm text-muted-foreground">
                <Mail className="h-4 w-4" />
                support@campusvoice.edu
              </li>
              <li className="flex items-center gap-2 text-sm text-muted-foreground">
                <Phone className="h-4 w-4" />
                +1 (555) 123-4567
              </li>
              <li className="flex items-center gap-2 text-sm text-muted-foreground">
                <MapPin className="h-4 w-4" />
                Student Affairs Office
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 border-t border-border/70 pt-6 text-center text-sm text-muted-foreground">
          <p>&copy; 2026 CampusVoice. Built for clearer student support workflows.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
