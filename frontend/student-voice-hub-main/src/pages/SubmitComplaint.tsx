import { useState, useEffect } from "react";
import { savePendingComplaint, deletePendingComplaint } from "@/offline/db";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { 
  Send, 
  AlertCircle, 
  FileText,
  Upload,
  CheckCircle2
} from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { categories } from "@/data/mockComplaints";
import { useAuth } from "@/contexts/AuthContext";
import { useCreateComplaint } from "@/hooks/useComplaints";

const SubmitComplaint = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { user, loading } = useAuth();
  const createComplaint = useCreateComplaint();
  const [formData, setFormData] = useState({
    title: "",
    category: "",
    priority: "medium",
    description: "",
  });

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [user, loading, navigate]);

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  if (!formData.title || !formData.category || !formData.description) {
    toast({
      title: "Missing Information",
      description: "Please fill in all required fields.",
      variant: "destructive",
    });
    return;
  }

  if (!user?.id) {
    toast({
      title: "Not signed in",
      description: "Please sign in again.",
      variant: "destructive",
    });
    navigate("/auth");
    return;
  }

  const payload = {
    title: formData.title,
    description: formData.description,
    category: formData.category,
    priority: formData.priority || "medium",
    user_id: user.id,
  };

  // ✅ ALWAYS save locally first (guaranteed offline support)
  const localId = await savePendingComplaint(payload);

  toast({
    title: "Saved ✅",
    description:
      "Saved locally. If you're online it will sync now; otherwise it will sync when internet returns.",
  });

  // ✅ Try to submit online; if it succeeds, remove local queued copy
  try {
    await createComplaint.mutateAsync({
      title: payload.title,
      description: payload.description,
      category: payload.category,
      priority: payload.priority,
    });

    await deletePendingComplaint(localId);
  } catch {
    // keep it queued for auto-sync
  }

  navigate("/dashboard");
};
// ✅ Show spinner only when ONLINE
if (loading && navigator.onLine) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
    </div>
  );
}

// ✅ OFFLINE: don't block the page
if (!navigator.onLine && loading) {
  // allow user to use offline submit (IndexedDB)
  // do NOT return spinner
}
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      <main className="flex-1">
        {/* Header */}
        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold text-foreground">Submit a Complaint</h1>
            <p className="text-muted-foreground mt-1">
              Share your concerns and we'll work to resolve them
            </p>
          </div>
        </section>

        {/* Form Section */}
        <section className="container mx-auto px-4 py-8">
          <div className="max-w-3xl mx-auto">
            <div className="grid gap-6">
              {/* Tips Card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <Card className="border-primary/20 bg-primary/5">
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-primary mt-0.5" />
                      <div className="space-y-1">
                        <h4 className="font-medium text-foreground">Tips for Effective Complaints</h4>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          <li>• Be specific about the issue and its location</li>
                          <li>• Include relevant dates and times</li>
                          <li>• Suggest a possible solution if you have one</li>
                          <li>• Attach evidence or photos if available</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Main Form */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card className="shadow-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-primary" />
                      Complaint Details
                    </CardTitle>
                    <CardDescription>
                      Provide as much detail as possible to help us address your concern
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-6">
                      {/* Title */}
                      <div className="space-y-2">
                        <Label htmlFor="title">
                          Complaint Title <span className="text-destructive">*</span>
                        </Label>
                        <Input
                          id="title"
                          placeholder="Brief summary of your complaint"
                          value={formData.title}
                          onChange={(e) =>
                            setFormData({ ...formData, title: e.target.value })
                          }
                          maxLength={100}
                        />
                        <p className="text-xs text-muted-foreground text-right">
                          {formData.title.length}/100 characters
                        </p>
                      </div>

                      {/* Category and Priority */}
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="category">
                            Category <span className="text-destructive">*</span>
                          </Label>
                          <Select
                            value={formData.category}
                            onValueChange={(value) =>
                              setFormData({ ...formData, category: value })
                            }
                          >
                            <SelectTrigger id="category">
                              <SelectValue placeholder="Select a category" />
                            </SelectTrigger>
                            <SelectContent>
                              {categories.map((cat) => (
                                <SelectItem key={cat} value={cat}>
                                  {cat}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="priority">Priority Level</Label>
                          <Select
                            value={formData.priority}
                            onValueChange={(value) =>
                              setFormData({ ...formData, priority: value })
                            }
                          >
                            <SelectTrigger id="priority">
                              <SelectValue placeholder="Select priority" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="low">
                                <span className="flex items-center gap-2">
                                  <span className="h-2 w-2 rounded-full bg-success" />
                                  Low
                                </span>
                              </SelectItem>
                              <SelectItem value="medium">
                                <span className="flex items-center gap-2">
                                  <span className="h-2 w-2 rounded-full bg-pending" />
                                  Medium
                                </span>
                              </SelectItem>
                              <SelectItem value="high">
                                <span className="flex items-center gap-2">
                                  <span className="h-2 w-2 rounded-full bg-destructive" />
                                  High
                                </span>
                              </SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      {/* Description */}
                      <div className="space-y-2">
                        <Label htmlFor="description">
                          Description <span className="text-destructive">*</span>
                        </Label>
                        <Textarea
                          id="description"
                          placeholder="Describe your complaint in detail. Include what happened, when it happened, where it happened, and who was involved if applicable."
                          value={formData.description}
                          onChange={(e) =>
                            setFormData({ ...formData, description: e.target.value })
                          }
                          rows={6}
                          maxLength={1000}
                        />
                        <p className="text-xs text-muted-foreground text-right">
                          {formData.description.length}/1000 characters
                        </p>
                      </div>

                      {/* File Upload */}
                      <div className="space-y-2">
                        <Label>Attachments (Optional)</Label>
                        <div className="border-2 border-dashed rounded-xl p-8 text-center hover:border-primary/50 transition-colors cursor-pointer bg-secondary/30">
                          <Upload className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                          <p className="text-sm text-muted-foreground">
                            Drag and drop files here, or click to browse
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            Supports images, PDFs up to 5MB
                          </p>
                        </div>
                      </div>

                      {/* Submit Button */}
                      <div className="flex gap-3 pt-4">
                        <Button
                          type="button"
                          variant="outline"
                          className="flex-1"
                          onClick={() => navigate("/dashboard")}
                        >
                          Cancel
                        </Button>
                        <Button
                          type="submit"
                          variant="hero"
                          className="flex-1"
                          disabled={createComplaint.isPending}
                        >
                          {createComplaint.isPending ? (
                            <>
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                              >
                                <CheckCircle2 className="h-4 w-4" />
                              </motion.div>
                              Submitting...
                            </>
                          ) : (
                            <>
                              <Send className="h-4 w-4" />
                              Submit Complaint
                            </>
                          )}
                        </Button>
                      </div>
                    </form>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default SubmitComplaint;
