import { useEffect, useMemo, useState } from "react";
import { savePendingComplaint, deletePendingComplaint } from "@/offline/db";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Send, AlertCircle, FileText, Upload, CheckCircle2 } from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { useCreateComplaint } from "@/hooks/useComplaints";
import { complaintsApi } from "@/integrations/aws/client";
import type { QueuedAttachment } from "@/offline/db";

const SubmitComplaint = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { user, loading } = useAuth();
  const createComplaint = useCreateComplaint();
  const adminEmails = (import.meta.env.VITE_ADMIN_EMAILS ?? "")
    .split(",")
    .map((v: string) => v.trim().toLowerCase())
    .filter((v: string) => v.length > 0);
  const isAdmin = !!user?.email && adminEmails.includes(user.email.toLowerCase());
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [formData, setFormData] = useState({
    title: "",
    description: "",
  });

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
    if (!loading && isAdmin) {
      navigate("/admin");
    }
  }, [user, loading, isAdmin, navigate]);

  const totalSelectedSizeMb = useMemo(
    () => selectedFiles.reduce((acc, file) => acc + file.size, 0) / 1024 / 1024,
    [selectedFiles]
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.title || !formData.description) {
      toast({
        title: "Missing information",
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

    let attachmentKeys: string[] = [];
    let queuedAttachments: QueuedAttachment[] = [];

    if (navigator.onLine && selectedFiles.length > 0) {
      try {
        setIsUploading(true);
        attachmentKeys = await Promise.all(
          selectedFiles.map(async (file) => {
            const contentType = file.type || "application/octet-stream";
            const uploadMeta = await complaintsApi.createUploadUrl({
              fileName: file.name,
              contentType,
            });
            await complaintsApi.uploadToS3(uploadMeta.uploadUrl, file, contentType);
            return uploadMeta.key;
          })
        );
      } catch {
        toast({
          title: "Attachment upload failed",
          description: "Could not upload one or more files. Please try again.",
          variant: "destructive",
        });
        return;
      } finally {
        setIsUploading(false);
      }
    } else if (selectedFiles.length > 0) {
      queuedAttachments = selectedFiles.map((file) => ({
        name: file.name,
        type: file.type || "application/octet-stream",
        size: file.size,
        file,
      }));
    }

    const payload = {
      title: formData.title,
      description: formData.description,
      user_id: user.id,
      attachment_keys: attachmentKeys,
      queued_attachments: queuedAttachments,
    };

    const localId = await savePendingComplaint(payload);

    toast({
      title: "Saved",
      description:
        "Saved locally. If online it will sync now, otherwise it will sync when internet returns.",
    });

    try {
      await createComplaint.mutateAsync({
        title: payload.title,
        description: payload.description,
        attachment_keys: payload.attachment_keys,
        queued_attachments: payload.queued_attachments,
        already_queued: true,
      });
      await deletePendingComplaint(localId);
    } catch {
      // Keep queued for auto-sync.
    }

    navigate("/dashboard");
  };

  if (loading && navigator.onLine) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1">
        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold text-foreground">Submit a Complaint</h1>
            <p className="text-muted-foreground mt-1">
              Share your concerns and we will work to resolve them
            </p>
          </div>
        </section>

        <section className="container mx-auto px-4 py-8">
          <div className="max-w-3xl mx-auto">
            <div className="grid gap-6">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <Card className="border-primary/20 bg-primary/5">
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-primary mt-0.5" />
                      <div className="space-y-1">
                        <h4 className="font-medium text-foreground">Tips for Effective Complaints</h4>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          <li>- Be specific about the issue and location</li>
                          <li>- Include relevant dates and times</li>
                          <li>- Suggest a possible solution if you have one</li>
                          <li>- Attach evidence if available</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

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
                      Provide details to help us address your concern
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-6">
                      <div className="space-y-2">
                        <Label htmlFor="title">
                          Complaint Title <span className="text-destructive">*</span>
                        </Label>
                        <Input
                          id="title"
                          placeholder="Brief summary of your complaint"
                          value={formData.title}
                          onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                          maxLength={100}
                        />
                        <p className="text-xs text-muted-foreground text-right">
                          {formData.title.length}/100 characters
                        </p>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="description">
                          Description <span className="text-destructive">*</span>
                        </Label>
                        <Textarea
                          id="description"
                          placeholder="Describe what happened, where, and when."
                          value={formData.description}
                          onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                          rows={6}
                          maxLength={1000}
                        />
                        <p className="text-xs text-muted-foreground text-right">
                          {formData.description.length}/1000 characters
                        </p>
                      </div>

                      <div className="space-y-2">
                        <Label>Attachments (Optional)</Label>
                        <label
                          htmlFor="attachments"
                          className="block border-2 border-dashed rounded-xl p-8 text-center hover:border-primary/50 transition-colors cursor-pointer bg-secondary/30"
                        >
                          <Upload className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                          <p className="text-sm text-muted-foreground">Click to choose files</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            Images and PDFs, up to 5 files
                          </p>
                        </label>
                        <Input
                          id="attachments"
                          type="file"
                          multiple
                          accept="image/*,.pdf"
                          className="hidden"
                          onChange={(e) => {
                            const files = Array.from(e.target.files ?? []).slice(0, 5);
                            setSelectedFiles(files);
                          }}
                        />
                        {selectedFiles.length > 0 && (
                          <div className="text-sm text-muted-foreground">
                            {selectedFiles.length} file(s) selected ({totalSelectedSizeMb.toFixed(2)} MB)
                          </div>
                        )}
                      </div>

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
                          disabled={createComplaint.isPending || isUploading}
                        >
                          {createComplaint.isPending || isUploading ? (
                            <>
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                              >
                                <CheckCircle2 className="h-4 w-4" />
                              </motion.div>
                              {isUploading ? "Uploading..." : "Submitting..."}
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
