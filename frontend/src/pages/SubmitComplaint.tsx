import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AlertCircle, ArrowRight, CheckCircle2, FileAudio, FileText, Files, Mic, MicOff, Send, ShieldOff, Sparkles, Upload } from "lucide-react";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { useCreateComplaint } from "@/hooks/useComplaints";
import { useAccessProfile } from "@/hooks/useAccessProfile";
import { complaintsApi } from "@/integrations/aws/client";
import { savePendingComplaint, deletePendingComplaint, type QueuedAttachment } from "@/offline/db";

const SubmitComplaint = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const { user, loading } = useAuth();
  const { data: accessProfile } = useAccessProfile();
  const createComplaint = useCreateComplaint();

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const voiceChunksRef = useRef<Blob[]>([]);
  const voiceStopPromiseRef = useRef<Promise<Blob | null> | null>(null);

  const isAdmin = Boolean(accessProfile?.is_admin);
  const isSuperAdmin = Boolean(accessProfile?.is_super_admin);

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedAudioFiles, setSelectedAudioFiles] = useState<File[]>([]);
  const [voiceBlob, setVoiceBlob] = useState<Blob | null>(null);
  const [voiceMimeType, setVoiceMimeType] = useState("audio/webm");
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnonymous, setIsAnonymous] = useState(true);
  const [formData, setFormData] = useState({ title: "", description: "" });
  const [studentDetails, setStudentDetails] = useState({
    studentName: "",
    studentEmail: user?.email ?? "",
    studentPhone: "",
    studentDepartment: "",
    studentRegistrationNumber: "",
  });
  const [isSuccessDialogOpen, setIsSuccessDialogOpen] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
    if (!loading && isSuperAdmin) {
      navigate("/super-admin");
    }
    if (!loading && isAdmin && !isSuperAdmin) {
      navigate("/admin");
    }
  }, [user, loading, isAdmin, isSuperAdmin, navigate]);

  useEffect(() => {
    setStudentDetails((current) => ({
      ...current,
      studentEmail: current.studentEmail || user?.email || "",
    }));
  }, [user?.email]);

  const totalSelectedSizeMb = useMemo(
    () =>
      [...selectedFiles, ...selectedAudioFiles].reduce((acc, file) => acc + file.size, 0) / 1024 / 1024,
    [selectedAudioFiles, selectedFiles]
  );

  const totalEvidenceCount = selectedFiles.length + selectedAudioFiles.length + (voiceBlob ? 1 : 0);
  const selectedDocumentNames = selectedFiles.map((file) => file.name);
  const selectedAudioNames = selectedAudioFiles.map((file) => file.name);

  const voicePreviewUrl = useMemo(() => {
    if (!voiceBlob) return null;
    return URL.createObjectURL(voiceBlob);
  }, [voiceBlob]);

  useEffect(() => {
    return () => {
      if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [voicePreviewUrl]);

  const evidenceTypes = useMemo(() => {
    const types = new Set<string>(["text"]);
    [...selectedFiles, ...selectedAudioFiles].forEach((file) => {
      if (file.type.startsWith("image/")) types.add("image");
      else if (file.type.startsWith("audio/")) types.add("voice");
      else types.add("document");
    });
    if (voiceBlob) types.add("voice");
    return Array.from(types);
  }, [selectedAudioFiles, selectedFiles, voiceBlob]);

  const getSupportedRecordingMimeType = () => {
    if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
      return "";
    }
    const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
    return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";
  };

  const resetRecordingState = () => {
    mediaRecorderRef.current = null;
    voiceStopPromiseRef.current = null;
    setIsRecording(false);
  };

  const startRecording = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
        throw new Error("Voice recording is not supported in this browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const preferredMimeType = getSupportedRecordingMimeType();
      const recorder = preferredMimeType
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);
      voiceChunksRef.current = [];
      setVoiceBlob(null);
      setVoiceMimeType(recorder.mimeType || preferredMimeType || "audio/webm");
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          voiceChunksRef.current.push(event.data);
        }
      };
      recorder.start(1000);
      setIsRecording(true);
      voiceStopPromiseRef.current = new Promise<Blob | null>((resolve) => {
        recorder.onstop = () => {
          const resolvedMimeType = recorder.mimeType || preferredMimeType || "audio/webm";
          const audioBlob =
            voiceChunksRef.current.length > 0
              ? new Blob(voiceChunksRef.current, { type: resolvedMimeType })
              : null;
          setVoiceMimeType(resolvedMimeType);
          setVoiceBlob(audioBlob);
          stream.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
          resetRecordingState();
          resolve(audioBlob);
        };
      });
      recorder.onerror = () => {
        stream.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        resetRecordingState();
      };
      mediaRecorderRef.current = recorder;
    } catch (error) {
      toast({
        title: "Recording failed",
        description: error instanceof Error ? error.message : "Could not access microphone.",
        variant: "destructive",
        duration: 2000,
      });
    }
  };

  const stopRecording = async () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return voiceBlob;
    if (recorder.state !== "inactive") {
      recorder.requestData();
      recorder.stop();
    }
    const audioBlob = await voiceStopPromiseRef.current;
    setVoiceBlob(audioBlob ?? null);
    return audioBlob;
  };

  const getExtensionForMimeType = (mimeType: string) => {
    if (mimeType.includes("ogg")) return "ogg";
    if (mimeType.includes("mp4") || mimeType.includes("aac")) return "m4a";
    if (mimeType.includes("mpeg") || mimeType.includes("mp3")) return "mp3";
    if (mimeType.includes("wav")) return "wav";
    return "webm";
  };

  const clearVoiceClip = () => {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    resetRecordingState();
    voiceChunksRef.current = [];
    setVoiceBlob(null);
  };

  const resetForm = () => {
    setFormData({ title: "", description: "" });
    setStudentDetails({
      studentName: "",
      studentEmail: user?.email ?? "",
      studentPhone: "",
      studentDepartment: "",
      studentRegistrationNumber: "",
    });
    setSelectedFiles([]);
    setSelectedAudioFiles([]);
    setIsAnonymous(true);
    clearVoiceClip();
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!formData.title.trim() || !formData.description.trim()) {
      toast({
        title: "Missing information",
        description: "Please add both a title and a description.",
        variant: "destructive",
        duration: 2000,
      });
      return;
    }

    if (!user?.id) {
      toast({
        title: "Not signed in",
        description: "Please sign in again to continue.",
        variant: "destructive",
        duration: 2000,
      });
      navigate("/auth");
      return;
    }

    let attachmentKeys: string[] = [];
    let queuedAttachments: QueuedAttachment[] = [];
    let voiceAttachment = voiceBlob;
    let voiceAttachmentMimeType = voiceMimeType;

    if (isRecording) {
      voiceAttachment = await stopRecording();
      voiceAttachmentMimeType = mediaRecorderRef.current?.mimeType || voiceMimeType;
      if (!voiceAttachment) {
        toast({
          title: "Voice recording unavailable",
          description: "Please try recording again or upload an audio file instead.",
          variant: "destructive",
          duration: 2000,
        });
        return;
      }
    }

    const allFiles: Array<{ file: Blob; name: string; type: string; size: number }> = [
      ...selectedFiles.map((file) => ({
        file,
        name: file.name,
        type: file.type || "application/octet-stream",
        size: file.size,
      })),
      ...selectedAudioFiles.map((file) => ({
        file,
        name: file.name,
        type: file.type || "audio/webm",
        size: file.size,
      })),
    ];

    if (voiceAttachment) {
      const resolvedVoiceMimeType = voiceAttachmentMimeType || voiceAttachment.type || "audio/webm";
      allFiles.push({
        file: voiceAttachment,
        name: `voice-${Date.now()}.${getExtensionForMimeType(resolvedVoiceMimeType)}`,
        type: resolvedVoiceMimeType,
        size: voiceAttachment.size,
      });
    }

    if (navigator.onLine && allFiles.length > 0) {
      try {
        setIsUploading(true);
        attachmentKeys = await Promise.all(
          allFiles.map(async (item) => {
            const uploadMeta = await complaintsApi.createUploadUrl({
              fileName: item.name,
              contentType: item.type,
              fileSize: item.size,
            });
            await complaintsApi.uploadToS3(uploadMeta.uploadUrl, item.file, item.type);
            if (uploadMeta.warnings?.length) {
              toast({
                title: "Attachment warning",
                description: uploadMeta.warnings[0],
                duration: 2500,
              });
            }
            return uploadMeta.key;
          })
        );
      } catch {
        toast({
          title: "Attachment upload failed",
          description: "Could not upload one or more files. Please try again.",
          variant: "destructive",
          duration: 2000,
        });
        return;
      } finally {
        setIsUploading(false);
      }
    } else if (allFiles.length > 0) {
      queuedAttachments = allFiles.map((item) => ({
        name: item.name,
        type: item.type,
        size: item.size,
        file: item.file,
      }));
    }

    const payload = {
      title: formData.title.trim(),
      description: formData.description.trim(),
      user_id: user.id,
      is_anonymous: isAnonymous,
      student_name: studentDetails.studentName.trim(),
      student_email: (studentDetails.studentEmail.trim() || user.email || "").trim(),
      student_phone: studentDetails.studentPhone.trim(),
      student_department: studentDetails.studentDepartment.trim(),
      student_registration_number: studentDetails.studentRegistrationNumber.trim(),
      attachment_keys: attachmentKeys,
      queued_attachments: queuedAttachments,
      evidence_types: evidenceTypes,
    };

    const localId = await savePendingComplaint(payload);

    toast({
      title: navigator.onLine ? "Complaint submitted" : "Saved offline",
      description: navigator.onLine
        ? "Your complaint has been submitted successfully."
        : "Your complaint is saved and will sync when you are online again.",
      duration: 2000,
    });

    let submitted = false;

    try {
      await createComplaint.mutateAsync({
        title: payload.title,
        description: payload.description,
        is_anonymous: payload.is_anonymous,
        student_name: payload.student_name,
        student_email: payload.student_email,
        student_phone: payload.student_phone,
        student_department: payload.student_department,
        student_registration_number: payload.student_registration_number,
        attachment_keys: payload.attachment_keys,
        queued_attachments: payload.queued_attachments,
        evidence_types: payload.evidence_types,
        already_queued: true,
      });
      await deletePendingComplaint(localId);
      await queryClient.invalidateQueries({ queryKey: ["complaints"], refetchType: "all" });
      await queryClient.invalidateQueries({ queryKey: ["notifications"], refetchType: "all" });
      submitted = true;
    } catch (error) {
      toast({
        title: navigator.onLine ? "Submission needs changes" : "Saved offline",
        description:
          navigator.onLine && error instanceof Error
            ? error.message
            : "Your complaint is saved locally and will sync later.",
        variant: navigator.onLine ? "destructive" : "default",
        duration: 2500,
      });
      if (navigator.onLine) {
        return;
      }
    }

    if (!navigator.onLine) {
      await queryClient.invalidateQueries({ queryKey: ["complaints"], refetchType: "all" });
    }

    if (submitted) {
      resetForm();
      setIsSuccessDialogOpen(true);
      return;
    }

    if (!navigator.onLine) {
      navigate("/dashboard");
    }
  };

  if (loading && navigator.onLine) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <Dialog open={isSuccessDialogOpen} onOpenChange={setIsSuccessDialogOpen}>
        <DialogContent className="overflow-hidden border-white/70 bg-white/95 p-0 shadow-elevated sm:max-w-xl">
          <div className="relative">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(110,231,183,0.3),transparent_34%),radial-gradient(circle_at_top_right,rgba(96,165,250,0.24),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(240,253,244,0.96))]" />
            <motion.div
              initial={{ opacity: 0, y: 14, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
              className="relative space-y-6 p-6 sm:p-8"
            >
              <DialogHeader className="items-center space-y-4 text-center sm:items-center sm:text-center">
                <motion.div
                  initial={{ scale: 0.85, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.08, duration: 0.24, ease: "easeOut" }}
                  className="relative flex h-20 w-20 items-center justify-center rounded-[28px] border border-emerald-200/80 bg-white/80 shadow-card"
                >
                  <div className="absolute inset-2 rounded-[22px] bg-emerald-100/80" />
                  <div className="absolute -right-1 -top-1 flex h-7 w-7 items-center justify-center rounded-full gradient-primary text-white shadow-sm">
                    <Sparkles className="h-3.5 w-3.5" />
                  </div>
                  <CheckCircle2 className="relative h-10 w-10 text-success" />
                </motion.div>

                <div className="space-y-2">
                  <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-emerald-700">
                    <Sparkles className="h-3.5 w-3.5" />
                    Successfully Submitted
                  </div>
                  <DialogTitle className="heading-display text-3xl font-bold leading-tight text-foreground">
                    Your complaint is now in the review queue
                  </DialogTitle>
                  <DialogDescription className="mx-auto max-w-md text-sm leading-6 text-muted-foreground">
                    Everything went through successfully. You can track updates from the dashboard or stay here and file another issue right away.
                  </DialogDescription>
                </div>
              </DialogHeader>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-2xl border border-white/80 bg-white/85 p-4 text-center shadow-card">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Status</p>
                  <p className="mt-2 text-base font-semibold text-foreground">Submitted</p>
                </div>
                <div className="rounded-2xl border border-white/80 bg-white/85 p-4 text-center shadow-card">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Visibility</p>
                  <p className="mt-2 text-base font-semibold text-foreground">Saved securely</p>
                </div>
                <div className="rounded-2xl border border-white/80 bg-white/85 p-4 text-center shadow-card">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Next Step</p>
                  <p className="mt-2 text-base font-semibold text-foreground">Team review</p>
                </div>
              </div>

              <div className="rounded-2xl border border-emerald-200/70 bg-emerald-50/80 px-4 py-3 text-sm text-emerald-900">
                You will be able to follow status changes, responses, and resolution updates from your complaint dashboard.
              </div>

              <DialogFooter className="flex-col gap-3 sm:flex-col sm:space-x-0">
                <Button type="button" variant="hero" size="lg" className="w-full" onClick={() => navigate("/dashboard")}>
                  Go to Dashboard
                  <ArrowRight className="h-4 w-4" />
                </Button>
                <Button type="button" variant="outline" size="lg" className="w-full bg-white/80" onClick={() => setIsSuccessDialogOpen(false)}>
                  Submit Another Complaint
                </Button>
              </DialogFooter>
            </motion.div>
          </div>
        </DialogContent>
      </Dialog>

      <main className="flex-1">
        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold text-foreground">Submit a Complaint</h1>
            <p className="mt-1 text-muted-foreground">
              Share what happened, add supporting evidence if you have it, and choose whether to stay anonymous.
            </p>
          </div>
        </section>

        <section className="container mx-auto px-4 py-8">
          <div className="mx-auto max-w-4xl space-y-6">
            <Card className="border-primary/20 bg-primary/5">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="mt-0.5 h-5 w-5 text-primary" />
                  <div className="space-y-1">
                    <h4 className="font-medium text-foreground">Tips for a strong complaint</h4>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      <li>- Keep the title short and clear.</li>
                      <li>- Mention what happened, where it happened, and when it started.</li>
                      <li>- Add photo, document, or audio proof if it helps explain the issue.</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              <Card className="shadow-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Complaint Details
                  </CardTitle>
                  <CardDescription>Provide enough detail for the team to review and act on your complaint.</CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="title">
                        Complaint Title <span className="text-destructive">*</span>
                      </Label>
                      <Input
                        id="title"
                        value={formData.title}
                        maxLength={100}
                        placeholder="Brief summary of your complaint"
                        onChange={(event) => setFormData((current) => ({ ...current, title: event.target.value }))}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="description">
                        Description <span className="text-destructive">*</span>
                      </Label>
                      <Textarea
                        id="description"
                        rows={6}
                        maxLength={1000}
                        placeholder="Describe what happened, where it happened, and what support you need."
                        value={formData.description}
                        onChange={(event) => setFormData((current) => ({ ...current, description: event.target.value }))}
                      />
                    </div>

                    <div className="flex items-start justify-between rounded-2xl border bg-muted/20 p-4">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <ShieldOff className="h-4 w-4 text-primary" />
                          <Label htmlFor="anonymous-toggle" className="text-base font-semibold">
                            Submit anonymously
                          </Label>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Anonymous mode is on by default. Turn it off only if you want your identity visible to reviewers.
                        </p>
                      </div>
                      <Switch
                        id="anonymous-toggle"
                        checked={isAnonymous}
                        onCheckedChange={setIsAnonymous}
                      />
                    </div>

                    <div className="space-y-4 rounded-2xl border bg-background p-5">
                      <div className="space-y-1">
                        <Label className="text-base font-semibold">Student Details</Label>
                        <p className="text-sm text-muted-foreground">
                          Add your student information so admins can contact or verify the case if follow-up is needed.
                          {isAnonymous
                            ? " These details stay hidden from reviewers while anonymous mode is on."
                            : " These details will be visible to admins because anonymous mode is off."}
                        </p>
                      </div>

                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-2">
                          <Label htmlFor="student-name">Student Name</Label>
                          <Input
                            id="student-name"
                            value={studentDetails.studentName}
                            placeholder="Your full name"
                            onChange={(event) =>
                              setStudentDetails((current) => ({ ...current, studentName: event.target.value }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="student-email">Student Email</Label>
                          <Input
                            id="student-email"
                            type="email"
                            value={studentDetails.studentEmail}
                            placeholder="student@college.edu"
                            onChange={(event) =>
                              setStudentDetails((current) => ({ ...current, studentEmail: event.target.value }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="student-phone">Phone Number</Label>
                          <Input
                            id="student-phone"
                            value={studentDetails.studentPhone}
                            placeholder="Contact number"
                            onChange={(event) =>
                              setStudentDetails((current) => ({ ...current, studentPhone: event.target.value }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="student-department">Department / Program</Label>
                          <Input
                            id="student-department"
                            value={studentDetails.studentDepartment}
                            placeholder="CSE, MBA, BCA, Hostel Block A..."
                            onChange={(event) =>
                              setStudentDetails((current) => ({ ...current, studentDepartment: event.target.value }))
                            }
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="student-registration-number">Student ID / Registration Number</Label>
                        <Input
                          id="student-registration-number"
                          value={studentDetails.studentRegistrationNumber}
                          placeholder="College roll number or registration ID"
                          onChange={(event) =>
                            setStudentDetails((current) => ({
                              ...current,
                              studentRegistrationNumber: event.target.value,
                            }))
                          }
                        />
                      </div>
                    </div>

                    <div className="space-y-4 rounded-2xl border bg-muted/20 p-4">
                      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                        <div>
                          <Label className="text-base font-semibold">Evidence Bundle</Label>
                          <p className="text-sm text-muted-foreground">
                            Upload documents, images, audio files, or record a quick voice note.
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-2 text-xs">
                          <span className="rounded-full border bg-background px-3 py-1 text-muted-foreground">
                            {totalEvidenceCount} attachment{totalEvidenceCount === 1 ? "" : "s"}
                          </span>
                          <span className="rounded-full border bg-background px-3 py-1 text-muted-foreground">
                            {totalSelectedSizeMb.toFixed(2)} MB selected
                          </span>
                        </div>
                      </div>

                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-3 rounded-xl border bg-background p-4">
                          <div className="flex items-center gap-2">
                            <Files className="h-4 w-4 text-primary" />
                            <Label className="text-sm font-medium">Images & Documents</Label>
                          </div>
                          <label
                            htmlFor="attachments"
                            className="block cursor-pointer rounded-xl border-2 border-dashed bg-secondary/30 p-8 text-center transition-colors hover:border-primary/50"
                          >
                            <Upload className="mx-auto mb-2 h-8 w-8 text-muted-foreground" />
                            <p className="text-sm font-medium text-foreground">Choose images or PDFs</p>
                            <p className="mt-1 text-xs text-muted-foreground">Up to 5 files.</p>
                          </label>
                          <Input
                            id="attachments"
                            type="file"
                            multiple
                            accept="image/*,.pdf"
                            className="hidden"
                            onChange={(event) => {
                              const files = Array.from(event.target.files ?? []).slice(0, 5);
                              setSelectedFiles(files);
                            }}
                          />
                          {selectedDocumentNames.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                              {selectedDocumentNames.map((name) => (
                                <span key={name} className="rounded-full bg-primary/10 px-3 py-1 text-xs text-primary">
                                  {name}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs text-muted-foreground">No image or document files selected yet.</p>
                          )}
                        </div>

                        <div className="space-y-3 rounded-xl border bg-background p-4">
                          <div className="flex items-center gap-2">
                            <FileAudio className="h-4 w-4 text-primary" />
                            <Label className="text-sm font-medium">Audio Upload</Label>
                          </div>
                          <Input
                            type="file"
                            multiple
                            accept="audio/*"
                            onChange={(event) => {
                              const files = Array.from(event.target.files ?? []).slice(0, 3);
                              setSelectedAudioFiles(files);
                            }}
                          />
                          <p className="text-xs text-muted-foreground">
                            Upload audio from your device if you prefer not to record live.
                          </p>
                          {selectedAudioNames.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                              {selectedAudioNames.map((name) => (
                                <span key={name} className="rounded-full bg-amber-100 px-3 py-1 text-xs text-amber-800">
                                  {name}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs text-muted-foreground">No external audio files selected yet.</p>
                          )}
                        </div>
                      </div>

                      <div className="space-y-3 rounded-xl border bg-background p-4">
                        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                          <div>
                            <Label className="text-sm font-medium">Live Voice Recording</Label>
                            <p className="text-xs text-muted-foreground">
                              Record directly in the browser. If you submit while recording, it will be attached automatically.
                            </p>
                          </div>
                          <span
                            className={`rounded-full px-3 py-1 text-xs font-medium ${
                              isRecording
                                ? "bg-red-100 text-red-700"
                                : voiceBlob
                                  ? "bg-emerald-100 text-emerald-700"
                                  : "bg-muted text-muted-foreground"
                            }`}
                          >
                            {isRecording ? "Recording..." : voiceBlob ? "Voice clip ready" : "No recording yet"}
                          </span>
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                          {!isRecording ? (
                            <Button type="button" variant="outline" onClick={startRecording}>
                              <Mic className="mr-2 h-4 w-4" />
                              Start Recording
                            </Button>
                          ) : (
                            <Button type="button" variant="destructive" onClick={() => void stopRecording()}>
                              <MicOff className="mr-2 h-4 w-4" />
                              Stop Recording
                            </Button>
                          )}
                          {voiceBlob && (
                            <Button type="button" variant="ghost" onClick={clearVoiceClip}>
                              Remove Voice Clip
                            </Button>
                          )}
                        </div>

                        {voicePreviewUrl ? (
                          <div className="rounded-xl bg-secondary/40 p-3">
                            <audio controls preload="metadata" className="w-full">
                              <source src={voicePreviewUrl} type={voiceMimeType} />
                            </audio>
                          </div>
                        ) : (
                          <div className="rounded-xl border border-dashed p-4 text-sm text-muted-foreground">
                            Recorded audio preview will appear here once a clip is captured.
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex gap-3 pt-4">
                      <Button type="button" variant="outline" className="flex-1" onClick={() => navigate("/dashboard")}>
                        Cancel
                      </Button>
                      <Button type="submit" variant="hero" className="flex-1" disabled={createComplaint.isPending || isUploading}>
                        {createComplaint.isPending || isUploading ? (
                          <>
                            <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
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
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default SubmitComplaint;
