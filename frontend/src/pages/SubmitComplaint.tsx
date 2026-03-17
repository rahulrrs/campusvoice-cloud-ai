import { useEffect, useMemo, useRef, useState } from "react";
import { savePendingComplaint, deletePendingComplaint } from "@/offline/db";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Send, AlertCircle, FileText, Upload, CheckCircle2, Mic, MicOff, FileAudio, Files } from "lucide-react";
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
import { complaintsApi, type ComplaintAnalysisBundle } from "@/integrations/aws/client";
import type { QueuedAttachment } from "@/offline/db";

const SubmitComplaint = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { user, loading } = useAuth();
  const createComplaint = useCreateComplaint();
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const voiceChunksRef = useRef<Blob[]>([]);
  const voiceStopPromiseRef = useRef<Promise<Blob | null> | null>(null);

  const adminEmails = (import.meta.env.VITE_ADMIN_EMAILS ?? "")
    .split(",")
    .map((v: string) => v.trim().toLowerCase())
    .filter((v: string) => v.length > 0);
  const isAdmin = !!user?.email && adminEmails.includes(user.email.toLowerCase());

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedAudioFiles, setSelectedAudioFiles] = useState<File[]>([]);
  const [voiceBlob, setVoiceBlob] = useState<Blob | null>(null);
  const [voiceMimeType, setVoiceMimeType] = useState("audio/webm");
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisPreview, setAnalysisPreview] = useState<ComplaintAnalysisBundle | null>(null);
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

  useEffect(() => {
    const title = formData.title.trim();
    const description = formData.description.trim();
    if (!navigator.onLine || (!title && description.length < 15)) {
      return;
    }

    const timer = setTimeout(async () => {
      try {
        setIsAnalyzing(true);
        const analysis = await complaintsApi.analyzeComplaint({
          title,
          description,
        });
        setAnalysisPreview(analysis);
      } catch {
        // Keep form functional even if analysis is unavailable.
      } finally {
        setIsAnalyzing(false);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [formData.title, formData.description]);

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
      if (voicePreviewUrl) {
        URL.revokeObjectURL(voicePreviewUrl);
      }
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [voicePreviewUrl]);

  const evidenceTypes = useMemo(() => {
    const set = new Set<string>(["text"]);
    [...selectedFiles, ...selectedAudioFiles].forEach((file) => {
      if (file.type.startsWith("image/")) {
        set.add("image");
      } else if (file.type.startsWith("audio/")) {
        set.add("voice");
      } else {
        set.add("document");
      }
    });
    if (voiceBlob) {
      set.add("voice");
    }
    return Array.from(set);
  }, [selectedAudioFiles, selectedFiles, voiceBlob]);

  const getSupportedRecordingMimeType = () => {
    if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
      return "";
    }
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/ogg;codecs=opus",
    ];
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
      const message = error instanceof Error ? error.message : "Could not access microphone.";
      toast({
        title: "Recording failed",
        description: message,
        variant: "destructive",
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
    if (audioBlob) {
      setVoiceBlob(audioBlob);
    } else {
      setVoiceBlob(null);
    }
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
    let voiceAttachment = voiceBlob;
    let voiceAttachmentMimeType = voiceMimeType;

    if (isRecording) {
      voiceAttachment = await stopRecording();
      voiceAttachmentMimeType = mediaRecorderRef.current?.mimeType || voiceMimeType;
      if (!voiceAttachment) {
        toast({
          title: "Voice recording unavailable",
          description: "We could not capture the recording. Please record again or upload an audio file.",
          variant: "destructive",
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
            });
            await complaintsApi.uploadToS3(uploadMeta.uploadUrl, item.file, item.type);
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
    } else if (allFiles.length > 0) {
      queuedAttachments = allFiles.map((item) => ({
        name: item.name,
        type: item.type,
        size: item.size,
        file: item.file,
      }));
    }

    const payload = {
      title: formData.title,
      description: formData.description,
      user_id: user.id,
      attachment_keys: attachmentKeys,
      queued_attachments: queuedAttachments,
      evidence_types: evidenceTypes,
      analysis: analysisPreview ?? undefined,
      source_language: analysisPreview?.source_language ?? "en",
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
        evidence_types: payload.evidence_types,
        analysis: payload.analysis,
        source_language: payload.source_language,
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
              Multimodal submission with AI classification, urgency, abuse, and duplicate checks.
            </p>
          </div>
        </section>

        <section className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            <div className="grid gap-6">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <Card className="border-primary/20 bg-primary/5">
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-primary mt-0.5" />
                      <div className="space-y-1">
                        <h4 className="font-medium text-foreground">Tips for Effective Complaints</h4>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          <li>- Add text details plus image/voice evidence when possible</li>
                          <li>- Keep the title concise and the description factual</li>
                          <li>- The system will detect urgency, abuse risk, and potential duplicates</li>
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
                    <CardDescription>Provide details to help us address your concern</CardDescription>
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
                      </div>

                      <div className="space-y-4 rounded-2xl border bg-muted/20 p-4">
                        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                          <div>
                            <Label className="text-base font-semibold">Evidence Bundle</Label>
                            <p className="text-sm text-muted-foreground">
                              Add files, upload audio, or record a voice note to strengthen the complaint.
                            </p>
                          </div>
                          <div className="flex flex-wrap gap-2 text-xs">
                            <span className="rounded-full bg-background px-3 py-1 text-muted-foreground border">
                              {totalEvidenceCount} attachment{totalEvidenceCount === 1 ? "" : "s"}
                            </span>
                            <span className="rounded-full bg-background px-3 py-1 text-muted-foreground border">
                              {totalSelectedSizeMb.toFixed(2)} MB selected
                            </span>
                          </div>
                        </div>

                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="rounded-xl border bg-background p-4 space-y-3">
                            <div className="flex items-center gap-2">
                              <Files className="h-4 w-4 text-primary" />
                              <Label className="text-sm font-medium">Images & Documents</Label>
                            </div>
                            <label
                              htmlFor="attachments"
                              className="block border-2 border-dashed rounded-xl p-8 text-center hover:border-primary/50 transition-colors cursor-pointer bg-secondary/30"
                            >
                              <Upload className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                              <p className="text-sm font-medium text-foreground">Choose images or PDFs</p>
                              <p className="text-xs text-muted-foreground mt-1">Up to 5 files. Useful for photos, screenshots, and letters.</p>
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

                          <div className="rounded-xl border bg-background p-4 space-y-3">
                            <div className="flex items-center gap-2">
                              <FileAudio className="h-4 w-4 text-primary" />
                              <Label className="text-sm font-medium">Audio File Upload</Label>
                            </div>
                            <Input
                              type="file"
                              multiple
                              accept="audio/*"
                              onChange={(e) => {
                                const files = Array.from(e.target.files ?? []).slice(0, 3);
                                setSelectedAudioFiles(files);
                              }}
                            />
                            <p className="text-xs text-muted-foreground">
                              Upload existing audio from your device if you do not want to record live.
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

                        <div className="rounded-xl border bg-background p-4 space-y-3">
                          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                            <div>
                              <Label className="text-sm font-medium">Live Voice Recording</Label>
                              <p className="text-xs text-muted-foreground">
                                Record directly in the browser. If you submit mid-recording, we auto-stop and attach it.
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
                                <Mic className="h-4 w-4 mr-2" />
                                Start Recording
                              </Button>
                            ) : (
                              <Button type="button" variant="destructive" onClick={() => void stopRecording()}>
                                <MicOff className="h-4 w-4 mr-2" />
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

                      {analysisPreview && (
                        <div className="rounded-xl border bg-muted/20 p-4 text-sm space-y-3">
                          <div className="flex items-center justify-between">
                            <p className="font-medium text-foreground">AI Review Snapshot</p>
                            <span className="rounded-full bg-background px-3 py-1 text-xs text-muted-foreground border">
                              {analysisPreview.source_language.toUpperCase()}
                            </span>
                          </div>
                          <div className="grid gap-3 md:grid-cols-2">
                            <div className="rounded-lg bg-background p-3 border">
                              <p className="text-xs uppercase tracking-wide text-muted-foreground">Predicted category</p>
                              <p className="mt-1 font-semibold text-foreground">{analysisPreview.classification.label}</p>
                            </div>
                            <div className="rounded-lg bg-background p-3 border">
                              <p className="text-xs uppercase tracking-wide text-muted-foreground">Urgency score</p>
                              <p className="mt-1 font-semibold text-foreground">{analysisPreview.sentiment.urgency_score.toFixed(2)}</p>
                            </div>
                            <div className="rounded-lg bg-background p-3 border">
                              <p className="text-xs uppercase tracking-wide text-muted-foreground">Detected emotion</p>
                              <p className="mt-1 font-semibold text-foreground">{analysisPreview.sentiment.emotion}</p>
                            </div>
                            <div className="rounded-lg bg-background p-3 border">
                              <p className="text-xs uppercase tracking-wide text-muted-foreground">Risk profile</p>
                              <p className="mt-1 font-semibold text-foreground">
                                Toxicity {analysisPreview.abuse.toxicity_score.toFixed(2)} | Spam {analysisPreview.abuse.spam_score.toFixed(2)}
                              </p>
                            </div>
                          </div>
                          <div className="rounded-lg bg-background p-3 border">
                            <p className="text-xs uppercase tracking-wide text-muted-foreground">Duplicate check</p>
                            <p className="mt-1 font-semibold text-foreground">
                              {analysisPreview.duplicate_detection.is_duplicate ? "Possible duplicate found" : "No strong duplicate"}
                            </p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Method: {analysisPreview.duplicate_detection.method}
                            </p>
                          </div>
                        </div>
                      )}
                      {isAnalyzing && <p className="text-xs text-muted-foreground">Analyzing complaint...</p>}

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
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default SubmitComplaint;
