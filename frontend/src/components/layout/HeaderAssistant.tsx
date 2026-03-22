import { useEffect, useMemo, useRef, useState } from "react";
import { Bot, Send, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  complaintsApi,
  type ChatbotResponse,
  type ChatTurnPayload,
} from "@/integrations/aws/client";
import { useAuth } from "@/contexts/AuthContext";

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
};

const CHAT_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 30000);
const CHAT_STORAGE_KEY = "campusvoice-assistant-chat";

type StoredAssistantState = {
  messages: ChatMessage[];
  response: ChatbotResponse | null;
};

const HeaderAssistant = () => {
  const { user } = useAuth();
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<ChatbotResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const inFlightRef = useRef(false);
  const scopedStorageKey = useMemo(
    () => `${CHAT_STORAGE_KEY}:${user?.id ?? "guest"}`,
    [user?.id]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.sessionStorage.getItem(scopedStorageKey);
      if (!raw) {
        setMessages([]);
        setResponse(null);
        return;
      }
      const parsed = JSON.parse(raw) as StoredAssistantState;
      setMessages(Array.isArray(parsed.messages) ? parsed.messages : []);
      setResponse(parsed.response ?? null);
    } catch {
      setMessages([]);
      setResponse(null);
    }
  }, [scopedStorageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const payload: StoredAssistantState = { messages, response };
    window.sessionStorage.setItem(scopedStorageKey, JSON.stringify(payload));
  }, [messages, response, scopedStorageKey]);

  const askAssistant = async () => {
    if (inFlightRef.current || loading) return;
    if (!message.trim()) return;
    if (!user) {
      setError("Sign in to use the assistant.");
      return;
    }

    const userMessage = message.trim();
    const historyPayload: ChatTurnPayload[] = messages.slice(-6).map((m) => ({
      role: m.role,
      text: m.text,
    }));
    inFlightRef.current = true;
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setMessage("");
    setLoading(true);
    setError(null);

    try {
      const res = await Promise.race([
        complaintsApi.chatbotRespond(userMessage, historyPayload),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("Assistant timeout. Please try again.")), CHAT_TIMEOUT_MS)
        ),
      ]);
      setResponse(res);
      setMessages((prev) => [...prev, { role: "assistant", text: res.reply }]);
    } catch (err) {
      const messageText = err instanceof Error ? err.message : "Assistant is unavailable right now.";
      setError(messageText);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Assistant is taking too long. Please retry, or ask a shorter question.",
        },
      ]);
    } finally {
      inFlightRef.current = false;
      setLoading(false);
    }
  };

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <Bot className="h-4 w-4" />
          AI Assistant
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-full sm:max-w-md p-0">
        <SheetHeader className="border-b p-4">
          <SheetTitle className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            CampusVoice Assistant
          </SheetTitle>
          <SheetDescription>
            80% project context + 20% general guidance for actionable answers.
          </SheetDescription>
        </SheetHeader>

        <div className="flex h-[calc(100vh-130px)] flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="rounded-md border p-3 text-sm text-muted-foreground">
                Ask about complaint status, duplicate checks, category routing, evidence, or trend insights.
              </div>
            )}
            {messages.map((item, index) => (
              <div
                key={`${item.role}-${index}`}
                className={`rounded-lg px-3 py-2 text-sm ${
                  item.role === "user"
                    ? "ml-8 bg-primary text-primary-foreground"
                    : "mr-8 border bg-muted/20"
                } whitespace-pre-line`}
              >
                {item.text}
              </div>
            ))}
            {error && <p className="text-sm text-destructive">{error}</p>}
          </div>

          <div className="border-t p-3">
            {messages.length > 0 && (
              <div className="mb-3 flex justify-end">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setMessages([]);
                    setResponse(null);
                    setError(null);
                    if (typeof window !== "undefined") {
                      window.sessionStorage.removeItem(scopedStorageKey);
                    }
                  }}
                >
                  Clear chat
                </Button>
              </div>
            )}
            {response?.analysis_preview && (
              <div className="mb-3 rounded-md border bg-muted/20 p-3 text-xs space-y-2">
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">
                    Category: {response.analysis_preview.classification.label}
                  </Badge>
                  <Badge variant="secondary">
                    Priority: {response.analysis_preview.classification.priority}
                  </Badge>
                  <Badge variant="secondary">
                    Dept: {response.analysis_preview.classification.department}
                  </Badge>
                </div>
                {response.suggested_title && (
                  <div>
                    <span className="font-medium">Suggested title:</span> {response.suggested_title}
                  </div>
                )}
                {response.follow_up_questions && response.follow_up_questions.length > 0 && (
                  <div className="space-y-1">
                    <div className="font-medium">Helpful follow-ups</div>
                    {response.follow_up_questions.map((question, index) => (
                      <div key={`${question}-${index}`}>- {question}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
            <div className="flex gap-2">
              <Input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type your question"
                disabled={loading}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !loading) {
                    e.preventDefault();
                    void askAssistant();
                  }
                }}
              />
              <Button onClick={askAssistant} disabled={loading || !message.trim()} className="gap-1">
                {loading ? "Thinking..." : "Send"}
                {!loading && <Send className="h-4 w-4" />}
              </Button>
            </div>

            {response && (
              <div className="mt-2 text-xs text-muted-foreground">
                intent: {response.intent}
                {typeof response.intent_confidence === "number"
                  ? ` (${(response.intent_confidence * 100).toFixed(1)}%)`
                  : ""}
              </div>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default HeaderAssistant;
