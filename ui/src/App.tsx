import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { BookmarkIcon } from "@heroicons/react/24/outline";
import { TrashIcon } from "@heroicons/react/24/outline";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SparklesIcon } from "@heroicons/react/24/outline";

// const API_PATH = "/chat";
// const THREAD_KEY = "lg_thread_id";

import { useChat } from "./useChat";
import type { ThreadMeta } from "./threads";

const SUGGESTIONS = [
  {
    title: "Overview of Krishna’s experience",
    text: "Give me an overview of Krishna Vamsi Dhulipalla's work experience across different roles?",
  },
  {
    title: "Data science stack",
    text: "What programming languages and tools does Krishna use for data science?",
  },
  {
    title: "Chatbot capabilities",
    text: "Can this chatbot tell me what Krishna's chatbot architecture looks like and how it works?",
  },
];

// --- Helpers for message actions ---
const copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    console.error("Failed to copy text to clipboard");
  }
};

const getLastUserMessage = (msgs: { role: string; content: string }[]) =>
  [...msgs].reverse().find((m) => m.role === "user") || null;

type Role = "user" | "assistant";
type Message = { id: string; role: Role; content: string };

function uid() {
  return Math.random().toString(36).slice(2);
}

export default function App() {
  const {
    threads,
    active,
    messages,
    setActiveThread,
    newChat,
    clearChat,
    deleteThread,
    send,
  } = useChat();

  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Scroll on message changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const isStreaming = useMemo(() => {
    const last = messages[messages.length - 1];
    return last?.role === "assistant" && (last.content ?? "") === "";
  }, [messages]);

  const handleShare = async () => {
    const url = window.location.href;
    const title = document.title || "My Chat";
    try {
      if (navigator.share) {
        await navigator.share({ title, url });
      } else {
        await navigator.clipboard.writeText(url);
        // optionally toast: "Link copied"
      }
    } catch {
      // ignored
    }
  };

  const handleBookmark = () => {
    // Browsers don't allow programmatic bookmarks; show the right shortcut.
    const isMac = navigator.platform.toUpperCase().includes("MAC");
    const combo = isMac ? "⌘ + D" : "Ctrl + D";
    alert(`Press ${combo} to bookmark this page.`);
  };

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming) return;
    send(text);
    setInput("");
  }, [input, isStreaming, send]);

  const selectSuggestion = useCallback((text: string) => {
    setInput(text);
    requestAnimationFrame(() => inputRef.current?.focus());
  }, []);

  const sendSuggestion = useCallback(
    (text: string) => {
      if (isStreaming) return;
      setInput(text);
      setTimeout(() => sendMessage(), 0);
    },
    [isStreaming, sendMessage]
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  const REGEN_PREFIX = "Regenerate this response with a different angle:\n\n";

  const sendPrefixed = useCallback(
    (prefix: string, text: string) => {
      if (!text || isStreaming) return;
      // your hook’s `send` already appends messages & streams
      send(`${prefix}${text}`);
      setInput("");
    },
    [send, isStreaming]
  );

  // Sidebar
  const Sidebar = useMemo(
    () => (
      <aside className="hidden md:flex w-64 shrink-0 flex-col bg-zinc-950/80 border-r border-zinc-800/60">
        <div className="p-4 border-b border-zinc-800/60">
          <h1 className="flex items-center text-zinc-100 font-semibold tracking-tight">
            <SparklesIcon className="h-4 w-4 text-zinc-300 mr-2" />
            ChatK
          </h1>
          <p className="text-xs text-zinc-400 mt-1">
            Chatbot ID:{" "}
            <span className="font-mono">
              {active?.id ? active.id.slice(0, 8) : "…"}{" "}
            </span>
          </p>
        </div>
        <div className="p-3 space-y-2">
          <button
            className="w-full rounded-xl bg-emerald-600 text-white hover:bg-emerald-500 px-3 py-2 text-sm"
            onClick={newChat}
            title="Start a new session"
          >
            New Chat
          </button>
          <button
            className="w-full rounded-xl bg-zinc-800 text-zinc-200 hover:bg-zinc-700 px-3 py-2 text-sm"
            onClick={clearChat}
            title="Clear current messages"
          >
            Clear Chat
          </button>
        </div>

        {/* Thread list */}
        <div className="px-3 pb-3 space-y-1 overflow-y-auto">
          {threads.map((t: ThreadMeta) => (
            <div
              key={t.id}
              className={`group w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-zinc-800 cursor-pointer ${
                t.id === active?.id ? "bg-zinc-800" : ""
              }`}
              onClick={() => setActiveThread(t)}
              title={t.id}
            >
              <div className="flex-1 min-w-0 p-1 hover:bg-gray-100">
                <div className="text-sm">
                  {t.title && t.title.length > 20
                    ? t.title.slice(0, 20) + "..."
                    : t.title || "Untitled"}
                </div>
                <div
                  className="text-zinc-500 truncate"
                  style={{ fontSize: "10px", fontStyle: "italic" }}
                >
                  {new Date(t.lastAt).toLocaleString()}
                </div>
              </div>

              {/* Delete button (shows on hover) */}
              <button
                type="button"
                className="opacity-0 group-hover:opacity-100 shrink-0 rounded-md p-1 border border-zinc-700/60 bg-zinc-900/60 hover:bg-zinc-800/80"
                title="Delete thread"
                aria-label="Delete thread"
                onClick={(e) => {
                  e.stopPropagation(); // don't switch threads
                  if (
                    window.confirm("Delete this thread? This cannot be undone.")
                  ) {
                    deleteThread(t.id);
                  }
                }}
              >
                <TrashIcon className="h-4 w-4 text-zinc-300" />
              </button>
            </div>
          ))}
        </div>

        <div className="mt-auto p-3 text-xs text-zinc-500">
          Tip: Press <kbd className="px-1 bg-zinc-800 rounded">Enter</kbd> to
          send,
          <span className="mx-1" />{" "}
          <kbd className="px-1 bg-zinc-800 rounded">Shift+Enter</kbd> for
          newline.
        </div>
      </aside>
    ),
    [active?.id, clearChat, newChat, setActiveThread, deleteThread, threads]
  );

  return (
    <div className="h-screen w-screen bg-[#0b0b0f] text-zinc-100 flex">
      {Sidebar}

      {/* Main column */}
      <main className="flex-1 flex flex-col">
        {/* Header minimal */}
        <div className="h-12 shrink-0 flex items-center justify-between px-3 md:px-6 border-b border-zinc-800/60 bg-zinc-950/60 backdrop-blur">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-medium">Krishna’s Assistant</span>
          </div>
          <div className="text-xs text-zinc-400 md:hidden">
            ID: {active?.id ? active.id.slice(0, 8) : "…"}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={handleShare}
              className="rounded-xl px-3 py-1.5 text-sm border border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
              title="Share"
            >
              Share
            </button>
            <button
              onClick={handleBookmark}
              className="rounded-xl px-3 py-1.5 text-sm border border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
              title="Bookmark"
            >
              <BookmarkIcon className="h-5 w-5 text-zinc-400 hover:text-zinc-200" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-3 md:px-6 py-4">
          {messages.length === 0 ? (
            <EmptyState onSelect={selectSuggestion} onSend={sendSuggestion} />
          ) : (
            <div className="mx-auto max-w-3xl space-y-3">
              {messages.map((m, idx) => {
                const isAssistant = m.role === "assistant";
                const emptyAssistant =
                  isAssistant && (!m.content || m.content.trim() === "");
                if (emptyAssistant) return null; // hide blank bubble
                const key =
                  (m as Message).id ??
                  `m-${idx}-${m.role}-${(m.content || "").length}-${uid()}`;
                return (
                  <div key={key} className={isAssistant ? "group" : undefined}>
                    {/* bubble row */}
                    <div
                      className={`flex ${
                        isAssistant ? "justify-start" : "justify-end"
                      }`}
                    >
                      <div
                        className={`max-w-[85%] md:max-w-[75%] leading-relaxed tracking-tight rounded-2xl px-4 py-3 shadow-sm ${
                          isAssistant
                            ? "bg-zinc-900/80 text-zinc-100 border border-zinc-800/60"
                            : "bg-emerald-600/90 text-white"
                        }`}
                      >
                        {isAssistant ? (
                          <>
                            {isStreaming ? (
                              <TypingDots />
                            ) : (
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                  a: (props) => (
                                    <a
                                      {...props}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="underline text-blue-400 hover:text-blue-600"
                                    />
                                  ),
                                  p: (props) => (
                                    <p className="mb-2 last:mb-0" {...props} />
                                  ),
                                  ul: (props) => (
                                    <ul
                                      className="list-disc list-inside mb-2 last:mb-0"
                                      {...props}
                                    />
                                  ),
                                  ol: (props) => (
                                    <ol
                                      className="list-decimal list-inside mb-2 last:mb-0"
                                      {...props}
                                    />
                                  ),
                                  li: (props) => (
                                    <li className="ml-4 mb-1" {...props} />
                                  ),
                                  code: (
                                    props: React.HTMLAttributes<HTMLElement> & {
                                      inline?: boolean;
                                    }
                                  ) => {
                                    // react-markdown v8+ passes 'node', 'inline', etc. in props, but types may not include 'inline'
                                    const {
                                      className,
                                      children,
                                      inline,
                                      ...rest
                                    } = props;
                                    const isInline = inline;
                                    return isInline ? (
                                      <code
                                        className="bg-zinc-800/80 px-1 py-0.5 rounded"
                                        {...rest}
                                      >
                                        {children}
                                      </code>
                                    ) : (
                                      <pre className="overflow-x-auto rounded-xl border border-zinc-800/60 bg-zinc-950/80 p-3 mb-2">
                                        <code className={className} {...rest}>
                                          {children}
                                        </code>
                                      </pre>
                                    );
                                  },
                                }}
                              >
                                {m.content}
                              </ReactMarkdown>
                            )}
                          </>
                        ) : (
                          m.content
                        )}
                      </div>
                    </div>

                    {/* actions row – only for assistant & only when not streaming */}
                    {isAssistant && !isStreaming && (
                      <div className="mt-1 pl-1 flex justify-start">
                        <MsgActions
                          content={m.content}
                          onEdit={() => {
                            // Prefill composer with the *last user* prompt (ChatGPT-style “Edit”)
                            const lastUser = getLastUserMessage(messages);
                            setInput(lastUser ? lastUser.content : m.content);
                            requestAnimationFrame(() =>
                              inputRef.current?.focus()
                            );
                          }}
                          onRegenerate={() => {
                            // Resend last user prompt
                            const lastUser = getLastUserMessage(messages);
                            if (!lastUser) return;
                            sendPrefixed(REGEN_PREFIX, lastUser.content);
                          }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
        {/* Warm bottom glow (simple bar + blur)
        <div
          aria-hidden
          className="fixed inset-x-0 bottom-0 z-30 pointer-events-none"
        >
          <div className="mx-auto max-w-3xl relative h-0">
            <div className="absolute left-6 right-6 bottom-0 h-[6px] rounded-full bg-amber-300/40 blur-3xl" />
          </div>
        </div> */}

        {/* Composer */}
        <div className="shrink-0 border-t border-zinc-800/60 bg-zinc-950/80">
          <div className="mx-auto max-w-3xl p-3 md:p-4">
            <div className="flex gap-2 items-end">
              <div className="relative flex-1">
                <textarea
                  ref={inputRef}
                  className="w-full flex-1 resize-none rounded-2xl bg-zinc-900 text-zinc-100 placeholder-zinc-500 p-3 pr-8 outline-none focus:ring-2 focus:ring-emerald-500/60 min-h-[56px] max-h-48 border border-zinc-800/60"
                  placeholder="Type a message…"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  disabled={isStreaming}
                />
                {input && (
                  <button
                    onClick={() => {
                      setInput("");
                      requestAnimationFrame(() => inputRef.current?.focus());
                    }}
                    className="absolute right-2 top-2.5 h-6 w-6 rounded-md border border-zinc-800 bg-zinc-950/70 hover:bg-zinc-800/70 text-zinc-400"
                    title="Clear"
                    aria-label="Clear input"
                  >
                    ×
                  </button>
                )}
              </div>
              <button
                className="rounded-2xl px-4 py-3 bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={sendMessage}
                disabled={!input.trim() || isStreaming}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function MsgActions({
  content,
  onEdit,
  onRegenerate,
}: {
  content: string;
  onEdit: () => void;
  onRegenerate: () => void;
}) {
  return (
    <div className="mt-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
      <button
        onClick={() => copyToClipboard(content)}
        className="text-xs px-2 py-1 rounded border border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
        title="Copy"
      >
        Copy
      </button>
      <button
        onClick={onEdit}
        className="text-xs px-2 py-1 rounded border border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
        title="Edit this as new prompt"
      >
        Edit
      </button>
      <button
        onClick={onRegenerate}
        className="text-xs px-2 py-1 rounded border border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
        title="Regenerate"
      >
        Regenerate
      </button>
    </div>
  );
}

function EmptyState({
  onSelect,
  onSend,
}: {
  onSelect: (text: string) => void;
  onSend: (text: string) => void;
}) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="mx-auto max-w-3xl w-full px-3 md:px-6">
        <div className="text-center text-zinc-400 mb-6">
          <h2 className="text-xl text-zinc-200 mb-2">Ask me anything</h2>
          <p className="text-sm">
            The agent can call tools and remember the conversation (per chatbot
            id).
          </p>
        </div>

        {/* Starter prompts */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {SUGGESTIONS.map((s) => (
            <div
              role="button"
              tabIndex={0}
              onClick={() => onSelect(s.text)}
              className="group text-left rounded-2xl border border-zinc-800/60 bg-zinc-900/60 hover:bg-zinc-900/90 transition-colors p-4 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/60"
              title="Click to prefill. Use the arrow to send."
            >
              <div className="flex items-start gap-3">
                <div className="shrink-0 h-8 w-8 rounded-xl bg-zinc-800/80 flex items-center justify-center">
                  <span className="text-sm text-zinc-300">💡</span>
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium text-zinc-200">
                    {s.title}
                  </div>
                  <div className="text-xs text-zinc-400 mt-1 line-clamp-2">
                    {s.text}
                  </div>
                </div>
                <button
                  type="button"
                  aria-label="Send this suggestion"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSend(s.text);
                  }}
                  className="shrink-0 rounded-xl border border-zinc-700/60 bg-zinc-950/60 px-2 py-2 hover:bg-zinc-900/80"
                  title="Send now"
                >
                  {/* Arrow icon */}
                  <svg
                    viewBox="0 0 24 24"
                    className="h-4 w-4 text-zinc-300 group-hover:text-emerald-400"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M7 17L17 7M7 7h10v10" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 align-middle">
      <span className="sr-only">Assistant is typing…</span>
      <span className="h-1.5 w-1.5 rounded-full bg-zinc-300 animate-bounce [animation-delay:-0.2s]" />
      <span className="h-1.5 w-1.5 rounded-full bg-zinc-300 animate-bounce" />
      <span className="h-1.5 w-1.5 rounded-full bg-zinc-300 animate-bounce [animation-delay:0.2s]" />
    </span>
  );
}
