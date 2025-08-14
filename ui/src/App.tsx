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
    title: "Quick intro to Krishna",
    text: "Give me a 90-second intro to Krishna Vamsi Dhulipalla—recent work, top strengths, and impact.",
  },
  {
    title: "Get Krishna’s resume",
    text: "Share Krishna’s latest resume and provide a download link.",
  },
  {
    title: "What this agent can do",
    text: "What tools and actions can you perform for me? Show examples and how to use them.",
  },
  {
    title: "Schedule/modify a meeting",
    text: "Schedule a 30-minute meeting with Krishna next week and show how I can reschedule or cancel.",
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
    isStreaming,
    hasFirstToken,
  } = useChat();

  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const prevThreadId = useRef<string | null>(null);

  // Scroll on message changes
  useEffect(() => {
    const currentThreadId = active?.id ?? null;

    // If the thread changed, scroll instantly to bottom
    if (currentThreadId !== prevThreadId.current) {
      prevThreadId.current = currentThreadId;
      bottomRef.current?.scrollIntoView({ behavior: "auto" }); // instant scroll
    } else {
      // If same thread but messages changed, smooth scroll
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, active?.id]);

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
          {/* View Source on GitHub */}
          <a
            href="https://github.com/krishna-dhulipalla/LangGraph_ChatBot"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center justify-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 px-3 py-2 text-sm text-zinc-300"
            title="View the source code on GitHub"
          >
            {/* GitHub Icon */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="h-4 w-4"
            >
              <path
                fillRule="evenodd"
                d="M12 0C5.37 0 0 5.37 0 
          12c0 5.3 3.438 9.8 8.205 
          11.387.6.113.82-.262.82-.58 
          0-.287-.01-1.045-.015-2.05-3.338.724-4.042-1.61-4.042-1.61-.546-1.385-1.333-1.754-1.333-1.754-1.09-.745.083-.73.083-.73 
          1.205.085 1.84 1.238 1.84 1.238 
          1.07 1.835 2.807 1.305 3.492.997.107-.775.418-1.305.762-1.605-2.665-.3-5.466-1.334-5.466-5.93 
          0-1.31.468-2.38 1.235-3.22-.124-.303-.536-1.523.117-3.176 
          0 0 1.008-.322 3.3 1.23a11.5 11.5 
          0 013.003-.404c1.018.005 2.045.138 
          3.003.404 2.29-1.552 3.297-1.23 
          3.297-1.23.655 1.653.243 2.873.12 
          3.176.77.84 1.233 1.91 1.233 3.22 
          0 4.61-2.803 5.625-5.475 5.92.43.372.823 
          1.102.823 2.222 0 1.606-.015 2.898-.015 3.293 
          0 .32.218.698.825.58C20.565 21.796 24 
          17.297 24 12c0-6.63-5.37-12-12-12z"
                clipRule="evenodd"
              />
            </svg>
            View Source
          </a>
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
            {/* LinkedIn */}
            <a
              href="https://www.linkedin.com/in/krishnavamsidhulipalla/"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl px-3 py-1.5 text-sm border border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
              title="LinkedIn"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-5 w-5 text-zinc-400 hover:text-zinc-200"
              >
                <path
                  d="M19 0h-14c-2.76 0-5 2.24-5 5v14c0 
        2.76 2.24 5 5 5h14c2.76 0 5-2.24 
        5-5v-14c0-2.76-2.24-5-5-5zm-11 
        19h-3v-10h3v10zm-1.5-11.27c-.97 
        0-1.75-.79-1.75-1.76s.78-1.76 
        1.75-1.76 1.75.79 
        1.75 1.76-.78 1.76-1.75 
        1.76zm13.5 11.27h-3v-5.5c0-1.31-.02-3-1.83-3-1.83 
        0-2.12 1.43-2.12 2.9v5.6h-3v-10h2.88v1.36h.04c.4-.75 
        1.38-1.54 2.85-1.54 3.05 0 3.61 
        2.01 3.61 4.63v5.55z"
                />
              </svg>
            </a>

            {/* GitHub */}
            <a
              href="https://github.com/krishna-dhulipalla"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl px-3 py-1.5 text-sm border border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
              title="GitHub"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-5 w-5 text-zinc-400 hover:text-zinc-200"
              >
                <path
                  fillRule="evenodd"
                  d="M12 0C5.37 0 0 5.37 0 
          12c0 5.3 3.438 9.8 8.205 
          11.387.6.113.82-.262.82-.58 
          0-.287-.01-1.045-.015-2.05-3.338.724-4.042-1.61-4.042-1.61-.546-1.385-1.333-1.754-1.333-1.754-1.09-.745.083-.73.083-.73 
          1.205.085 1.84 1.238 1.84 1.238 
          1.07 1.835 2.807 1.305 3.492.997.107-.775.418-1.305.762-1.605-2.665-.3-5.466-1.334-5.466-5.93 
          0-1.31.468-2.38 1.235-3.22-.124-.303-.536-1.523.117-3.176 
          0 0 1.008-.322 3.3 1.23a11.5 11.5 
          0 013.003-.404c1.018.005 2.045.138 
          3.003.404 2.29-1.552 3.297-1.23 
          3.297-1.23.655 1.653.243 2.873.12 
          3.176.77.84 1.233 1.91 1.233 3.22 
          0 4.61-2.803 5.625-5.475 5.92.43.372.823 
          1.102.823 2.222 0 1.606-.015 2.898-.015 3.293 
          0 .32.218.698.825.58C20.565 21.796 24 
          17.297 24 12c0-6.63-5.37-12-12-12z"
                  clipRule="evenodd"
                />
              </svg>
            </a>
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
            <div className="mx-auto max-w-3xl space-y-3 relative">
              {messages.map((m, idx) => {
                const isAssistant = m.role === "assistant";
                const emptyAssistant =
                  isAssistant && (!m.content || m.content.trim() === "");
                if (emptyAssistant) return null; // hide blank bubble
                const key = m.id ?? `m-${idx}`; // NEW stable key
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
              {/* Thinking indicator (only BEFORE first token) */}
              {isStreaming && !hasFirstToken && (
                <div className="pointer-events-none relative top-3 left-0 bottom-0 translate-y-2 z-20">
                  <TypingDots />
                </div>
              )}
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
