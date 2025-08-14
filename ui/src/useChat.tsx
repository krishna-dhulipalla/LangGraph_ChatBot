// src/useChat.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import type { ThreadMeta } from "./threads";
import {
  loadThreads,
  newThreadMeta,
  upsertThread,
  removeThread,
} from "./threads";
import type { ChatMessage } from "./messages";
import { loadMessages, saveMessages, clearMessages } from "./messages";

export function useChat() {
  const [threads, setThreads] = useState<ThreadMeta[]>(() => loadThreads());
  const [active, setActive] = useState<ThreadMeta>(
    () => threads[0] ?? newThreadMeta()
  );
  const [messagesByThread, setMessagesByThread] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasFirstToken, setHasFirstToken] = useState(false); // NEW
  const firstTokenSeenRef = useRef(false);
  const esRef = useRef<EventSource | null>(null);

  // Load messages whenever the active thread changes (covers initial mount too)
  useEffect(() => {
    if (!active?.id) return;
    setMessagesByThread((prev) => ({
      ...prev,
      [active.id]: loadMessages(active.id),
    }));
  }, [active?.id]);

  // Close SSE on unmount
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, []);

  const setActiveThread = useCallback((t: ThreadMeta) => {
    setActive(t);
    upsertThread({ ...t, lastAt: new Date().toISOString() });
    setThreads(loadThreads());
  }, []);

  const newChat = useCallback(() => {
    const t = newThreadMeta();
    setActive(t);
    upsertThread(t);
    setThreads(loadThreads());
  }, []);

  const clearChat = useCallback(() => {
    if (!active?.id) return;
    setMessagesByThread((prev) => ({ ...prev, [active.id]: [] }));
    clearMessages(active.id);
  }, [active?.id]);

  const deleteThread = useCallback(
    (tid: string) => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
      setMessagesByThread((prev) => {
        const copy = { ...prev };
        delete copy[tid];
        return copy;
      });
      removeThread(tid);
      setThreads(loadThreads());

      if (active?.id === tid) {
        const list = loadThreads();
        if (list.length) setActive(list[0]);
        else {
          const t = newThreadMeta();
          setActive(t);
          upsertThread(t);
          setThreads(loadThreads());
        }
      }
    },
    [active?.id]
  );

  const persist = useCallback((tid: string, msgs: ChatMessage[]) => {
    saveMessages(tid, msgs);
  }, []);

  const appendMsg = useCallback(
    (tid: string, msg: ChatMessage) => {
      setMessagesByThread((prev) => {
        const arr = prev[tid] ?? [];
        const next = [...arr, msg];
        persist(tid, next);
        return { ...prev, [tid]: next };
      });
    },
    [persist]
  );

  // ✅ keep mutateLastAssistant as a useCallback
  const mutateLastAssistant = useCallback(
    (tid: string, chunk: string) => {
      setMessagesByThread((prev) => {
        const arr = (prev[tid] ?? []) as ChatMessage[]; // keep strict type
        if (arr.length === 0) return prev;

        const last = arr[arr.length - 1];
        let next: ChatMessage[];

        if (last.role === "assistant") {
          const merged: ChatMessage = {
            ...last,
            content: (last.content ?? "") + (chunk ?? ""),
          };
          next = [...arr.slice(0, -1), merged];
        } else {
          // 👈 important: literal role type to avoid widening to string
          next = [
            ...arr,
            {
              id: crypto.randomUUID(),
              role: "assistant" as const,
              content: chunk,
            },
          ];
        }

        persist(tid, next); // ChatMessage[]
        return { ...prev, [tid]: next }; // Record<string, ChatMessage[]>
      });
    },
    [persist]
  );

  // const makeMsg = (
  //   role: "user" | "assistant",
  //   content: string
  // ): ChatMessage => ({
  //   id: crypto.randomUUID(),
  //   role,
  //   content,
  // });

  const send = useCallback(
    (text: string) => {
      if (!active?.id) return;
      const thread_id = active.id;

      // optimistic UI
      appendMsg(thread_id, {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
      });
      // appendMsg(thread_id, makeMsg("assistant", ""));

      // bump thread meta (derive title from first user msg if needed)
      const title =
        active.title && active.title !== "New chat"
          ? active.title
          : text.slice(0, 40);
      const bumped = { ...active, lastAt: new Date().toISOString(), title };
      setActive(bumped);
      upsertThread(bumped);
      setThreads(loadThreads());

      // Close any prior stream
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }

      if (typeof window === "undefined") return; // SSR guard

      // Start SSE
      const url = new URL("/chat", window.location.origin);
      url.searchParams.set("message", text);
      url.searchParams.set("thread_id", thread_id);

      const es = new EventSource(url.toString());
      esRef.current = es;
      setIsStreaming(true);
      setHasFirstToken(false);
      firstTokenSeenRef.current = false;

      es.addEventListener("token", (ev: MessageEvent) => {
        const data = (ev as MessageEvent<string>).data ?? "";
        const hasVisibleChars = data.trim().length > 0; // 👈 NEW

        // flip the flag only when the first *visible* token arrives
        if (!firstTokenSeenRef.current && hasVisibleChars) {
          firstTokenSeenRef.current = true;
          setHasFirstToken(true);
        }

        mutateLastAssistant(thread_id, data);
      });

      const close = () => {
        es.close();
        esRef.current = null;
        setIsStreaming(false);
      };

      es.addEventListener("done", close);
      es.onerror = close;
    },
    [active, appendMsg, mutateLastAssistant]
  );

  return {
    threads,
    active,
    messages: messagesByThread[active?.id ?? ""] ?? [],
    setActiveThread,
    newChat,
    clearChat,
    deleteThread,
    send,
    isStreaming,
    hasFirstToken,
  };
}
