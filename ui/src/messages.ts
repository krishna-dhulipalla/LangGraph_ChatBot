// src/messages.ts
export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

const ensureSessionId = (): string => {
  let sid = sessionStorage.getItem("session_id");
  if (!sid) {
    sid = crypto.randomUUID();
    sessionStorage.setItem("session_id", sid);
  }
  return sid;
};

const keyFor = (sid: string, tid: string) => `messages:${sid}:${tid}`;

export const loadMessages = (threadId: string): ChatMessage[] => {
  const sid = ensureSessionId();
  const raw = localStorage.getItem(keyFor(sid, threadId));
  return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
};

export const saveMessages = (threadId: string, msgs: ChatMessage[]) => {
  const sid = ensureSessionId();
  localStorage.setItem(keyFor(sid, threadId), JSON.stringify(msgs));
};

export const clearMessages = (threadId: string) => {
  const sid = ensureSessionId();
  localStorage.removeItem(keyFor(sid, threadId));
};
