// threads.ts
export type ThreadMeta = {
  id: string;
  title: string;
  createdAt: string;
  lastAt: string;
};

const ensureSessionId = (): string => {
  let sid = sessionStorage.getItem("session_id");
  if (!sid) {
    sid = crypto.randomUUID();
    sessionStorage.setItem("session_id", sid);
  }
  return sid;
};

const keyFor = (sid: string) => `threads:${sid}`;

export const loadThreads = (): ThreadMeta[] => {
  const sid = ensureSessionId();
  const raw = localStorage.getItem(keyFor(sid));
  return raw ? JSON.parse(raw) : [];
};

export const saveThreads = (threads: ThreadMeta[]) => {
  const sid = ensureSessionId();
  localStorage.setItem(keyFor(sid), JSON.stringify(threads));
};

export const upsertThread = (t: ThreadMeta) => {
  const threads = loadThreads();
  const i = threads.findIndex((x) => x.id === t.id);
  if (i >= 0) threads[i] = t;
  else threads.push(t);
  saveThreads(threads);
};

export const removeThread = (id: string) => {
  const threads = loadThreads().filter((t) => t.id !== id);
  saveThreads(threads);
};

export const newThreadMeta = (title = "New chat"): ThreadMeta => {
  const now = new Date().toISOString();
  return { id: crypto.randomUUID(), title, createdAt: now, lastAt: now };
};
