import os
import json
import re
import hashlib
import numpy as np
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Annotated, Sequence, Dict, Optional, List, Type
from typing_extensions import Literal, TypedDict
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# import function from api.py
from .api import get_gcal_service

from dateutil import parser as date_parser
from datetime import datetime, timedelta
import pytz

GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

CREDS_DIR = Path("backend/credentials")
CREDS_DIR.mkdir(parents=True, exist_ok=True)
CLIENT_SECRET_FILE = CREDS_DIR / "credentials.json"  # download from Google Cloud
TOKEN_FILE = CREDS_DIR / "token.json"

if not CLIENT_SECRET_FILE.exists():
        raise FileNotFoundError(
            f"Missing OAuth client file: {CLIENT_SECRET_FILE}\n"
            "Create an OAuth 2.0 Client ID (Desktop) and download JSON."
        )

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("🚨 NVIDIA_API_KEY not found in environment!")

# Constants
FAISS_PATH = "backend/data/faiss_store/v41_1000-250"
CHUNKS_PATH = "backend/data/all_chunks.json"

# Validate files
if not Path(FAISS_PATH).exists():
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")
if not Path(CHUNKS_PATH).exists():
    raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")

KRISHNA_BIO = """Krishna Vamsi Dhulipalla completed masters in Computer Science at Virginia Tech, awarded degree in december 2024, with over 3 years of experience across data engineering, machine learning research, and real-time analytics. He specializes in building scalable data systems and intelligent LLM-powered agents, with strong expertise in Python, PyTorch,Langgraph, autogen Hugging Face Transformers, and end-to-end ML pipelines."""

# Load resources
def load_chunks(path=CHUNKS_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_faiss(path=FAISS_PATH, model_name="sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_faiss()
all_chunks = load_chunks()
all_texts = [chunk["text"] for chunk in all_chunks]
metadatas = [chunk["metadata"] for chunk in all_chunks]
bm25_retriever = BM25Retriever.from_texts(texts=all_texts, metadatas=metadatas)

K_PER_QUERY    = 10         # how many from each retriever
TOP_K          = 8         # final results to return
RRF_K          = 60        # reciprocal-rank-fusion constant
RERANK_TOP_N   = 50        # rerank this many fused hits
MMR_LAMBDA     = 0.7       # 0..1 (higher favors query relevance; lower favors diversity)
CE_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ALPHA = 0.7

from sentence_transformers import CrossEncoder
_cross_encoder = CrossEncoder(CE_MODEL)

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: mxd, B: nxd, both should be L2-normalized
    return A @ B.T

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def _mmr_select(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, mmr_lambda: float):
    # returns list of selected indices using MMR
    selected = []
    remaining = list(range(cand_vecs.shape[0]))

    # precompute similarities
    q_sim = (cand_vecs @ query_vec.reshape(-1, 1)).ravel()  # cosine since normalized
    doc_sims = _cosine_sim_matrix(cand_vecs, cand_vecs)

    # pick first by highest query similarity
    first = int(np.argmax(q_sim))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < k:
        # for each remaining, compute MMR score = λ * Sim(q, d) - (1-λ) * max Sim(d, s in selected)
        sub = np.array(remaining)
        sim_to_selected = doc_sims[np.ix_(sub, selected)].max(axis=1)
        mmr_scores = mmr_lambda * q_sim[sub] - (1.0 - mmr_lambda) * sim_to_selected
        nxt = int(sub[np.argmax(mmr_scores)])
        selected.append(nxt)
        remaining.remove(nxt)

    return selected

@tool("retriever")
def retriever(query: str) -> list[str]:
    """Retrieve relevant chunks from the profile using FAISS + BM25, fused with RRF."""
    # ensure both retrievers return K_PER_QUERY
    # For BM25Retriever in LangChain this is usually `.k`
    try:
        bm25_retriever.k = K_PER_QUERY
    except Exception:
        pass

    vec_hits = vectorstore.similarity_search_with_score(query, k=K_PER_QUERY)  # [(Document, score)]
    bm_hits  = bm25_retriever.invoke(query)                                    # [Document]

    # --- fuse via RRF (rank-only) ---
    fused = defaultdict(lambda: {
        "rrf": 0.0,
        "vec_rank": None, "bm_rank": None,
        "content": None, "metadata": None,
    })

    for rank, (doc, _score) in enumerate(vec_hits):
        key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        fused[key]["rrf"] += 1.0 / (rank + 1 + RRF_K)
        fused[key]["vec_rank"] = rank
        fused[key]["content"] = doc.page_content    # keep FULL text
        fused[key]["metadata"] = getattr(doc, "metadata", {}) or {}

    for rank, doc in enumerate(bm_hits):
        key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        fused[key]["rrf"] += 1.0 / (rank + 1 + RRF_K)
        fused[key]["bm_rank"] = rank
        fused[key]["content"] = doc.page_content    # keep FULL text
        fused[key]["metadata"] = getattr(doc, "metadata", {}) or {}

    items = list(fused.values())
    items.sort(key=lambda x: x["rrf"], reverse=True)

    # --- cross-encoder rerank on top-N (keeps exact text; just reorders) ---
    topN = items[:RERANK_TOP_N] if RERANK_TOP_N > 0 else items
    try:
        pairs = [(query, it["content"] or "") for it in topN]
        ce_scores = _cross_encoder.predict(pairs)  # higher is better
        for it, s in zip(topN, ce_scores):
            it["rerank"] = float(s)
        topN.sort(key=lambda x: x.get("rerank", 0.0), reverse=True)
    except Exception as e:
        # if CE fails, fall back to RRF order
        for it in topN:
            it["rerank"] = it["rrf"]

    # --- MMR diversity on the reranked list (uses your HF embeddings) ---
    try:
        # embed the query + candidates; normalize to cosine space
        emb_fn = getattr(vectorstore, "embedding_function", embeddings)
        q_vec  = np.array(emb_fn.embed_query(query), dtype=np.float32).reshape(1, -1)
        d_vecs = np.array(emb_fn.embed_documents([it["content"] or "" for it in topN]), dtype=np.float32)

        q_vec  = _l2_normalize(q_vec)[0]     # (d,)
        d_vecs = _l2_normalize(d_vecs)       # (N, d)

        sel_idx = _mmr_select(q_vec, d_vecs, k=TOP_K, mmr_lambda=MMR_LAMBDA)
        final_items = [topN[i] for i in sel_idx]
    except Exception as e:
        # fallback: take first TOP_K after rerank
        final_items = topN[:TOP_K]

    # --- return verbatim content, with soft dedupe by (source, first 300 normalized chars) ---
    results = []
    seen = set()
    for it in final_items:
        content = it["content"] or ""
        meta = it["metadata"] or {}
        source = meta.get("source", "")

        # fingerprint for dedupe (does NOT modify returned text)
        clean = re.sub(r"\W+", "", content.lower())[:300]
        fp = (source, clean)
        if fp in seen:
            continue
        seen.add(fp)
        results.append(content)

        if len(results) >= TOP_K:
            break
        
    # optional: quick debug
    # from pprint import pprint
    # pprint([{
    #     "content": i["content"],
    #   "src": (i["metadata"] or {}).get("source", ""),
    #   "rrf": round(i["rrf"], 6),
    #   "vec_rank": i["vec_rank"],
    #   "bm_rank": i["bm_rank"],
    # } for i in final_items], width=120)

    return results

# --- memory globals ---
MEM_FAISS_PATH = os.getenv("MEM_FAISS_PATH", "/data/memory_faiss")
mem_embeddings = embeddings
memory_vs = None
memory_dirty = False
memory_write_count = 0
MEM_AUTOSAVE_EVERY = 20  

def _ensure_memory_vs():
    global memory_vs
    if memory_vs is None:
        try:
            memory_vs = FAISS.load_local(MEM_FAISS_PATH, mem_embeddings, allow_dangerous_deserialization=True)
        except Exception:
            memory_vs = None
    return memory_vs

def _thread_id_from_config(config) -> str:
    return (config or {}).get("configurable", {}).get("thread_id", "default")

@tool("memory_search")
def memory_search(query: str, thread_id: Optional[str] = None) -> list[str]:
    """Search long-term memory (FAISS) for relevant text chunks."""
    vs = _ensure_memory_vs()
    if vs is None:
        return []
    docs = vs.similarity_search(query, k=6)
    return [d.page_content for d in docs if not thread_id or d.metadata.get("thread_id") == thread_id]

def memory_add(text: str, thread_id: str):
    global memory_vs, memory_dirty, memory_write_count
    if memory_vs is None:
        memory_vs = FAISS.from_texts(
            [text],
            mem_embeddings,
            metadatas=[{"thread_id": thread_id, "scope": "memory", "ts": datetime.utcnow().isoformat()}],
        )
    else:
        memory_vs.add_texts(
            [text],
            metadatas=[{"thread_id": thread_id, "scope": "memory", "ts": datetime.utcnow().isoformat()}],
        )
    memory_dirty = True
    memory_write_count += 1
    
def memory_flush():
    global memory_vs, memory_dirty, memory_write_count
    if memory_vs is not None and memory_dirty:
        memory_vs.save_local(MEM_FAISS_PATH)
        memory_dirty = False
        memory_write_count = 0
        
# def _get_gcal_service():
#     creds = None
#     if TOKEN_FILE.exists():
#         creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), GOOGLE_SCOPES)
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())  # type: ignore
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), GOOGLE_SCOPES)
#             # This opens a local browser once per server instance to authorize
#             creds = flow.run_local_server(port=8765, prompt="consent")
#         with open(TOKEN_FILE, "w") as f:
#             f.write(creds.to_json())
#     return build("calendar", "v3", credentials=creds)

class Attendee(TypedDict):
    email: str
    optional: Optional[bool]
    
@tool("schedule_meeting")
def schedule_meeting(
    title: str,
    start_rfc3339: str,   # e.g., "2025-08-13T10:00:00-05:00"
    end_rfc3339: str,     # e.g., "2025-08-13T10:30:00-05:00"
    attendees: Optional[List[Attendee]] = None, # [{"email":"a@x.com"}, ...]
    description: Optional[str] = None,
    location: Optional[str] = None,
    calendar_id: str = "primary",
    make_meet_link: bool = True,
) -> str:
    """
    Create a Google Calendar event (and optional Google Meet link).
    Returns a human-readable confirmation.
    """
    svc = get_gcal_service()

    body = {
        "summary": title,
        "description": description or "",
        "location": location or "",
        "start": {"dateTime": start_rfc3339},
        "end": {"dateTime": end_rfc3339},
        "attendees": [{"email": a["email"], "optional": a.get("optional", False)} for a in (attendees or [])],
    }

    params = {}
    if make_meet_link:
        body["conferenceData"] = {
            "createRequest": {"requestId": str(uuid4())}
        }
        params["conferenceDataVersion"] = 1

    event = svc.events().insert(calendarId=calendar_id, body=body, **params).execute()
    meet = (
        (event.get("conferenceData") or {})
        .get("entryPoints", [{}])[0]
        .get("uri")
        if make_meet_link
        else None
    )

    attendee_str = ", ".join([a["email"] for a in (attendees or [])]) or "no attendees"
    when = f'{event["start"]["dateTime"]} → {event["end"]["dateTime"]}'
    return f"✅ Scheduled: {title}\n📅 When: {when}\n👥 With: {attendee_str}\n🔗 Meet: {meet or '—'}\n🗂️ Calendar: {calendar_id}\n🆔 Event ID: {event.get('id','')}"

@tool("update_meeting")
def update_meeting(
    event_id: str,
    calendar_id: str = "primary",
    # RFC3339 fields are optional — only send the pieces you want to change
    title: Optional[str] = None,
    start_rfc3339: Optional[str] = None,
    end_rfc3339: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[Attendee]] = None,  # full replacement if provided
    add_meet_link: Optional[bool] = None,        # set True to add / False to remove
    send_updates: str = "all",                   # "all" | "externalOnly" | "none"
) -> str:
    """
    Partially update a Google Calendar event (PATCH). Only provided fields are changed.
    Returns a human-readable confirmation with the updated times and Meet link if present.
    """
    svc = get_gcal_service()

    body = {}
    if title is not None:
        body["summary"] = title
    if description is not None:
        body["description"] = description
    if location is not None:
        body["location"] = location
    if start_rfc3339 is not None:
        body.setdefault("start", {})["dateTime"] = start_rfc3339
    if end_rfc3339 is not None:
        body.setdefault("end", {})["dateTime"] = end_rfc3339
    if attendees is not None:
        body["attendees"] = [{"email": a["email"], "optional": a.get("optional", False)} for a in attendees]

    params = {"calendarId": calendar_id, "eventId": event_id, "sendUpdates": send_updates}

    # Handle Google Meet link toggling
    if add_meet_link is True:
        body["conferenceData"] = {"createRequest": {"requestId": str(uuid4())}}
        params["conferenceDataVersion"] = 1
    elif add_meet_link is False:
        # Remove conference data
        body["conferenceData"] = None
        params["conferenceDataVersion"] = 1

    ev = svc.events().patch(body=body, **params).execute()

    meet_url = None
    conf = ev.get("conferenceData") or {}
    for ep in conf.get("entryPoints", []):
        if ep.get("entryPointType") == "video":
            meet_url = ep.get("uri")
            break

    when = f'{ev["start"].get("dateTime") or ev["start"].get("date")} → {ev["end"].get("dateTime") or ev["end"].get("date")}'
    return f"✏️ Updated event {event_id}\n📅 When: {when}\n📝 Title: {ev.get('summary','')}\n🔗 Meet: {meet_url or '—'}"


@tool("delete_meeting")
def delete_meeting(
    event_id: str,
    calendar_id: str = "primary",
    send_updates: str = "all",    # notify attendees
) -> str:
    """
    Delete an event. Returns a short confirmation. If the event is part of a series,
    this deletes the single instance unless you pass the series master id.
    """
    svc = get_gcal_service()
    svc.events().delete(calendarId=calendar_id, eventId=event_id, sendUpdates=send_updates).execute()
    return f"🗑️ Deleted event {event_id} from {calendar_id} (notifications: {send_updates})."

@tool("find_meetings")
def find_meetings(
    q: Optional[str] = None,
    time_min_rfc3339: Optional[str] = None,
    time_max_rfc3339: Optional[str] = None,
    max_results: int = 10,
    calendar_id: str = "primary",
) -> str:
    """
    List upcoming events, optionally filtered by time window or free-text q.
    Returns a compact table with event_id, start, summary.
    """
    svc = get_gcal_service()
    events = svc.events().list(
        calendarId=calendar_id,
        q=q,
        timeMin=time_min_rfc3339,
        timeMax=time_max_rfc3339,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute().get("items", [])

    if not events:
        return "No events."
    rows = []
    for ev in events:
        start = (ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date") or "")
        rows.append(f"{ev.get('id','')} | {start} | {ev.get('summary','')}")
    return "event_id | start | title\n" + "\n".join(rows)

@tool("parse_datetime")
def parse_datetime(natural_text: str, default_duration_minutes: int = 30, tz: str = "America/New_York") -> dict:
    """
    Parse natural language date/time (e.g., 'next Monday 3pm', 'today 10am') into
    RFC3339 start and end timestamps. Falls back to current year if year missing.
    """
    try:
        now = datetime.now(pytz.timezone(tz))
        dt = date_parser.parse(natural_text, default=now)
        # if year not provided, enforce current year
        if dt.year < now.year:
            dt = dt.replace(year=now.year)

        start = dt.astimezone(pytz.timezone(tz))
        end = start + timedelta(minutes=default_duration_minutes)

        return {
            "start_rfc3339": start.isoformat(),
            "end_rfc3339": end.isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to parse datetime: {str(e)}"}

@tool("download_resume")
def download_resume() -> str:
    """
    Return a direct download link to Krishna's latest resume PDF.
    """
    BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8080")
    url = f"{BASE_URL}/resume/download"
    return (
        f"Here is Krishna’s latest resume:\n\n"
        f"- **PDF**: [Download the resume]({url})\n"
        f"[download_url]={url}"
    )


# tools for the agent
tools = [retriever, memory_search, schedule_meeting, update_meeting, delete_meeting, find_meetings, download_resume]

model = ChatOpenAI(
    model="gpt-4o",              
    temperature=0.3,             
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True
).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
system_prompt = SystemMessage(
    content=f"""
You are Krishna's personal AI assistant — answer **clearly, thoroughly, and professionally** with rich detail and well-structured explanations.

### When the user asks about Krishna:
- Use the `retriever` tool to fetch facts (no fabrication) and memory search tool to query long-term memory for past context.
- **Integrate** retrieved facts with your own synthesis — do **not** copy sentences verbatim; instead **paraphrase and expand** with reasoning, examples, and implications.
- Provide **multi-layered answers**:
  1. **Direct answer** to the query
  2. **Expanded context** (projects, skills, achievements, or relevant experiences)
  3. **Implications or impact** (how it demonstrates expertise, results achieved)
- If retrieval yields nothing, **ask clarifying questions** to narrow the request and explain what details you could provide if available.

### When the topic is unrelated to Krishna:
- Respond with **light humor or curiosity**, then **gracefully redirect** the conversation back to Krishna’s **skills, projects, or experiences** by linking the topic to relevant work.

### Formatting & Style:
- Use **Markdown** formatting.
- Always include **section headings** for clarity (e.g., `🔍 Overview`, `🛠️ Tools & Technologies`, `📊 Results & Impact`).
- Use **bullet points** for lists of skills, tools, projects, and metrics.
- For **work experience**, summarize **chronologically** and **quantify achievements** where possible.
- Keep a **friendly, peer-like tone** while remaining professional.
- When possible, **compare past and current projects**, highlight **technical depth**, and **connect skills across domains**.

### Depth Cues:
When describing Krishna’s skills or projects:
- Include **technical stack** and **specific tools** used
- Explain **challenges faced** and **how they were overcome**
- Mention **metrics** (accuracy, latency, cost savings, throughput improvements)
- Add **real-world applications** or **business/research impact**
- Where relevant, include **links between different domains** (e.g., connecting bioinformatics work to data engineering expertise)

**When asked to schedule a meeting:**
- Call the `schedule_meeting` tool with these arguments:
  - `title`: Short title for the meeting.
  - `start_rfc3339`: Start time in RFC3339 format with timezone (e.g., "2025-08-13T10:00:00-05:00").
  - `end_rfc3339`: End time in RFC3339 format with timezone.
  - `attendees`: List of objects with `email` and optional `optional` boolean (e.g., [{{"email": "alex@company.com"}}]).
  - `description` (optional): Meeting agenda or context.
  - `location` (optional): Physical or virtual location if not using Meet.
  - `calendar_id` (optional): Defaults to "primary".
  - `make_meet_link`: Set to true if a Google Meet link should be generated.
- Use parse_datetime tool to convert natural language date/time (e.g., "tomorrow 3pm CT for 30 minutes") into precise RFC3339 format before calling.
- Confirm details back to the user after scheduling, including date, time, attendees, and meeting link if available.

If the user asks to edit or cancel a meeting, call update_meeting or delete_meeting. Prefer PATCH semantics (only change fields the user mentions). Always include event_id (ask for it or infer from the last created event in this thread).

If the user asks for the resume or CV, call download_resume tool and return the link.
---
**Krishna’s Background:**  
{KRISHNA_BIO}
"""
)

LAST_K = 6  # how many messages to keep in context for the model

def _safe_window(msgs: Sequence[BaseMessage]) -> list[BaseMessage]:
    msgs = list(msgs)

    # Find the last assistant message that requested tools
    last_tool_call_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if getattr(m, "tool_calls", None):
            last_tool_call_idx = i
            break

    if last_tool_call_idx is not None:
        # Include ONLY from that assistant tool-call onward.
        # This guarantees all required ToolMessages are present and in order.
        return msgs[last_tool_call_idx:]

    # No tools in play → keep recent dialog (human/ai only)
    return [m for m in msgs if m.type in ("human", "ai")][-LAST_K:]


def model_call(state: AgentState, config=None) -> AgentState:
    window = _safe_window(state["messages"])
    tid = _thread_id_from_config(config)
    thread_note = SystemMessage(content=f"[thread_id]={tid} (pass this to memory_search.thread_id)")
    msgs = [system_prompt, thread_note, *window]
    ai_msg = model.invoke(msgs)
    return {"messages": [ai_msg]}

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine if the agent should continue."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

def write_memory(state: AgentState, config=None) -> AgentState:
    thread_id = _thread_id_from_config(config)
    # compact last pair
    turns = [m for m in state["messages"] if m.type in ("human","ai")]
    if len(turns) >= 2:
        user_msg = turns[-2].content
        ai_msg   = turns[-1].content
        summary = f"[Q]: {user_msg}\n[A]: {ai_msg}"
        memory_add(summary, thread_id)

    # optional safety autosave
    if memory_write_count >= MEM_AUTOSAVE_EVERY:
        memory_flush()

    # flush ONLY if this is the final turn for the thread
    is_final = (config or {}).get("configurable", {}).get("is_final", False)
    if is_final:
        memory_flush()

    return {}

graph = StateGraph(AgentState)

graph.add_node('agent', model_call)
tools_node = ToolNode(tools=tools)
graph.add_node('tools', tools_node)
graph.add_node("memory_write", write_memory)

graph.add_edge(START, 'agent')
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": "memory_write"})
graph.add_edge("tools", "agent")
graph.add_edge("memory_write", END)

checkpointer = MemorySaver()  # dev-only; for prod use SQLite/Postgres
app = graph.compile(checkpointer=checkpointer)