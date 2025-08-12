import os
import json
import re
import hashlib
import numpy as np
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import TypedDict, Annotated, Sequence, Dict, Optional, List, Literal, Type
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

tools = [ retriever, memory_search]

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
ou are Krishna's personal AI assistant — answer **clearly, thoroughly, and professionally** with rich detail and well-structured explanations.

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