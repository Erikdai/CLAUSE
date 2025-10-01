# -*- coding: utf-8 -*-
# file: logikg_mappo/train.py
# Due to supplementary material size limitations, this is a preliminary version of the code that demonstrates only the core components of our framework.
import os
import argparse
import time
from typing import List, Dict, Any, Optional
import random
import numpy as np
import torch
import unicodedata
import re

from logikg_mappo.agents.enhanced_graph_builder import EnhancedGraphBuilder
from logikg_mappo.agents.traversal import EnhancedTraversalAgent, paths_to_snippets
from logikg_mappo.agents.reranker import build_reranker as make_reranker
from logikg_mappo.agents.reranker import build_answerer

from logikg_mappo.agents.decoder import GeneratorScorer

from logikg_mappo.rl.lc_mappo import (
    MultiHeadCentralCritic,
    create_lc_mappo_policies,
    create_lc_mappo_optimizers,
    lc_mappo_update,
)

try:
    from action_encoding import DEC_COMMON_ID_THRESHOLD, DEC_SHORT_OUTPUT_LEN
except Exception:
    DEC_COMMON_ID_THRESHOLD = 128
    DEC_SHORT_OUTPUT_LEN = 8


# ---------------- helpers ----------------
def split_multi_answers(s: str) -> List[str]:
    """Split a single answer field into multiple answers by common separators."""
    seps = ["||", "///", "|", ";", ",", " and "]
    parts = [s]
    for sp in seps:
        if any(sp in p for p in parts):
            nxt = []
            for p in parts:
                nxt.extend(p.split(sp))
            parts = nxt
    return [p.strip() for p in parts if p.strip()]


class QADataset:
    """Simple TSV loader: each line is 'question\\tanswer(s)'."""

    def __init__(self, path: str, limit: int = None):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 1:
                    q, ans = parts[0], ""
                    answers = []
                else:
                    q, ans = parts[0], parts[1]
                    answers = split_multi_answers(ans)
                self.items.append({"question": q, "answers": answers})
        if isinstance(limit, int) and limit > 0:
            self.items = self.items[:limit]

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ---- robust EM ----
def _normalize_ans(s: str) -> str:
    """Normalize string for robust EM: strip accents, lowercase, keep [a-z0-9] with spaces."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_multi_pred(s: str) -> List[str]:
    """Split a predicted string into multiple candidates by common separators."""
    if not s:
        return []
    parts = re.split(r"\s*(?:\|\|\|?|\s*\|\s*|///|/|,|;|\band\b)\s*", s)
    return [p for p in parts if p]


def em_match(pred: str, golds: List[str]) -> int:
    """Return 1 if the (possibly multi-valued) prediction matches any gold after normalization."""
    gold_norm = {_normalize_ans(g) for g in golds if g}
    gold_norm.discard("")
    if not gold_norm:
        return 0
    parts = _split_multi_pred(pred) or [pred]
    for p in parts:
        if _normalize_ans(p) in gold_norm:
            return 1
    return 0


def em_at_k(pred_list: List[str], golds: List[str], k: int) -> int:
    """EM@K over a ranked list of predictions (deduplicated, truncated to K)."""
    if k <= 0:
        return 0
    seen = set()
    ordered = []
    for p in pred_list:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
        if len(ordered) >= k:
            break
    for p in ordered:
        if em_match(p, golds):
            return 1
    return 0


def token_overlap_f1(pred: str, golds: List[str]) -> float:
    """Simple token-level F1 against the best gold (0..1)."""
    if not pred or not golds:
        return 0.0
    def toks(s: str) -> List[str]:
        return _normalize_ans(s).split()
    pt = toks(pred)
    if not pt:
        return 0.0
    best = 0.0
    for g in golds:
        gt = toks(g)
        if not gt:
            continue
        inter = len(set(pt) & set(gt))
        if inter == 0:
            continue
        prec = inter / max(1, len(pt))
        rec  = inter / max(1, len(gt))
        f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        if f1 > best:
            best = f1
    return float(best)


def extract_candidates_from_action(action: Dict[str, Any], topk: int = 64) -> List[str]:
    """
    Extract candidate entity names from a GB/TR action.
    Priority: node names if present, otherwise aggregate heads/tails from edges.
    """
    cands: List[str] = []
    nodes = action.get("nodes", [])
    if isinstance(nodes, list):
        for x in nodes:
            if isinstance(x, str):
                cands.append(x)
    if not cands:
        edges = action.get("edges", [])
        if isinstance(edges, list):
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 3 and all(isinstance(v, str) for v in e):
                    h, r, t = e
                    # Prefer head entities first (often the answer), then tails
                    cands.append(h)
                    cands.append(t)
    seen, out = set(), []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= topk:
            break
    return out


def _node_label_from_id(gb: EnhancedGraphBuilder, nid: str) -> str:
    """Resolve a human-readable node label from an internal node id."""
    idx = gb._node_index_map.get(nid)
    texts = getattr(gb.node_pack, "texts", {})
    if isinstance(texts, dict):
        return texts.get(nid, texts.get(str(nid), str(nid))) or str(nid)
    if isinstance(texts, list) and isinstance(idx, int) and 0 <= idx < len(texts):
        return texts[idx] or str(nid)
    return str(nid)


def extract_fallback_candidates(gb: EnhancedGraphBuilder, question: str, topk: int = 32) -> List[str]:
    """Dense-retrieval fallback: map candidate ids to labels and deduplicate."""
    try:
        ids = gb._get_candidate_entities(question, max_candidates=topk)
    except Exception:
        ids = []
    out, seen = [], set()
    for nid in ids[:topk]:
        name = _node_label_from_id(gb, nid)
        if name and name not in seen:
            seen.add(name); out.append(name)
    return out


def now_ms() -> float:
    """Monotonic wall time in milliseconds."""
    return time.perf_counter() * 1000.0


def to_tensor1(x, device):
    """Convert a list/array to a float32 tensor shaped (1, -1) on device."""
    return torch.as_tensor(x, dtype=torch.float32, device=device).reshape(1, -1)


def build_obs(gb_summary, tr_summary, rr_summary, device) -> torch.Tensor:
    """Pack GB/TR/RR summaries into a single observation tensor of shape (1, 3, 3)."""
    gb_obs = [float(gb_summary["num_nodes"]), float(gb_summary["num_edges"]), float(gb_summary["steps"])]
    tr_obs = [float(tr_summary["path_len"]), float(tr_summary["candidate_count"]), float(tr_summary["stop_flag"])]
    rr_obs = [float(rr_summary["k"]), float(rr_summary["token_mass"]), float(rr_summary.get("entropy", 0.0))]
    obs = torch.tensor([gb_obs, tr_obs, rr_obs], dtype=torch.float32, device=device).unsqueeze(0)
    return obs


def build_actions(gb_act, tr_act, rr_info, device) -> torch.Tensor:
    """Pack last GB/TR/RR actions into a small action-feature tensor of shape (1, 3, 3)."""
    op = gb_act.get("operation_type") or gb_act.get("op") or "stop"
    op_map = {"add_edge": 0, "delete_edge": 1, "stop": 2}
    gb_idx = float(op_map.get(op, 2))
    gb_vec = [gb_idx, float(gb_idx == 1.0), float(gb_idx == 2.0)]
    stop_flag = 1.0 if tr_act.get("stop", True) else 0.0
    path_len = float(tr_act.get("path_len", len(tr_act.get("path", []))))
    cand_cnt = float(tr_act.get("candidate_count", 0))
    tr_vec = [stop_flag, min(1.0, path_len / 16.0), min(1.0, cand_cnt / 64.0)]
    rr_vec = [
        float(rr_info.get("k_norm", 0.0)),
        float(rr_info.get("token_norm", 0.0)),
        float(rr_info.get("stop_flag", 1.0)),
    ]
    acts = torch.tensor([gb_vec, tr_vec, rr_vec], dtype=torch.float32, device=device).unsqueeze(0)
    return acts


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- New: pool building for LLM-oriented answering ----------
def _entity_type_hint_from_question(q: str):
    """Heuristic entity-type hint derived from question prefix."""
    qs = (q or "").lower().strip()
    if qs.startswith(("who", "which actor", "actor", "whom")):
        return {"Person"}
    if qs.startswith(("where", "which location", "what country", "what city")):
        return {"Location", "Place"}
    if qs.startswith(("when", "what year", "which year", "what date")):
        return {"Year", "Date", "Time"}
    return None


def _short_fact(gb: EnhancedGraphBuilder, nid: str) -> str:
    """
    One-line 'evidence' text for LLM display.
    Prefer an already-explored subgraph edge involving nid; otherwise try a short description.
    """
    try:
        for (h, r, t) in list(getattr(gb, "current_edges", []))[:256]:
            if nid in (h, t):
                rn = getattr(gb, "relation_label", lambda x: x)(r) if hasattr(gb, "relation_label") else r
                other = t if nid == h else h
                other_name = _node_label_from_id(gb, other)
                if other_name:
                    return f"{rn}: {other_name}"
    except Exception:
        pass
    try:
        txt = getattr(gb.node_pack, "desc", {}).get(nid, "")
        if txt:
            return txt.split(".")[0][:80]
    except Exception:
        pass
    return ""


def collect_pool_candidates(gb: EnhancedGraphBuilder, q: str, last_gb_action: dict,
                            max_pool: int = 128, dense_fallback: int = 64):
    """
    Build a candidate pool for LLM selection:
      pool = nodes seen in subgraph + 1-hop neighbors (optionally filtered by relation whitelist)
             + dense retrieval fallback + type hint filter + dedupe + evidence lines.

    Returns:
      pool_dicts: [{'id','name','text','approx_tokens','prior'}, ...]
      name_list:  list of canonical names only (for EM)
    """
    # 1) nodes already seen in the subgraph
    cand_ids = []
    edges = last_gb_action.get("edges", []) or []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 3:
            h, r, t = e
            cand_ids.extend([h, t])
    cand_ids = [x for x in cand_ids if isinstance(x, str)]

    # 2) expand one-hop neighbors
    rel_whitelist = gb._route_relations(q) if hasattr(gb, "_route_relations") else None
    neigh = []
    try:
        for nid in list(dict.fromkeys(cand_ids))[:64]:
            for (h, r, t) in gb.get_neighbors(nid):
                if (rel_whitelist is None) or (r in rel_whitelist):
                    neigh.append(h); neigh.append(t)
    except Exception:
        pass
    cand_ids.extend(neigh)
    cand_ids = list(dict.fromkeys(cand_ids))  # dedupe

    # 3) dense fallback (if too few)
    if len(cand_ids) < max(32, max_pool // 2):
        try:
            import re as _re
            q_clean = _re.sub(r"\[(.+?)\]", " ", q or "").strip()
            q_vec = gb.embedder.encode([q_clean])[0].astype("float32")
            all_ids = list(getattr(gb, "all_node_ids", lambda: [])() or [])
            labels = [_node_label_from_id(gb, nid) for nid in all_ids]
            mask = [bool(x) for x in labels]
            ids_used = [nid for nid, m in zip(all_ids, mask) if m]
            texts = [lab for lab, m in zip(labels, mask) if m]
            if texts:
                doc_vecs = gb.embedder.encode(texts).astype("float32")
                qn = q_vec / (np.linalg.norm(q_vec) + 1e-8)
                dn = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
                scores = (dn @ qn).reshape(-1)
                top_idx = np.argsort(-scores)[:dense_fallback]
                fallback_ids = [ids_used[i] for i in top_idx]
                cand_ids.extend(fallback_ids)
        except Exception:
            pass

    # 4) optional type filtering (question → entity type)
    tset = _entity_type_hint_from_question(q)
    if tset:
        try:
            cand_ids = [nid for nid in cand_ids if getattr(gb, "node_type", lambda x: None)(nid) in tset]
        except Exception:
            pass

    # 5) canonicalize + add evidence + compute priors
    seen = set()
    pool_dicts, name_list = [], []
    try:
        import re as _re
        q_clean = _re.sub(r"\[(.+?)\]", " ", q or "").strip()
        qv = gb.embedder.encode([q_clean])[0].astype("float32")
        qn = qv / (np.linalg.norm(qv) + 1e-8)
    except Exception:
        qn = None

    for nid in cand_ids:
        name = _node_label_from_id(gb, nid)
        if not name:
            continue
        key = name.strip().lower()
        if key in seen:
            continue
        seen.add(key)

        proof = _short_fact(gb, nid)
        display = name if not proof else f"{name} — {proof}"

        prior = 0.0
        if qn is not None:
            try:
                nv = gb.embedder.encode([name])[0].astype("float32")
                nn = nv / (np.linalg.norm(nv) + 1e-8)
                prior = float(nn @ qn)
            except Exception:
                prior = 0.0

        approx_tok = max(1, int(len(display.split()) / 0.75))
        pool_dicts.append({
            "id": nid,
            "name": name,          # canonical name (for EM)
            "text": display,       # one-line display for LLM (with evidence)
            "approx_tokens": approx_tok,
            "prior": float(prior),
        })
        name_list.append(name)
        if len(pool_dicts) >= max_pool:
            break

    return pool_dicts, name_list


# ---------------- schedules ----------------
def linear_warmup(now_step: int, start_v: float, end_v: float, total_steps: int) -> float:
    """Linear warmup from start_v to end_v over total_steps."""
    if total_steps <= 0:
        return end_v
    alpha = min(1.0, max(0.0, now_step / float(total_steps)))
    return (1.0 - alpha) * start_v + alpha * end_v


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--kb_cache_dir", required=True)
    ap.add_argument("--embedder", default="thenlper/gte-small")

    # Reranker / Generator
    ap.add_argument("--rerank_max_steps", type=int, default=15)
    ap.add_argument("--rerank_stop_penalty", type=float, default=-0.2, help="negative bias to encourage selecting before stopping")
    ap.add_argument("--rerank_min_k", type=int, default=2, help="at least 2 picks in dynamic reranker")
    ap.add_argument("--token_budget", type=float, default=512.0, help="budget (tokens) for reranked context")
    ap.add_argument("--rerank_use_lm", action="store_true", help="augment DynamicReranker with LM scoring via LogicAwareDecoder")
    ap.add_argument("--rerank_lm_name", type=str, default="gpt2", help="LM name for dynamic reranker (passed correctly)")
    ap.add_argument("--rerank_w_lm", type=float, default=0.5)
    ap.add_argument("--use_llm_reranker", action="store_true", help="use local GPT-based reranker agent instead of DynamicReranker")
    ap.add_argument("--llm_reranker_model", type=str, default="gpt2")
    # New: API reranking (different interface than answer generation)
    ap.add_argument("--use_llm_reranker_api", action="store_true",
                    help="use external API LLM to rank the whole candidate list at once")
    ap.add_argument("--api_reranker_model", type=str, default=None,
                    help="API reranker model name (e.g., 'Qwen/Qwen3-Reranker-4B'); if None, falls back to --llm_reranker_model")

    # External LLM for final answer (Groq)
    ap.add_argument("--llm_answer", action="store_true", help="Use Groq LLM to generate final answer from reranked contexts")
    ap.add_argument("--llm_answer_model", type=str, default="llama-3.1-8b-instant", help="Groq model for final answering")
    ap.add_argument("--llm_ctx_k", type=int, default=6, help="Top-K snippets (or fallback ctx) passed to the LLM answerer")

    # ---- New: pool & direct answering ----
    ap.add_argument("--pool_max", type=int, default=128, help="upper bound for collected candidates from subgraph+neighbors+fallback")
    ap.add_argument("--pool_dense_fallback", type=int, default=64, help="dense fallback count to add if pool is small")
    ap.add_argument("--direct_answer_from_pool", action="store_true",
                    help="skip reranker, pass pool candidates directly to an LLM for single-choice")

    # legacy / RL hyper
    ap.add_argument("--lambda_token", type=float, default=0.0)
    ap.add_argument("--budget_token", type=float, default=1.0)
    ap.add_argument("--gen_disable_lm", action="store_true")
    ap.add_argument("--gen_lm_name", default="gpt2")

    ap.add_argument("--disable_ner", action="store_true")

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--tr_max_hops", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs/lcmappo_txt_cache_only")
    ap.add_argument("--acc_reward", type=float, default=1.0)

    # latency budgets & lambdas (with curriculum)
    ap.add_argument("--latency_budget_ms", type=float, default=200.0)
    ap.add_argument("--lambda_edge", type=float, default=0.0)
    ap.add_argument("--lambda_latency", type=float, default=0.0)
    ap.add_argument("--lambda_logic", type=float, default=0.0)

    ap.add_argument("--lambda_latency_start", type=float, default=None)
    ap.add_argument("--lambda_latency_end", type=float, default=None)
    ap.add_argument("--lambda_warmup_steps", type=int, default=0)

    ap.add_argument("--latency_budget_start_ms", type=float, default=None)
    ap.add_argument("--latency_budget_end_ms", type=float, default=None)
    ap.add_argument("--latency_budget_warmup_steps", type=int, default=0)

    # Delayed cost warmup for selection/logic (reduce early 'greedy stop')
    ap.add_argument("--sel_cost_warmup_steps", type=int, default=8000)

    # dual & PPO
    ap.add_argument("--budget_edge", type=float, default=0.25)
    ap.add_argument("--budget_latency", type=float, default=1.0)
    ap.add_argument("--budget_logic", type=float, default=0.20)
    ap.add_argument("--dual_lr", type=float, default=5e-2)
    ap.add_argument("--ppo_eps0", type=float, default=0.4)
    ap.add_argument("--ppo_kappa", type=float, default=1.0)
    ap.add_argument("--ppo_eps_min", type=float, default=0.02)
    ap.add_argument("--kl_coef", type=float, default=0.0)
    ap.add_argument("--kl_target", type=float, default=0.01)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--disable_highprior_fallback", action="store_true", help="Disable highest-prior fallback when reranker picks none")
    # Reward shaping
    ap.add_argument("--alpha_em", type=float, default=1.0, help="global multiplier for EM in task reward")
    ap.add_argument("--beta_sim", type=float, default=0.3, help="weight for token-overlap similarity in task reward")
    ap.add_argument("--delta_lm", type=float, default=0.0, help="weight for LM-eval score in task reward (if available)")
    # Cost shaping options
    ap.add_argument("--k_soft_cap", type=int, default=0, help="soft cap for K (0 disables)")
    ap.add_argument("--k_cost_coef", type=float, default=0.0, help="cost coefficient when K exceeds the soft cap")
    ap.add_argument("--seed", type=int, default=42)

    # EM@K & logging
    ap.add_argument("--em_topk", type=int, default=3, help="report EM@K besides EM@1")
    ap.add_argument("--limit", type=int, default=5, help="only run first N samples for debug (0 = full)")
    ap.add_argument("--observe_only", action="store_true", help="skip updates; just print metrics")
    ap.add_argument("--verbose_trace", action="store_true", help="print per-sample verbose traces")
    ap.add_argument("--single_episode", action="store_true", help="run a single episode and print K/tokens/reward")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    ds = QADataset(args.train_file, limit=args.limit if args.limit and args.limit > 0 else None)
    print(f"[data] loaded {len(ds)} samples")

    gb = EnhancedGraphBuilder(
        ttl_file=None,
        cache_dir=args.kb_cache_dir,
        embedder_name=args.embedder,
        device=args.device,
        use_cache_only=True,
        enable_ner=not args.disable_ner,
    )
    tr = EnhancedTraversalAgent(max_hops=args.tr_max_hops, device=args.device)

    # Unified construction: API LLM reranker > local LLM reranker agent > DynamicReranker
    rerank = make_reranker(
        use_llm_reranker_api=bool(args.use_llm_reranker_api),
        llm_reranker_model=(args.api_reranker_model or args.llm_reranker_model),
        use_llm_reranker=bool(args.use_llm_reranker),
        embedder=gb.embedder,
        token_budget=args.token_budget,
        max_steps=args.rerank_max_steps,
        stop_penalty=args.rerank_stop_penalty,
        device=args.device,
        use_lm_in_dynamic=bool(args.rerank_use_lm),
        rerank_lm_name=str(args.rerank_lm_name),
        w_lm=float(args.rerank_w_lm),
        min_k=int(args.rerank_min_k),
    )

    # Print reranker path in use
    if args.use_llm_reranker_api:
        print(f"[rerank] Using API reranker: model={args.api_reranker_model or args.llm_reranker_model}")
    elif args.use_llm_reranker:
        print(f"[rerank] Using local LLM reranker agent (model={args.llm_reranker_model}); fallback to Dynamic if unavailable")
    else:
        print(f"[rerank] Using DynamicReranker (embedder + tiny-LM={'on' if args.rerank_use_lm else 'off'})")

    # If final LLM answering is enabled, initialize once here (reads GROQ_API_KEY internally)
    answerer = None
    if args.llm_answer:
        try:
            answerer = build_answerer(provider="groq", model=args.llm_answer_model)
            print(f"[llm] Groq answerer enabled: model={args.llm_answer_model}")
        except Exception as e:
            print(f"[llm] Failed to init Groq answerer: {e}. Will fallback to non-LLM answer.")

    generator = GeneratorScorer()

    # critic/policies
    critic = MultiHeadCentralCritic(state_dim=11, obs_dim=3, act_dim=3, dll_dim=0, gnn_dim=0, hidden=256, n_agents=3).to(device)
    policies = create_lc_mappo_policies({"gb": gb, "tr": tr, "rerank": rerank}, device)
    critic_opt, policy_opts = create_lc_mappo_optimizers(critic, policies, lr=args.lr)

    # --- key normalization helper ---
    def _normalize_keys(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "latency" in d and "lat" not in d:
            d["lat"] = d.pop("latency")
        if "edges" in d and "edge" not in d:
            d["edge"] = d.pop("edges")
        if "token" in d and "logic" not in d:
            d["logic"] = d.pop("token")
        return d

    # budgets / lambdas (initial) + normalization
    budgets = _normalize_keys({
        "edge": torch.tensor(args.budget_edge, device=device),
        "lat": torch.tensor(args.budget_latency, device=device),
        "logic": torch.tensor(getattr(args, "budget_token", getattr(args, "budget_logic", 1.0)), device=device),
    })
    lambdas = _normalize_keys({
        "edge": torch.tensor(args.lambda_edge, device=device),
        "lat": torch.tensor(args.lambda_latency, device=device),
        "logic": torch.tensor(getattr(args, "lambda_token", getattr(args, "lambda_logic", 0.0)), device=device),
    })

    # curriculum configs
    lambda_lat_start = args.lambda_latency_start
    lambda_lat_end = args.lambda_latency_end
    lambda_warm_steps = max(0, int(args.lambda_warmup_steps))

    lat_budget_start = args.latency_budget_start_ms
    lat_budget_end = args.latency_budget_end_ms
    lat_budget_warm_steps = max(0, int(args.latency_budget_warmup_steps))

    # Warmup for selection/logic costs (0→1), mitigating early 'stop' bias
    sel_cost_warm_steps = max(0, int(args.sel_cost_warmup_steps))

    if lambda_lat_start is None and lambda_lat_end is None:
        lambda_lat_start = float(lambdas["lat"].item()); lambda_lat_end = float(lambdas["lat"].item())
    elif lambda_lat_start is None:
        lambda_lat_start = float(lambdas["lat"].item())
    elif lambda_lat_end is None:
        lambda_lat_end = float(lambdas["lat"].item())

    if lat_budget_start is None and lat_budget_end is None:
        lat_budget_start = float(budgets["lat"].item()) * args.latency_budget_ms
        lat_budget_end = float(budgets["lat"].item()) * args.latency_budget_ms
    elif lat_budget_start is None:
        lat_budget_start = args.latency_budget_ms
    elif lat_budget_end is None:
        lat_budget_end = args.latency_budget_ms

    os.makedirs(args.out_dir, exist_ok=True)

    # running stats
    seen = 0
    em1_sum = 0.0
    emk_sum = 0.0
    reward_sum = 0.0
    lat_sum = 0.0
    edges_sum = 0.0
    edgecost_sum = 0.0
    loss_sum = 0.0
    kl_sum = 0.0
    K = max(1, int(args.em_topk))

    def maybe_print(step_end=False):
        """Periodic progress printer."""
        if seen == 0:
            return
        if (seen % max(1, args.log_every) == 0) or step_end:
            avg_em1 = em1_sum / seen
            avg_emk = emk_sum / seen
            avg_r = reward_sum / seen
            avg_lat = lat_sum / seen
            avg_edges = edges_sum / seen
            avg_edgecost = edgecost_sum / seen
            avg_loss = (loss_sum / max(1, seen)) if not args.observe_only else float("nan")
            avg_kl = (kl_sum / max(1, seen)) if not args.observe_only else float("nan")
            print(f"[progress] #{seen}  EM1={avg_em1:.3f}  EM@{K}={avg_emk:.3f}  "
                  f"R={avg_r:.3f}  lat={avg_lat:.1f}ms  edges={avg_edges:.2f}  "
                  f"edge_cost={avg_edgecost:.3f}  loss={avg_loss:.4f}  KL={avg_kl:.4f}  "
                  f"λ=({float(lambdas['edge']):.3f},{float(lambdas['lat']):.3f},{float(lambdas.get('logic', 0.0)):.3f})")

    for ep in range(1, max(1, args.epochs) + 1):
        # Diagnostics per-epoch
        pool_recall_hits = 0
        fallback_uses = 0
        total_pools = 0
        for ex_id, ex in enumerate(ds.items, 1):
            q = ex["question"]
            golds = ex.get("answers", [])

            # curriculum update (per step)
            if lambda_warm_steps > 0:
                new_lat_lambda = linear_warmup(seen, lambda_lat_start, lambda_lat_end, lambda_warm_steps)
                lambdas["lat"] = torch.tensor(new_lat_lambda, device=device)
            if lat_budget_warm_steps > 0:
                new_budget_ms = linear_warmup(seen, lat_budget_start, lat_budget_end, lat_budget_warm_steps)
                latency_budget_ms_effective = new_budget_ms
            else:
                latency_budget_ms_effective = args.latency_budget_ms

            # reset agents
            if hasattr(gb, "reset_subgraph"):
                gb.reset_subgraph()
            tr._maybe_reset_path({"reset": True})

            if args.verbose_trace:
                try:
                    ner_spans = gb._extract_ner_spans(q)
                except Exception:
                    ner_spans = []
                try:
                    seeds_preview_ids = gb._get_candidate_entities(q, max_candidates=16)
                    seeds_preview = [_node_label_from_id(gb, nid) for nid in seeds_preview_ids[:16]]
                except Exception:
                    seeds_preview = []
            else:
                ner_spans = []
                seeds_preview = []

            gb_steps, k_selected = 0, 0
            last_gb_action, last_tr_action = {}, {}
            ms_gb, ms_tr, ms_rr = 0.0, 0.0, 0.0

            for step in range(1, args.max_steps + 1):
                # --- GB step ---
                t0 = now_ms()
                gb_out = gb({"question": q})
                ms_gb += (now_ms() - t0)

                last_gb_action = gb_out.get("action", {}) or {}
                gb_steps += 1

                # Greedy fallback: if no edge, force-add one best candidate edge
                if (last_gb_action.get("operation_type") in (None, "stop")) and not last_gb_action.get("edges"):
                    seed_ids = gb._get_candidate_entities(q, max_candidates=8)
                    rel_ids = gb._get_candidate_relations(q, max_candidates=32)
                    qv = gb.embedder.encode([q])[0].astype("float32")
                    best = None
                    for s in seed_ids[:3]:
                        best = gb._best_new_edge_from(qv, s, set(rel_ids) if rel_ids else None) or gb._best_new_edge_from(qv, s, None)
                        if best: break
                    if best:
                        _, h, r, t = best
                        gb._add_edge(h, r, t)
                        last_gb_action = {
                            "operation_type": "add_edge",
                            "subject": h, "predicate": r, "object": t,
                            "nodes": list(gb.current_nodes),
                            "edges": list(gb.current_edges),
                            "num_nodes": len(gb.current_nodes),
                            "num_edges": len(gb.current_edges),
                        }

                # --- TR step ---
                tr_obs = {
                    "question": q,
                    "neighbors": last_gb_action.get("edges", []),
                    "path": last_gb_action.get("edges", [])[:2],
                    "t": step,
                }
                t0 = now_ms()
                tr_out = tr(tr_obs)
                ms_tr += (now_ms() - t0)
                last_tr_action = tr_out.get("action", {}) or {}

                # Early stop
                if (last_gb_action.get("operation_type") in (None, "stop")) and last_tr_action.get("stop", False):
                    if last_gb_action.get("edges"):
                        break
                    if step >= 2:
                        break

            # ==== Stronger strategy to build a pool for the LLM ====
            pool_dicts, pool_names = collect_pool_candidates(
                gb, q, last_gb_action,
                max_pool=int(args.pool_max),
                dense_fallback=int(args.pool_dense_fallback),
            )

            # rerank / direct answer
            pred = ""
            topk_list: List[str] = []
            lp_rr = torch.tensor(0.0, device=device)
            rr_entropy = torch.tensor(0.0, device=device)
            k_selected = 0
            approx_tokens = 0.0
            rr_steps: List[Dict[str, Any]] = []
            prior_mean = 0.0

            if pool_dicts:
                try:
                    prior_mean = float(sum(d["prior"] for d in pool_dicts) / max(1, len(pool_dicts)))
                except Exception:
                    prior_mean = 0.0

                if args.direct_answer_from_pool and args.llm_answer and answerer is not None:
                    # === Direct answering: give pool display lines to the LLM and pick one ===
                    ctx_list = [d["text"] for d in pool_dicts][:max(1, int(args.llm_ctx_k))]
                    try:
                        pred_llm = (answerer.answer(q, ctx_list) or "").strip()
                    except Exception as e:
                        print(f"[llm] direct-from-pool failed: {e}")
                        pred_llm = ""

                    # Map back to canonical name
                    def _map_to_name(s: str) -> str:
                        s_low = s.strip().lower()
                        for d in pool_dicts:
                            nm = d["name"].strip().lower()
                            if s_low == nm or s_low.startswith(nm):
                                return d["name"]
                        return s

                    pred = _map_to_name(pred_llm) if pred_llm else ""
                    topk_list = [pred] + [d["name"] for d in pool_dicts[:max(1, int(args.em_topk) - 1)]]
                    k_selected = min(len(ctx_list), int(args.llm_ctx_k))
                    approx_tokens = float(sum(max(1, int(len(x.split()) / 0.75)) for x in ctx_list))

                else:
                    # === Keep the original reranker path ===
                    rerank.start_episode(q, pool_dicts)
                    t0 = now_ms()
                    for _ in range(args.rerank_max_steps):
                        step_out = rerank.forward({})
                        rr_steps.append(step_out)
                        if step_out["action"]["type"] == "stop":
                            break
                    ms_rr += (now_ms() - t0)

                    picked = rr_steps[-1]["action"].get("picked", []) if rr_steps else []
                    selected_outputs = []
                    for i in picked:
                        try:
                            item = pool_dicts[i]
                            selected_outputs.append(item.get("name") or item.get("text", ""))
                        except Exception:
                            pass

                    if args.llm_answer and answerer is not None and selected_outputs:
                        try:
                            pred_llm = answerer.answer(q, selected_outputs[:max(1, int(args.llm_ctx_k))]).strip()
                            # Prevent LLM from returning evidence text; map to canonical name
                            def _map_to_name2(s: str) -> str:
                                s_low = s.strip().lower()
                                for d in pool_dicts:
                                    nm = d["name"].strip().lower()
                                    if s_low == nm or s_low.startswith(nm):
                                        return d["name"]
                                return s
                            pred = _map_to_name2(pred_llm) or (selected_outputs[0] if selected_outputs else "")
                        except Exception as e:
                            print(f"[llm] Answerer failed: {e}. Fallback to top-1 candidate.")
                            pred = selected_outputs[0] if selected_outputs else ""
                    else:
                        pred = selected_outputs[0] if selected_outputs else ""

                    topk_list = selected_outputs[:]

                    try:
                        lp_rr = torch.stack([s["log_prob"].reshape(()) for s in rr_steps], dim=0).sum()
                        rr_entropy = torch.stack([s["entropy"].reshape(()) for s in rr_steps], dim=0).mean()
                    except Exception:
                        lp_rr = torch.tensor(0.0, device=device)
                        rr_entropy = torch.tensor(0.0, device=device)

                    k_selected = len(picked)
                    approx_tokens = float(sum(pool_dicts[i]["approx_tokens"] for i in picked))

            else:
                # === No candidates: let the LLM answer with empty or light context ===
                ctx = []
                if args.llm_answer and answerer is not None:
                    try:
                        pred = (answerer.answer(q, ctx) or "").strip()
                        topk_list = ctx[:]
                        k_selected = len(ctx)
                        approx_tokens = float(sum(max(1, int(len(x.split()) / 0.75)) for x in ctx)) if ctx else 0.0
                    except Exception as e:
                        print(f"[llm] direct-answer failed (no candidates): {e}")
                        pred = ""
                else:
                    pred = ""

            # --- pool recall@pool diagnostic ---
            try:
                total_pools += 1
                gold_norm = {_normalize_ans(g) for g in golds if g}
                gold_norm.discard("")
                hit = 0
                for it in (pool_dicts or []):
                    cand_lbl = it.get("name") or it.get("text", "")
                    if _normalize_ans(cand_lbl) in gold_norm:
                        hit = 1; break
                pool_recall_hits += hit
            except Exception:
                pass

            r_em1 = em_match(pred, golds)
            r_emk = em_at_k(topk_list, golds, K)
            sim_score = token_overlap_f1(pred, golds) if pred else 0.0

            # ---- Task reward (shaped) ----
            # Emphasize EM: 3*EM@1 + 1*EM@K + a small similarity term to reduce reward sparsity
            w_em1 = 3.0 * float(args.alpha_em)
            w_emk = 1.0 * float(args.alpha_em)
            scalar_r = (w_em1 * float(r_em1)) + (w_emk * float(r_emk)) + (float(args.beta_sim) * float(sim_score))

            # ---- Costs: normalized edge count + latency ----
            num_edges = len(last_gb_action.get("edges", [])) if isinstance(last_gb_action.get("edges", []), list) else 0
            latency_ms = 0.0
            try:
                latency_ms = float((ms_gb + ms_tr + ms_rr))
            except Exception:
                latency_ms = 0.0
            edge_cost = min(1.0, num_edges / 8.0)
            budget_ms = latency_budget_ms_effective
            latency_cost = min(1.0, latency_ms / max(1e-6, budget_ms))

            # ====== Train or observe-only ======
            if not args.observe_only:
                gb_summary = {"num_nodes": len(last_gb_action.get("nodes", [])) if isinstance(last_gb_action.get("nodes", []), list) else 0,
                              "num_edges": num_edges, "steps": gb_steps}
                tr_summary = {"path_len": int(last_tr_action.get("path_len", len(last_tr_action.get("path", [])))) if isinstance(last_tr_action.get("path", []), list) else 0,
                              "candidate_count": int(num_edges), "stop_flag": 1 if last_tr_action.get("stop", False) else 0}
                rr_summary = {"k": k_selected, "token_mass": approx_tokens, "entropy": float(rr_entropy.reshape(()).item() if hasattr(rr_entropy, "reshape") else float(rr_entropy))}
                obs = build_obs(gb_summary, tr_summary, rr_summary, device)

                acts = build_actions(
                    last_gb_action,
                    last_tr_action,
                    {
                        "k_norm": min(1.0, float(k_selected) / max(1.0, float(args.rerank_max_steps))),
                        "token_norm": min(1.0, float(approx_tokens) / max(1.0, float(args.token_budget))),
                        "stop_flag": 1.0,
                    },
                    device,
                )
                state_vec = [
                    float(gb_summary["num_nodes"]), float(gb_summary["num_edges"]), float(gb_summary["steps"]),
                    float(tr_summary["path_len"]), float(tr_summary["candidate_count"]), float(tr_summary["stop_flag"]),
                    float(k_selected), float(approx_tokens), float(rr_summary["entropy"]),
                    float(prior_mean), float(scalar_r)
                ]
                state = to_tensor1(state_vec, device)

                dll = torch.zeros(1, 0, device=device)
                gnn = torch.zeros(1, 0, device=device)

                lp_gb = (gb_out.get("log_prob", torch.tensor(0.0, device=device))).reshape(())
                lp_tr = (tr_out.get("log_prob", torch.tensor(0.0, device=device))).reshape(())
                lp_dec = lp_rr.reshape(())
                logp_old = torch.stack([lp_gb, lp_tr, lp_dec], dim=0).unsqueeze(0).detach()

                rewards = torch.tensor([[scalar_r*args.acc_reward]*3], device=device, dtype=torch.float32)

                # Logic cost (K / token mass) with warmup coefficient: reduce early 'select less' pressure
                sel_coef = linear_warmup(seen, 0.0, 1.0, sel_cost_warm_steps) if sel_cost_warm_steps > 0 else 1.0
                logic_base = min(1.0, float(approx_tokens) / max(1.0, float(args.token_budget)))
                logic_cost = torch.tensor([[logic_base * sel_coef]*3], device=device)

                costs = {
                    "edge": torch.tensor([[edge_cost]*3], device=device),
                    "lat": torch.tensor([[latency_cost]*3], device=device),
                    "logic": logic_cost,
                }
                try:
                    # Optional extra cost if K exceeds a soft cap
                    if int(args.k_soft_cap) > 0 and float(args.k_cost_coef) > 0 and k_selected > int(args.k_soft_cap):
                        over = float(k_selected - int(args.k_soft_cap))
                        extra = min(1.0, float(args.k_cost_coef) * over)
                        costs["logic"][0, 2] = torch.clamp(costs["logic"][0, 2] + extra, 0.0, 1.0)
                except Exception:
                    pass

                def _align_keys(costs, budgets, lambdas):
                    """Ensure costs/budgets/lambdas share the same keyset and aliases are normalized."""
                    def _norm(d):
                        if "latency" in d and "lat" not in d:
                            d["lat"] = d.pop("latency")
                        if "edges" in d and "edge" not in d:
                            d["edge"] = d.pop("edges")
                        if "token" in d and "logic" not in d:
                            d["logic"] = d.pop("token")
                        return d
                    costs = _norm(costs); budgets = _norm(budgets); lambdas = _norm(lambdas)
                    expected = set(costs.keys())
                    for k in expected:
                        if k not in budgets:
                            budgets[k] = torch.tensor(1.0, device=device)
                        if k not in lambdas:
                            lambdas[k] = torch.tensor(0.0, device=device)
                    for d in (budgets, lambdas):
                        for kk in list(d.keys()):
                            if kk not in expected:
                                d.pop(kk, None)
                    assert expected.issubset(budgets.keys()), f"budgets missing keys: {expected - set(budgets.keys())}"
                    assert expected.issubset(lambdas.keys()), f"lambdas missing keys: {expected - set(lambdas.keys())}"
                    return costs, budgets, lambdas

                costs, budgets, lambdas = _align_keys(costs, budgets, lambdas)

                ctx = {
                    "gb_obs": {"question": q},
                    "tr_obs": {"question": q, "path": last_tr_action.get("path", [])},
                    "dec_obs": {
                        "question": q,
                        "context_pool": pool_dicts if 'pool_dicts' in locals() else [],
                        "token_budget": float(args.token_budget),
                    },
                    "gb_action": last_gb_action,
                    "tr_action": last_tr_action,
                    "dec_action": {"ordered_indices": list(rr_steps[-1]["action"].get("picked", []) if rr_steps else [])},
                    "dec_rule_masks": [],
                    "dec_prev_tokens": list(rr_steps[-1]["action"].get("picked", []) if rr_steps else []),
                    "dec_vocab_size": max(2, int(args.rerank_max_steps) + 1),
                }

                stats = lc_mappo_update(
                    batch=(state, obs, dll, gnn, acts, logp_old, rewards, costs, ctx),
                    critic=critic, policies=policies, lambdas=lambdas, budgets=budgets,
                    eps0=args.ppo_eps0, kappa=args.ppo_kappa, dual_lr=args.dual_lr,
                    critic_optimizer=critic_opt, policy_optimizers=policy_opts,
                    max_grad_norm=args.max_grad_norm, agents={"gb": gb, "tr": tr, "rerank": rerank},
                    eps_min=args.ppo_eps_min, kl_coef=args.kl_coef, kl_target=args.kl_target, ent_coef=args.ent_coef,
                    actor_update_interval=1, critic_update_interval=1, dual_update_interval=10, global_step=seen
                )
                loss_val = stats.get("total_loss", 0.0)
                if torch.is_tensor(loss_val):
                    loss_val = loss_val.detach()
                loss_sum += float(loss_val)
                kl_sum += float(stats.get("approx_kl", 0.0))

            # Update running aggregates
            seen += 1
            em1_sum += float(r_em1)
            emk_sum += float(r_emk)
            reward_sum += float(scalar_r * args.acc_reward)
            lat_sum += float(latency_ms)
            edges_sum += float(num_edges)
            edgecost_sum += float(edge_cost)

            if args.verbose_trace:
                print(f"[trace] #{seen} Q={q!r} pred={pred!r} em={r_em1} em@{K}={r_emk} "
                      f"simF1={sim_score:.2f} r={scalar_r:.2f} "
                      f"lat={latency_ms:.1f}ms edges={num_edges} edge_cost={edge_cost:.2f} "
                      f"k={k_selected} tokens={approx_tokens:.0f}")

            if args.single_episode:
                print(f"single-episode: K={k_selected} tokens={approx_tokens:.0f} reward={scalar_r:.3f}")
                return
            maybe_print(step_end=False)

        # Save checkpoint + epoch summary
        if not args.observe_only:
            torch.save({"critic": critic.state_dict()}, os.path.join(args.out_dir, f"critic-ep{ep}.pt"))
            torch.save(gb.state_dict(), os.path.join(args.out_dir, f"gb-ep{ep}.pt"))
            torch.save(tr.state_dict(), os.path.join(args.out_dir, f"tr-ep{ep}.pt"))
            if hasattr(rerank, "state_dict"):
                torch.save(rerank.state_dict(), os.path.join(args.out_dir, f"rerank-ep{ep}.pt"))
        recall_pool_rate = pool_recall_hits / max(1, total_pools)
        fallback_rate = fallback_uses / max(1, total_pools)
        print(f"[ckpt] saved epoch {ep}  "
              f"EM1={em1_sum/max(1,seen):.3f}  EM@{K}={emk_sum/max(1,seen):.3f}  "
              f"recall@pool={recall_pool_rate:.3f}  fallback_rate={fallback_rate:.3f}")
        maybe_print(step_end=True)


if __name__ == "__main__":
    main()
