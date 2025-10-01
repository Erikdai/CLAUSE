#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, re, unicodedata, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from urllib.parse import unquote

import numpy as np

# ---- FAISS ----
try:
    import faiss
except Exception as e:
    raise ImportError("FAISS is not installed. Please run: pip install faiss-cpu") from e


# ============== Text utilities ==============

def _normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9_ \-:/\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_key(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("_", " ")
    s = " ".join(s.split())
    return s

def pretty_name(uri_or_label: str) -> str:
    if not uri_or_label:
        return ""
    s = uri_or_label
    if s.startswith("http://") or s.startswith("https://"):
        s = unquote(s).rstrip("/").split("/")[-1]
    s = s.replace("_", " ")
    return s

def _dedup_preserve_triples(edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    seen = set()
    out = []
    for h, r, t in edges:
        key = (h, r, t)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ============== Cache loading ==============

REQUIRED_FILES = [
    "node_ids.json", "rel_ids.json", "edges.json",
    "node_texts.json", "rel_texts.json",
    "node_vecs.npy", "rel_vecs.npy",
    "node_index.faiss", "rel_index.faiss",
]

def _check_cache_dir(cache_dir: Path) -> Optional[str]:
    if not cache_dir.exists():
        return f"Index directory not found: {cache_dir}"
    for fn in REQUIRED_FILES:
        if not (cache_dir / fn).exists():
            return f"Required file not found: {fn} (dir: {cache_dir})"
    return None

def _load_cache(cache_dir: Path):
    node_ids = json.loads((cache_dir / "node_ids.json").read_text(encoding="utf-8"))
    rel_ids  = json.loads((cache_dir / "rel_ids.json").read_text(encoding="utf-8"))
    edges_js = json.loads((cache_dir / "edges.json").read_text(encoding="utf-8"))
    edges = [(e["h"], e["r"], e["t"]) for e in edges_js]

    node_texts = json.loads((cache_dir / "node_texts.json").read_text(encoding="utf-8"))
    rel_texts  = json.loads((cache_dir / "rel_texts.json").read_text(encoding="utf-8"))

    node_vecs = np.load(cache_dir / "node_vecs.npy")
    rel_vecs  = np.load(cache_dir / "rel_vecs.npy")

    node_index = faiss.read_index(str(cache_dir / "node_index.faiss"))
    rel_index  = faiss.read_index(str(cache_dir / "rel_index.faiss"))

    return node_ids, rel_ids, edges, node_texts, rel_texts, node_vecs, rel_vecs, node_index, rel_index


# ============== Query helpers ==============

def _extract_bracket_mentions(question: str) -> List[str]:
    m = re.findall(r"\[([^\]]+)\]", question or "")
    seen, out = set(), []
    for x in m:
        x = x.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def _build_label_indexes(node_ids: List[str], node_texts: List[str]) -> Dict[str, List[str]]:
    """
    Build a reverse index from normalized labels to node ids.
    Matching supports both the label used during vectorization and the tail segment of the id.
    """
    idx: Dict[str, List[str]] = {}
    for nid, lab in zip(node_ids, node_texts):
        key1 = _norm_key(lab)                     # normalized label text
        key2 = _norm_key(pretty_name(str(nid)))   # normalized id tail / human-readable name
        idx.setdefault(key1, []).append(nid)
        idx.setdefault(key2, []).append(nid)
    return idx

def _match_mentions_to_ids(mentions: List[str], label_norm2ids: Dict[str, List[str]], topk_per_mention: int = 5) -> List[str]:
    hits: List[str] = []
    for name in mentions:
        key = _norm_key(name)
        exact = label_norm2ids.get(key, [])
        if exact:
            hits.extend(exact[:topk_per_mention])
            continue
        # Fallback: containment match
        cnt = 0
        for k, ids in label_norm2ids.items():
            if key in k or k in key:
                for nid in ids:
                    hits.append(nid)
                    cnt += 1
                    if cnt >= topk_per_mention:
                        break
            if cnt >= topk_per_mention:
                break
    # Deduplicate while preserving order
    out, seen = [], set()
    for nid in hits:
        if nid not in seen:
            out.append(nid); seen.add(nid)
    return out

def _embed_query_hash(q: str, dim: int) -> np.ndarray:
    """
    Map a query into a hashed vector with the same dimensionality as cached vectors.
    This enables coarse retrieval without storing TF-IDF/SVD models.
    """
    hv = np.zeros((dim,), dtype="float32")
    tokens = _normalize_text(q).split()
    if not tokens:
        hv[0] = 1.0
        return hv.reshape(1, -1)
    for t in tokens:
        idx = (hash(t) % dim)
        hv[idx] += 1.0
    n = np.linalg.norm(hv) + 1e-12
    hv = (hv / n).astype("float32")
    return hv.reshape(1, -1)

def _faiss_topk(index: faiss.Index, q_vec: np.ndarray, ids: List[str], topk: int) -> List[str]:
    k = min(topk, len(ids))
    D, I = index.search(q_vec.astype("float32"), k=k)
    idxs = [int(i) for i in I[0] if 0 <= i < len(ids)]
    return [ids[i] for i in idxs]


# ============== Subgraph construction (neighbor full expansion) ==============

def build_subgraph(
    cache_dir: str,
    question: str,
    k_nodes: int = 20,
    cap_nodes: int = 30000,
    cap_edges: int = 60000,
    verbose: bool = True,
):
    """
    Pipeline:
    1) Use bracketed entities → ids; otherwise, use hashed full-sentence vector → FAISS → seed nodes.
    2) Collect all outgoing/incoming edges for seed nodes.
    3) Record all "1-hop neighbor" nodes and perform a full expansion (all outgoing/incoming edges) on them.
    4) Deduplicate and clip to caps.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    err = _check_cache_dir(cache_dir)
    if err:
        print(err)
        return {"error": err}

    node_ids, rel_ids, edges, node_texts, rel_texts, node_vecs, rel_vecs, node_index, rel_index = _load_cache(cache_dir)

    # Reverse label index: bracket mention → node_id
    label_norm2ids = _build_label_indexes(node_ids, node_texts)

    # Adjacency (out/in)
    adj_out: Dict[str, List[Tuple[str, str]]] = {}
    adj_in:  Dict[str, List[Tuple[str, str]]]  = {}
    for h, r, t in edges:
        adj_out.setdefault(h, []).append((r, t))
        adj_in.setdefault(t, []).append((h, r))

    q_norm = _normalize_text(question)
    mentions = _extract_bracket_mentions(question)

    # 1) Seed nodes
    hard_seeds = _match_mentions_to_ids(mentions, label_norm2ids, topk_per_mention=5) if mentions else []
    if hard_seeds:
        seed_entities = hard_seeds[:k_nodes]
    else:
        qv = _embed_query_hash(q_norm, node_vecs.shape[1])
        seed_entities = _faiss_topk(node_index, qv, node_ids, topk=k_nodes)

    if verbose:
        print(f"[query] mentions={mentions}  seeds={seed_entities[:5]}{'...' if len(seed_entities)>5 else ''}")

    # 2) One-hop edges of seeds
    collected_nodes: set = set(seed_entities)
    collected_edges: List[Tuple[str, str, str]] = []
    neighbors: set = set()  # 1-hop neighbors

    def add_out_edges(s: str):
        for (r, t) in adj_out.get(s, []):
            collected_edges.append((s, r, t))
            collected_nodes.add(s); collected_nodes.add(t)
            neighbors.add(t)
            if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes:
                return

    def add_in_edges(s: str):
        for (h, r) in adj_in.get(s, []):
            collected_edges.append((h, r, s))
            collected_nodes.add(h); collected_nodes.add(s)
            neighbors.add(h)
            if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes:
                return

    for s in seed_entities:
        add_out_edges(s)
        if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes: break
        add_in_edges(s)
        if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes: break

    # 3) Full expansion for each neighbor (all out/in edges)
    for fn in list(neighbors):
        # Outgoing edges
        for (r, t) in adj_out.get(fn, []):
            collected_edges.append((fn, r, t))
            collected_nodes.add(fn); collected_nodes.add(t)
            if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes: break
        if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes: break
        # Incoming edges
        for (h, r) in adj_in.get(fn, []):
            collected_edges.append((h, r, fn))
            collected_nodes.add(h); collected_nodes.add(fn)
            if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes: break
        if len(collected_edges) >= cap_edges or len(collected_nodes) >= cap_nodes: break

    # 4) Deduplicate and clip
    edges_final = _dedup_preserve_triples(collected_edges)[:cap_edges]
    nodes_final = list(collected_nodes)[:cap_nodes]

    return {
        "mentions": mentions,
        "seed_entities": seed_entities,
        "nodes": nodes_final,
        "edges": edges_final,   # raw (h, r, t)
        "meta": {
            "cache_dir": str(cache_dir),
            "caps": {"nodes": cap_nodes, "edges": cap_edges},
            "note": "seed 1-hop + neighbors full expansion (both in/out); no hand-crafted rules"
        }
    }


# ============== Output ==============

def dump_result(res: Dict[str, Any], fmt: str = "tsv", pretty: bool = False, out_path: Optional[str] = None):
    """
    fmt: tsv | jsonl | json
    out_path: if provided, write to file; otherwise print to stdout
    """
    if "error" in res:
        return

    edges = res.get("edges", [])
    out_lines: List[str] = []

    if fmt == "tsv":
        for h, r, t in edges:
            if pretty:
                out_lines.append(f"{pretty_name(h)}\t{pretty_name(r)}\t{pretty_name(t)}")
            else:
                out_lines.append(f"{h}\t{r}\t{t}")
    elif fmt == "jsonl":
        for h, r, t in edges:
            if pretty:
                obj = {"h": pretty_name(h), "r": pretty_name(r), "t": pretty_name(t)}
            else:
                obj = {"h": h, "r": r, "t": t}
            out_lines.append(json.dumps(obj, ensure_ascii=False))
    else:  # json
        if pretty:
            out_obj = {"edges": [{"h": pretty_name(h), "r": pretty_name(r), "t": pretty_name(t)} for (h, r, t) in edges]}
        else:
            out_obj = {"edges": [{"h": h, "r": r, "t": t} for (h, r, t) in edges]}
        out_lines = [json.dumps(out_obj, ensure_ascii=False, indent=2)]

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        print(f"[info] Saved {len(edges)} edges to {out_path}", file=sys.stderr)
    else:
        for line in out_lines:
            print(line)


# ============== CLI ==============

def _build_argparser():
    p = argparse.ArgumentParser(description="Build a subgraph from an existing index (neighbor full expansion; no hand-crafted rules).")
    p.add_argument("--cache", required=True, help="Index directory")
    p.add_argument("--q", "--question", required=True, dest="question", help="Question")
    p.add_argument("--k_nodes", type=int, default=20, help="Upper bound on retrieved seed nodes")
    p.add_argument("--cap_nodes", type=int, default=30000, help="Subgraph node cap")
    p.add_argument("--cap_edges", type=int, default=60000, help="Subgraph edge cap")
    p.add_argument("--format", choices=["tsv", "jsonl", "json"], default="tsv", help="Output format (default: tsv)")
    p.add_argument("--pretty", action="store_true", help="Apply pretty_name to h/r/t (default off, outputs raw KB strings)")
    p.add_argument("--out", type=str, default="", help="Output file path (leave empty to print to stdout)")
    p.add_argument("--quiet", action="store_true", help="Reduce log output")
    return p

def main():
    args = _build_argparser().parse_args()
    res = build_subgraph(
        cache_dir=args.cache,
        question=args.question,
        k_nodes=args.k_nodes,
        cap_nodes=args.cap_nodes,
        cap_edges=args.cap_edges,
        verbose=not args.quiet,
    )
    if "error" in res:
        sys.exit(1)
    dump_result(res, fmt=args.format, pretty=args.pretty, out_path=args.out or None)

if __name__ == "__main__":
    main()
