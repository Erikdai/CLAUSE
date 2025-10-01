# -*- coding: utf-8 -*-
from typing import Dict, Any, Tuple, Set, List, Optional, Iterable
from enum import Enum
import os, json, logging, re, unicodedata

import numpy as np
import torch
from torch import nn
import torch.distributions as D
from types import SimpleNamespace

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import faiss
except Exception as e:
    raise ImportError("Missing dependency faiss. Install: pip install faiss-cpu") from e

# ---- subgraph utilities ----
try:
    from ..subgraph import (
        Embedder,
        parse_ttl_cached,
        build_indices_cached,
        retrieve_candidates,
        textualize,
    )
    logger.info("Successfully imported subgraph module using relative import")
except ImportError as e:
    logger.error(f"Failed to import subgraph module: {e}")
    raise ImportError(
        "Could not import required subgraph module. "
        "Ensure LogiKG-MAPPO is properly installed and subgraph module is available. "
        f"Error: {e}"
    )

# ---- Optional spaCy NER ----
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False


class EdgeOperation(Enum):
    ADD_EDGE = "add_edge"
    DELETE_EDGE = "delete_edge"
    STOP = "stop"


class EnhancedGraphBuilder(nn.Module):
    def __init__(
        self,
        ttl_file: Optional[str] = None,
        cache_dir: str = "cache_beam",
        embedder_name: str = "thenlper/gte-small",
        device: Optional[str] = None,
        hidden: int = 128,
        max_operations: int = 20,
        max_candidates_per_operation: int = 50,
        use_cache_only: bool = False,
        # NER
        enable_ner: bool = True,
        ner_model: str = "en_core_web_sm",
        ner_label_whitelist: Optional[Iterable[str]] = None,
        # fallback
        fallback_nodes_k: int = 24,
        # soft re-ranking
        w_name: float = 0.55,
        w_rel: float  = 0.15,
        w_ctx: float  = 0.25,
        w_pop: float  = 0.05,
        # cross-encoder (optional)
        enable_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_topk: int = 20,
    ):
        super().__init__()
        self.ttl_file = ttl_file
        self.cache_dir = cache_dir
        self.max_operations = max_operations
        self.max_candidates_per_operation = max_candidates_per_operation
        self.use_cache_only = use_cache_only

        self.enable_ner = enable_ner and _HAS_SPACY
        self.ner_model = ner_model
        self.ner_label_whitelist = set(ner_label_whitelist) if ner_label_whitelist else None
        self.fallback_nodes_k = max(1, int(fallback_nodes_k))

        self.w_name = float(w_name)
        self.w_rel  = float(w_rel)
        self.w_ctx  = float(w_ctx)
        self.w_pop  = float(w_pop)

        self.enable_cross_encoder = enable_cross_encoder
        self.cross_encoder_model = cross_encoder_model
        self.cross_encoder_topk  = int(cross_encoder_topk)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device

        # sentence embedder
        self.embedder = Embedder(model_name=embedder_name, device=device)

        # probe input dim
        try:
            with torch.no_grad():
                _probe = self.embedder.encode(["probe"])[0]
            in_dim = int(_probe.shape[0])
        except Exception:
            in_dim = 384

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.operation_head = nn.Linear(hidden, 3)
        # Initialize to prefer "add_edge" (index 0) over "stop" (index 2)
        with torch.no_grad():
            self.operation_head.bias[0] = 1.0  # add_edge
            self.operation_head.bias[1] = 0.0  # delete_edge  
            self.operation_head.bias[2] = -1.0  # stop

        # (reserved) learned heads
        self.entity_head = nn.Linear(hidden, hidden // 2)
        self.relation_head = nn.Linear(hidden, hidden // 2)
        self.candidate_scorer = nn.Linear(hidden, 1)

        # KB/indices
        self._init_subgraph_data()
        self._build_graph_views()
        self._build_popularity()
        self._build_rel_embeddings()
        self._build_anchor_indexes()  # <<< New: anchor name indexes

        # NER
        self._nlp = None
        if self.enable_ner:
            try:
                self._nlp = spacy.load(self.ner_model)
                logger.info(f"spaCy model '{self.ner_model}' loaded for NER.")
            except Exception as e:
                logger.warning(f"spaCy model load failed ({e}); NER disabled.")
                self.enable_ner = False

        # Cross-Encoder (optional)
        self._ce = None
        if self.enable_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder(self.cross_encoder_model, device=self.device_str)
                logger.info(f"Cross-Encoder '{self.cross_encoder_model}' loaded for final re-ranking.")
            except Exception as e:
                logger.warning(f"Cross-Encoder load failed ({e}); disabled.")
                self.enable_cross_encoder = False
                self._ce = None

        # subgraph state
        self.current_nodes: Set[str] = set()
        self.current_edges: Set[Tuple[str, str, str]] = set()
        self.operation_count: int = 0
        self.latest_outputs: Dict[str, Any] = {}

        self.to(self.device_str)

        # --- safety belt: if some method is lost due to indentation/hot reload, use fallback ---
        if not hasattr(self, "_align_names_to_nodes") or not callable(getattr(self, "_align_names_to_nodes", None)):
            def _align_fallback(names, topk_each=8, cap_total=50):
                outs, seen = [], set()
                for n in names or []:
                    for nid in self._match_nodes_by_mention(n, topk=max(1, topk_each)):
                        if nid not in seen:
                            seen.add(nid); outs.append(nid)
                        if len(outs) >= cap_total:
                            break
                    if len(outs) >= cap_total:
                        break
                return outs[:cap_total]
            self._align_names_to_nodes = _align_fallback  # bind

        if not hasattr(self, "_match_nodes_by_mention") or not callable(getattr(self, "_match_nodes_by_mention", None)):
            self._match_nodes_by_mention = lambda mention, topk=20: \
                self._align_names_to_nodes([mention], topk_each=max(1, topk), cap_total=max(1, topk))

    # ------------------------------ utils ------------------------------ #
    @staticmethod
    def _normalize(s: str) -> str:
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = s.replace("\t", " ").replace("\n", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("_", " ")
        return s

    @staticmethod
    def _query_text_from_obs(obs: Any) -> str:
        """Supports str or dict (containing question/query/text); only takes left side of \t"""
        if isinstance(obs, str):
            q = obs
        elif isinstance(obs, dict):
            q = None
            for k in ("question", "query", "text"):
                if k in obs and isinstance(obs[k], str):
                    q = obs[k]; break
            if q is None:
                return ""
        else:
            return ""
        if "\t" in q:
            q = q.split("\t", 1)[0]
        return q.strip()

    @staticmethod
    def _sanitize_question(q: str) -> str:
        """Remove bracketed entity mentions from question to reduce name-matching bias."""
        try:
            import re
            return re.sub(r"\[(.+?)\]", " ", q or "").strip()
        except Exception:
            return q or ""

    # --- Add near other utilities in EnhancedGraphBuilder -----------------
    def _textualize_node_with_neigh(self, nid: str, max_neigh: int = 2) -> str:
        """Compact snippet: node label + up to a few incident triples (out+in)."""
        # Resolve node label
        if isinstance(self.node_pack.texts, dict):
            name_txt = self.node_pack.texts.get(nid, self.node_pack.texts.get(str(nid), str(nid)))
        else:
            idx = self._node_index_map.get(nid)
            if isinstance(self.node_pack.texts, list) and isinstance(idx, int) and idx < len(self.node_pack.texts):
                name_txt = self.node_pack.texts[idx] or str(nid)
            else:
                name_txt = str(nid)

        parts = [name_txt]
        # out edges
        for r, t in self.adj.get(nid, [])[:max_neigh]:
            t_label = t
            if isinstance(self.node_pack.texts, dict):
                t_label = self.node_pack.texts.get(t, self.node_pack.texts.get(str(t), str(t)))
            elif isinstance(self.node_pack.texts, list):
                ti = self._node_index_map.get(t)
                if isinstance(ti, int) and ti < len(self.node_pack.texts):
                    t_label = self.node_pack.texts[ti] or str(t)
            parts.append(f"{name_txt} {r} {t_label}")

        # in edges
        for r, h in self.adj_in.get(nid, [])[:max_neigh]:
            h_label = h
            if isinstance(self.node_pack.texts, dict):
                h_label = self.node_pack.texts.get(h, self.node_pack.texts.get(str(h), str(h)))
            elif isinstance(self.node_pack.texts, list):
                hi = self._node_index_map.get(h)
                if isinstance(hi, int) and hi < len(self.node_pack.texts):
                    h_label = self.node_pack.texts[hi] or str(h)
            parts.append(f"{h_label} {r} {name_txt}")

        snippet = " ; ".join(parts)
        return snippet

    def build_context_pool(self, question: str, max_pool: int = 40, max_neigh_per_node: int = 2):
        """
        Build a list of candidates for the Dynamic Reranker to select from.
        Each item: {'id': <node_id>, 'text': <snippet>, 'approx_tokens': int, 'prior': float}.
        """
        # Start from high-confidence seeds + current subgraph nodes
        try:
            seeds = self._get_candidate_entities(question, max_candidates=min(max_pool, self.max_candidates_per_operation))
        except Exception:
            seeds = []
        node_ids = list(self.current_nodes) + [nid for nid in seeds if nid not in self.current_nodes]
        node_ids = node_ids[: max(1, max_pool)]

        # Precompute query embedding (sanitized)
        try:
            qv = self.embedder.encode([self._sanitize_question(question)])[0].astype("float32")
            qv = qv / (np.linalg.norm(qv) + 1e-8)
        except Exception:
            qv = None

        # Build snippets first
        uniq, texts, ids = set(), [], []
        for nid in node_ids:
            if nid in uniq:
                continue
            uniq.add(nid)
            txt = self._textualize_node_with_neigh(nid, max_neigh=max_neigh_per_node)
            texts.append(txt)
            ids.append(nid)
            if len(texts) >= max_pool:
                break

        # Batch compute priors for all snippets
        priors = None
        try:
            if texts:
                mat = self.embedder.encode(texts).astype("float32")
                mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
                if qv is not None:
                    priors = (mat @ qv).astype("float32")
        except Exception:
            priors = None

        pool = []
        for i, (nid, text) in enumerate(zip(ids, texts)):
            try:
                approx_tokens = int(max(1, round(len(text.split()) / 0.75)))
            except Exception:
                approx_tokens = max(1, len(text.split()))
            prior = float(priors[i]) if priors is not None and i < len(priors) else 0.0
            label = self._node_id_to_text(nid)
            pool.append({"id": nid, "label": label, "text": text, "approx_tokens": int(approx_tokens), "prior": prior})

        # Stash to latest outputs for downstream usage
        try:
            self.latest_outputs["context_pool"] = pool
        except Exception:
            pass
        return pool
    # ---------------- Context pool for reranking ---------------- #
    def _node_id_to_text(self, nid: str) -> str:
        idx = self._node_index_map.get(nid)
        texts = getattr(self.node_pack, "texts", None)
        if isinstance(texts, dict):
            return texts.get(nid, texts.get(str(nid), str(nid)))
        if isinstance(texts, list) and isinstance(idx, int) and 0 <= idx < len(texts):
            return texts[idx] or str(nid)
        return str(nid)

    def get_context_pool(self, question: str, limit: int = 64) -> List[str]:
        """
        Build a pool of short context strings for reranking, combining textualized
        edges from the current subgraph and node labels; fall back to seed recall.
        """
        pool: List[str] = []
        seen = set()

        # 1) textualize current edges
        try:
            for (h, r, t) in list(self.current_edges)[: max(1, limit // 2)]:
                h_txt = self._node_id_to_text(h)
                t_txt = self._node_id_to_text(t)
                s = f"{h_txt} {r} {t_txt}"
                if s and s not in seen:
                    seen.add(s); pool.append(s)
        except Exception:
            pass

        # 2) node labels
        try:
            for nid in list(self.current_nodes)[: max(1, limit // 2)]:
                name = self._node_id_to_text(nid)
                if name and name not in seen:
                    seen.add(name); pool.append(name)
        except Exception:
            pass

        # 3) fallback: seeds by retriever if not enough
        if len(pool) < limit and isinstance(question, str) and question.strip():
            try:
                seeds = self._get_candidate_entities(question, max_candidates=limit)
            except Exception:
                seeds = []
            for nid in seeds:
                name = self._node_id_to_text(nid)
                if name and name not in seen:
                    seen.add(name); pool.append(name)
                if len(pool) >= limit:
                    break

        return pool[:limit]

    # --------------------- cache / KB init --------------------- #
    def _resolve_cache_files(self) -> dict:
        cd = self.cache_dir
        untagged = {
            "node_ids":   os.path.join(cd, "node_ids.json"),
            "node_texts": os.path.join(cd, "node_texts.json"),
            "node_vecs":  os.path.join(cd, "node_vecs.npy"),
            "node_index": os.path.join(cd, "node_index.faiss"),
            "rel_ids":    os.path.join(cd, "rel_ids.json"),
            "rel_texts":  os.path.join(cd, "rel_texts.json"),
            "rel_vecs":   os.path.join(cd, "rel_vecs.npy"),
            "rel_index":  os.path.join(cd, "rel_index.faiss"),
            "edges":      os.path.join(cd, "edges.json"),
        }
        must = [untagged[k] for k in ("node_ids","node_texts","node_vecs","node_index",
                                      "rel_ids","rel_texts","rel_vecs","rel_index")]
        if all(os.path.exists(p) for p in must):
            return untagged

        tag = None
        if os.path.isdir(cd):
            for fname in os.listdir(cd):
                m = re.match(r"(node_ids|node_texts|rel_ids|rel_texts)_(.+)\.(json)$", fname)
                if m:
                    tag = m.group(2); break
        if tag is None:
            raise FileNotFoundError(
                f"No compatible cache files found in cache_dir: {cd}. "
                "Expected untagged (node_ids.json, ...) or tagged (node_ids_<tag>.json, ...)."
            )

        tagged = {
            "node_ids":   os.path.join(cd, f"node_ids_{tag}.json"),
            "node_texts": os.path.join(cd, f"node_texts_{tag}.json"),
            "node_vecs":  os.path.join(cd, f"node_vecs_{tag}.npy"),
            "node_index": os.path.join(cd, f"node_index_{tag}.faiss"),
            "rel_ids":    os.path.join(cd, f"rel_ids_{tag}.json"),
            "rel_texts":  os.path.join(cd, f"rel_texts_{tag}.json"),
            "rel_vecs":   os.path.join(cd, f"rel_vecs_{tag}.npy"),
            "rel_index":  os.path.join(cd, f"rel_index_{tag}.faiss"),
            "edges":      os.path.join(cd, "edges.json"),
        }
        must = [tagged[k] for k in ("node_ids","node_texts","node_vecs","node_index",
                                    "rel_ids","rel_texts","rel_vecs","rel_index")]
        if not all(os.path.exists(p) for p in must):
            missing = [p for p in must if not os.path.exists(p)]
            raise FileNotFoundError(f"Tagged cache selected (tag='{tag}'), but missing files: {missing}")
        return tagged

    def _init_subgraph_data(self) -> None:
        if self.use_cache_only:
            paths = self._resolve_cache_files()

            with open(paths["node_ids"], "r", encoding="utf-8") as f:
                self.node_ids = json.load(f)
            with open(paths["rel_ids"], "r", encoding="utf-8") as f:
                self.rel_ids = json.load(f)

            self.node_vecs = np.load(paths["node_vecs"])
            self.rel_vecs  = np.load(paths["rel_vecs"])

            self.node_index = faiss.read_index(paths["node_index"])
            self.rel_index  = faiss.read_index(paths["rel_index"])

            with open(paths["node_texts"], "r", encoding="utf-8") as f:
                self.node_texts = json.load(f)
            with open(paths["rel_texts"], "r", encoding="utf-8") as f:
                self.rel_texts = json.load(f)

            if os.path.exists(paths["edges"]):
                with open(paths["edges"], "r", encoding="utf-8") as f:
                    self.edges = [(e["h"], e["r"], e["t"]) for e in json.load(f)]
            else:
                self.edges = []

            print(f"[✓] Loaded cached KB index from {self.cache_dir} "
                  f"(nodes={len(self.node_ids)}, rels={len(self.rel_ids)}, edges={len(self.edges)})")

            self.node_pack = SimpleNamespace(
                ids=self.node_ids, vecs=self.node_vecs, index=self.node_index, texts=self.node_texts
            )
            self.rel_pack = SimpleNamespace(
                ids=self.rel_ids,  vecs=self.rel_vecs,  index=self.rel_index,  texts=self.rel_texts
            )

        else:
            nodes, edges, rel_text = parse_ttl_cached(
                ttl_path=self.ttl_file,
                max_desc=2,
                cache_dir=self.cache_dir,
                rebuild=False,
            )
            self.nodes = nodes
            self.edges = edges
            self.rel_text = rel_text

            node_pack, rel_pack = build_indices_cached(
                nodes=self.nodes,
                rel_text=self.rel_text,
                embedder=self.embedder,
                cache_dir=self.cache_dir,
                rebuild=False,
            )
            self.node_pack = node_pack
            self.rel_pack = rel_pack

    def _build_graph_views(self):
        self.adj: Dict[str, List[Tuple[str, str]]] = {}
        self.adj_in: Dict[str, List[Tuple[str, str]]] = {}  # tail -> [(rel, head)]

        for h, r, t in getattr(self, "edges", []):
            self.adj.setdefault(h, []).append((r, t))
            self.adj_in.setdefault(t, []).append((r, h))

        self.edge_set: Set[Tuple[str, str, str]] = set(getattr(self, "edges", []))

        # id -> index mapping
        if isinstance(self.node_pack.ids, list):
            self._node_index_map = {nid: i for i, nid in enumerate(self.node_pack.ids)}
        else:
            self._node_index_map = dict(self.node_pack.ids)
        if isinstance(self.rel_pack.ids, list):
            self._rel_index_map = {rid: i for i, rid in enumerate(self.rel_pack.ids)}
        else:
            self._rel_index_map = dict(self.rel_pack.ids)

        # normalized texts
        self._node_texts_norm: List[str] = []
        self._node_id_norm: List[str] = []
        node_texts = getattr(self.node_pack, "texts", [])
        ids = list(self.node_pack.ids) if isinstance(self.node_pack.ids, list) else list(self.node_pack.ids.keys())
        for i, nid in enumerate(ids):
            if isinstance(node_texts, list):
                t = node_texts[i] if i < len(node_texts) else ""
            elif isinstance(node_texts, dict):
                t = node_texts.get(nid, node_texts.get(str(nid), ""))
            else:
                t = ""
            self._node_texts_norm.append(self._normalize(f"{nid} {t}"))
            self._node_id_norm.append(self._normalize(str(nid)))

    def _build_popularity(self):
        """Degree as structural prior."""
        self._deg: Dict[str, int] = {}
        for h, r, t in getattr(self, "edges", []):
            self._deg[h] = self._deg.get(h, 0) + 1
            self._deg[t] = self._deg.get(t, 0) + 1

    def _build_rel_embeddings(self):
        """Embed rel_texts; fallback to 'X <rel> Y' when missing."""
        rel_ids_obj = getattr(self.rel_pack, "ids", [])
        if isinstance(rel_ids_obj, list):
            rid_list = rel_ids_obj
            rid_to_idx = {rid: i for i, rid in enumerate(rid_list)}
        elif isinstance(rel_ids_obj, dict):
            rid_to_idx = dict(rel_ids_obj)  # rid -> idx
            rid_list = [rid for rid, _ in sorted(rid_to_idx.items(), key=lambda kv: kv[1])]
        else:
            rid_list = []
            rid_to_idx = {}

        texts_obj = getattr(self.rel_pack, "texts", {})

        rel_text_map: Dict[str, str] = {}
        if isinstance(texts_obj, dict):
            for rid in rid_list:
                txt = texts_obj.get(rid, texts_obj.get(str(rid), "")) or ""
                if not txt:
                    txt = f"X {rid} Y"
                rel_text_map[str(rid)] = txt
        elif isinstance(texts_obj, list):
            if rid_to_idx:
                for rid in rid_list:
                    idx = rid_to_idx.get(rid, None)
                    txt = ""
                    if isinstance(idx, int) and 0 <= idx < len(texts_obj):
                        txt = texts_obj[idx] or ""
                    if not txt:
                        txt = f"X {rid} Y"
                    rel_text_map[str(rid)] = txt
            else:
                for i, rid in enumerate(rid_list):
                    txt = texts_obj[i] if i < len(texts_obj) else ""
                    if not txt:
                        txt = f"X {rid} Y"
                    rel_text_map[str(rid)] = txt
        else:
            for rid in rid_list:
                rel_text_map[str(rid)] = f"X {rid} Y"

        self.rel_text_map = rel_text_map

        # embed relation texts
        self.rel_vecs: Dict[str, np.ndarray] = {}
        for rid, txt in rel_text_map.items():
            vec = self.embedder.encode([txt])[0].astype("float32")
            self.rel_vecs[rid] = vec

    # ---------------- NER ---------------- #
    def _extract_ner_spans(self, question: str) -> List[str]:
        if not (self.enable_ner and self._nlp):
            return []
        doc = self._nlp(question)
        if self.ner_label_whitelist is None:
            return [ent.text for ent in doc.ents]
        return [ent.text for ent in doc.ents if ent.label_ in self.ner_label_whitelist]

    # ---------------- Anchor name mapping (strong) ---------------- #
    @staticmethod
    def _name_tokens(s: str) -> List[str]:
        s = EnhancedGraphBuilder._normalize(s)
        toks = re.split(r"[^a-z0-9]+", s)
        return [t for t in toks if len(t) >= 2]

    @staticmethod
    def _str_ratio(a: str, b: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(a=EnhancedGraphBuilder._normalize(a), b=EnhancedGraphBuilder._normalize(b)).ratio()

    def _gen_name_variants(self, name: str) -> List[str]:
        n = self._normalize(name)
        res = {n, n.replace("-", " "), n.replace(".", " ")}
        if "," in n:
            parts = [p.strip() for p in n.split(",")]
            if len(parts) == 2:
                res.add(f"{parts[1]} {parts[0]}")
        toks = n.split()
        if len(toks) >= 3:
            mid = toks[1]
            if len(mid) > 1:
                res.add(" ".join([toks[0], mid[0], toks[-1]]))
                res.add(" ".join([toks[0], toks[-1]]))
        return [x for x in res if x]

    def _build_anchor_indexes(self):
        """Build for all nodes: name variant dictionary + token inverted index + lightweight stopwords (auto-counted from KB)"""
        self._variant2ids: Dict[str, Set[str]] = {}
        self._tok2ids: Dict[str, Set[str]] = {}
        self._stopwords: Set[str] = set()

        # Count word frequency → take top-50 as stopwords (avoid manual writing)
        freq: Dict[str, int] = {}
        ids = list(self.node_pack.ids) if isinstance(self.node_pack.ids, list) else list(self.node_pack.ids.keys())
        for i, nid in enumerate(ids):
            if isinstance(self.node_pack.texts, list):
                name_txt = self.node_pack.texts[i] if i < len(self.node_pack.texts) else str(nid)
            elif isinstance(self.node_pack.texts, dict):
                name_txt = self.node_pack.texts.get(nid, self.node_pack.texts.get(str(nid), str(nid)))
            else:
                name_txt = str(nid)
            for t in self._name_tokens(name_txt):
                freq[t] = self._deg.get(t, 0) + 1 if t in self._deg else freq.get(t, 0) + 1
        for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            self._stopwords.add(w)

        # Build inverted index and variant mapping
        for i, nid in enumerate(ids):
            if isinstance(self.node_pack.texts, list):
                name_txt = self.node_pack.texts[i] if i < len(self.node_pack.texts) else str(nid)
            elif isinstance(self.node_pack.texts, dict):
                name_txt = self.node_pack.texts.get(nid, self.node_pack.texts.get(str(nid), str(nid)))
            else:
                name_txt = str(nid)

            variants = set(self._gen_name_variants(name_txt))
            variants.add(self._normalize(str(nid)))
            for v in variants:
                self._variant2ids.setdefault(v, set()).add(nid)

            toks = set(t for t in self._name_tokens(name_txt) if t not in self._stopwords)
            for t in toks:
                self._tok2ids.setdefault(t, set()).add(nid)

    def _resolve_anchor_exact(self, raw_name: str, question: str, limit: int = 10) -> List[str]:
        """
        Strong anchor entity→node matcher:
        1) Complete variant match
        2) Inverted token recall + token coverage threshold + character similarity threshold
        3) If ambiguous, use relation/neighborhood soft disambiguation (not hard-coded)
        """
        name = self._normalize(raw_name)
        if not name:
            return []

        # 1) Complete variant match
        vid = list(self._variant2ids.get(name, []))
        if vid:
            return vid[:limit]

        # 2) Inverted token recall
        name_toks = [t for t in self._name_tokens(raw_name) if t not in self._stopwords]
        if not name_toks:
            return []

        pool: Set[str] = set()
        for t in name_toks:
            pool |= self._tok2ids.get(t, set())
        if not pool:
            return []

        # Coverage/similarity threshold
        def _cand_text(nid: str) -> str:
            idx = self._node_index_map.get(nid)
            if isinstance(self.node_pack.texts, list) and isinstance(idx, int) and idx < len(self.node_pack.texts):
                return self.node_pack.texts[idx] or str(nid)
            if isinstance(self.node_pack.texts, dict):
                return self.node_pack.texts.get(nid, self.node_pack.texts.get(str(nid), str(nid)))
            return str(nid)

        scored = []
        name_tok_set = set(name_toks)
        min_required = 2 if len(name_tok_set) >= 2 else 1

        for nid in pool:
            cand = _cand_text(nid)
            ctoks = [t for t in self._name_tokens(cand) if t not in self._stopwords]
            overlap = len(name_tok_set & set(ctoks))
            if overlap < min_required:   # Insufficient token coverage
                continue
            sim = self._str_ratio(raw_name, cand)
            if sim < 0.88:               # Character similarity gate (empirical threshold)
                continue
            scored.append((overlap + sim, sim, -len(cand), nid))

        if not scored:
            return []

        scored.sort(reverse=True)
        cands = [nid for _, __, ___, nid in scored[:max(limit, 20)]]

        # 3) Ambiguous → soft disambiguation (relation/neighborhood)
        if len(cands) <= 2:
            return cands[:limit]

        # Use existing soft reranking, but execute only within cands subset
        reranked = self._soft_rank_candidates(question, cands, topk=limit)
        return reranked[:limit]

    # ---------------- Lightweight name alignment (for fallback and tests) ---------------- #
    def _align_names_to_nodes(self, names: List[str], topk_each: int = 8, cap_total: int = 50) -> List[str]:
        """
        Lightweight name alignment: for each name, first use strong anchor matcher (_resolve_anchor_exact),
        then use FAISS + character similarity gate as fallback if insufficient; no full table scan, stable performance.
        """
        if not names:
            return []
        out: List[str] = []
        seen: Set[str] = set()

        for name in names:
            # 1) Strong anchor
            cands = self._resolve_anchor_exact(name, name, limit=topk_each)
            for nid in cands:
                if nid not in seen:
                    seen.add(nid); out.append(nid)
                if len(out) >= cap_total:
                    return out[:cap_total]

            # 2) FAISS fallback + light gate
            try:
                qv = self.embedder.encode([name])[0].astype("float32")
                k = max(topk_each, 20)
                _, Im = self.node_pack.index.search(qv[None, :], k)

                if isinstance(self.node_pack.ids, list):
                    ids = [self.node_pack.ids[i] for i in Im[0]]
                    idxs = Im[0]
                    def _name_by_idx(i, nid):
                        if isinstance(self.node_pack.texts, list) and 0 <= i < len(self.node_pack.texts):
                            return self.node_pack.texts[i] or str(nid)
                        return str(nid)
                    txts = [_name_by_idx(i, nid) for i, nid in zip(idxs, ids)]
                else:
                    inv = {v: k2 for k2, v in self.node_pack.ids.items()}
                    ids = [inv.get(i) for i in Im[0] if i in inv]
                    def _name_by_id(nid):
                        if isinstance(self.node_pack.texts, dict):
                            return self.node_pack.texts.get(nid, self.node_pack.texts.get(str(nid), str(nid)))
                        return str(nid)
                    txts = [_name_by_id(nid) for nid in ids]

                src_toks = set(self._name_tokens(name))
                scored = []
                for nid, nm in zip(ids, txts):
                    if not nid or nid in seen:
                        continue
                    if len(src_toks & set(self._name_tokens(nm))) < 1:
                        continue
                    sim = self._str_ratio(name, nm)
                    if sim < 0.82:
                        continue
                    scored.append((sim, -len(nm), nid))

                scored.sort(reverse=True)
                for _, __, nid in scored[:topk_each]:
                    if nid not in seen:
                        seen.add(nid); out.append(nid)
                    if len(out) >= cap_total:
                        return out[:cap_total]
            except Exception:
                pass

        return out[:cap_total]

    # Compatible with old interface (used by tests)
    def _match_nodes_by_mention(self, mention: str, topk: int = 20) -> List[str]:
        return self._align_names_to_nodes([mention], topk_each=max(1, topk), cap_total=max(1, topk))

    # ---------------- Lexical fallback ---------------- #
    def _fallback_lexical(self, question: str, texts, ids, topk=10) -> List[str]:
        q = self._normalize(question)
        # Downweight WH/aux tokens that spuriously match titles (e.g., "what richard did")
        STOP = {
            'what','which','who','whom','whose','does','did','do','is','are','was','were','the','a','an',
            'in','on','of','for','to','and','or','with','that','this','these','those','movie','movies','film','films',
            'appear','appears','appearing','star','stars','starred','starring','act','acts','acted','acting'
        }
        toks = [w for w in q.split() if (len(w) >= 3 and w not in STOP)]
        if not toks:
            return []

        items: List[Tuple[str, int]] = []
        if isinstance(ids, list):
            for i, _id in enumerate(ids):
                if isinstance(texts, list) and 0 <= i < len(texts):
                    t = texts[i] or ""
                elif isinstance(texts, dict):
                    t = texts.get(_id, texts.get(str(_id), "")) or ""
                else:
                    t = ""
                score = sum(tok in self._normalize(t) for tok in toks)
                if score > 0:
                    items.append((_id, score))
        elif isinstance(ids, dict):  # id -> idx
            for _id, idx in ids.items():
                if isinstance(texts, list) and 0 <= idx < len(texts):
                    t = texts[idx] or ""
                elif isinstance(texts, dict):
                    t = texts.get(_id, texts.get(str(_id), "")) or ""
                else:
                    t = ""
                score = sum(tok in self._normalize(t) for tok in toks)
                if score > 0:
                    items.append((_id, score))

        items.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in items[:topk]]

    # ---------------- Soft constraint reranking (core) ---------------- #
    def _soft_rank_candidates(self, question: str, cand_ids: List[str], topk: int = 10) -> List[str]:
        if not cand_ids:
            return []
        qv = self.embedder.encode([self._sanitize_question(question)])[0].astype("float32")
        qv /= (np.linalg.norm(qv) + 1e-8)

        r_best = 0.0
        for rv in self.rel_vecs.values():
            rvn = rv / (np.linalg.norm(rv) + 1e-8)
            r_best = max(r_best, float(np.dot(qv, rvn)))

        scored: List[Tuple[float, str]] = []
        for nid in cand_ids:
            idx = self._node_index_map.get(nid)
            if isinstance(self.node_pack.texts, dict):
                name_txt = self.node_pack.texts.get(nid, self.node_pack.texts.get(str(nid), str(nid)))
            elif isinstance(self.node_pack.texts, list) and isinstance(idx, int) and idx < len(self.node_pack.texts):
                name_txt = self.node_pack.texts[idx] or str(nid)
            else:
                name_txt = str(nid)

            name_vec = self.embedder.encode([name_txt])[0].astype("float32")
            name_vec /= (np.linalg.norm(name_vec) + 1e-8)
            s_name = float(np.dot(qv, name_vec))

            neigh_snips: List[str] = []
            for r, t in self.adj.get(nid, [])[:2]:
                tname = t
                if isinstance(self.node_pack.texts, dict):
                    tname = self.node_pack.texts.get(t, self.node_pack.texts.get(str(t), str(t)))
                elif isinstance(self.node_pack.texts, list):
                    ti = self._node_index_map.get(t)
                    if isinstance(ti, int) and ti < len(self.node_pack.texts):
                        tname = self.node_pack.texts[ti] or str(t)
                neigh_snips.append(f"{name_txt} {r} {tname}")
            for r, h in self.adj_in.get(nid, [])[:2]:
                hname = h
                if isinstance(self.node_pack.texts, dict):
                    hname = self.node_pack.texts.get(h, self.node_pack.texts.get(str(h), str(h)))
                elif isinstance(self.node_pack.texts, list):
                    hi = self._node_index_map.get(h)
                    if isinstance(hi, int) and hi < len(self.node_pack.texts):
                        hname = self.node_pack.texts[hi] or str(h)
                neigh_snips.append(f"{hname} {r} {name_txt}")

            s_ctx = 0.0
            for s in neigh_snips[:4]:
                sv = self.embedder.encode([s])[0].astype("float32")
                sv /= (np.linalg.norm(sv) + 1e-8)
                s_ctx = max(s_ctx, float(np.dot(qv, sv)))

            deg = self._deg.get(nid, 0)
            s_pop = np.tanh(np.log1p(deg) / 3.0)

            score = self.w_name*s_name + self.w_rel*r_best + self.w_ctx*s_ctx + self.w_pop*s_pop
            scored.append((score, nid))

        scored.sort(reverse=True)
        return [nid for _, nid in scored[:topk]]

    def _cross_encoder_rerank(self, question: str, cand_ids: List[str], topk: int = 10) -> List[str]:
        if not (self.enable_cross_encoder and self._ce and cand_ids):
            return cand_ids[:topk]
        from typing import Tuple as _Tuple
        pairs: List[_Tuple[str, str]] = []
        id2txt: List[str] = []
        limit = max(topk, self.cross_encoder_topk)
        for nid in cand_ids[:limit]:
            idx = self._node_index_map.get(nid)
            if isinstance(self.node_pack.texts, dict):
                name_txt = self.node_pack.texts.get(nid, self.node_pack.texts.get(str(nid), str(nid)))
            elif isinstance(self.node_pack.texts, list) and isinstance(idx, int) and idx < len(self.node_pack.texts):
                name_txt = self.node_pack.texts[idx] or str(nid)
            else:
                name_txt = str(nid)
            neigh = []
            for r, t in self.adj.get(nid, [])[:2]:
                neigh.append(f"{name_txt} {r} {t}")
            for r, h in self.adj_in.get(nid, [])[:2]:
                neigh.append(f"{h} {r} {name_txt}")
            ctx = " ; ".join(neigh) if neigh else name_txt
            pairs.append((question, f"{name_txt} || {ctx}"))
            id2txt.append(nid)

        scores = self._ce.predict(pairs)
        order = np.argsort(scores)[::-1]
        return [id2txt[i] for i in order[:topk]]

    # ---------------- Candidate recall (including strong anchor + soft reranking) ---------------- #
    def _get_candidate_entities(self, question: str, max_candidates: int = None) -> List[str]:
        if max_candidates is None:
            max_candidates = self.max_candidates_per_operation

        seeds: List[str] = []

        # 1) NER → use strong anchor matcher
        names = self._extract_ner_spans(question)
        if names:
            seen: Set[str] = set()
            for nm in names:
                top_ids = self._resolve_anchor_exact(nm, question, limit=8)
                for nid in top_ids:
                    if nid not in seen:
                        seen.add(nid); seeds.append(nid)
                if len(seeds) >= max_candidates:
                    break

        # 2) If strong anchor is empty, fallback to lightweight name alignment + full sentence vector recall + lexical completion
        if not seeds:
            seeds = self._align_names_to_nodes(names, topk_each=8, cap_total=max_candidates) if names else []
            if not seeds:
                k_nodes = max(1, min(max_candidates, self.fallback_nodes_k))
                k_rels  = 1
                node_cand, _, _ = retrieve_candidates(
                    question=question,
                    node_pack=self.node_pack,
                    rel_pack=self.rel_pack,
                    k_nodes=k_nodes,
                    k_rels=k_rels,
                    sim_thr=0.30,
                    embedder=self.embedder,
                )
                seeds = node_cand[:max_candidates]
            if len(seeds) < max_candidates:
                more = self._fallback_lexical(
                    question,
                    getattr(self.node_pack, "texts", []),
                    getattr(self.node_pack, "ids", []),
                    topk=max(1, max_candidates - len(seeds))
                )
                for nid in more:
                    if nid not in seeds:
                        seeds.append(nid)
                    if len(seeds) >= max_candidates:
                        break

        # 3) Soft constraint reranking (only on existing seeds to avoid being biased by full sentence semantics)
        seeds = self._soft_rank_candidates(question, seeds, topk=max_candidates)
        # 4) (Optional) CE reranking
        seeds = self._cross_encoder_rerank(question, seeds, topk=max_candidates)

        return seeds[:max_candidates]

    def _get_candidate_relations(self, question: str, max_candidates: int = None) -> List[str]:
        if max_candidates is None:
            max_candidates = max(64, self.max_candidates_per_operation)
        k_nodes = 1
        k_rels  = max(1, max_candidates)
        _, rel_cand, _ = retrieve_candidates(
            question=question,
            node_pack=self.node_pack,
            rel_pack=self.rel_pack,
            k_nodes=k_nodes,
            k_rels=k_rels,
            sim_thr=0.30,
            embedder=self.embedder,
        )
        if not rel_cand:
            _, rel_cand, _ = retrieve_candidates(
                question=question,
                node_pack=self.node_pack,
                rel_pack=self.rel_pack,
                k_nodes=k_nodes,
                k_rels=k_rels,
                sim_thr=0.0,
                embedder=self.embedder,
            )
        if not rel_cand:
            rel_cand = self._fallback_lexical(
                question, getattr(self.rel_pack, "texts", []), getattr(self.rel_pack, "ids", []),
                topk=min(100, k_rels)
            )
        return rel_cand or []

    # ---------------- subgraph ops ---------------- #
    def _add_edge(self, subject: str, predicate: str, object_: str) -> bool:
        if (subject, predicate, object_) not in self.current_edges:
            self.current_edges.add((subject, predicate, object_))
            self.current_nodes.add(subject)
            self.current_nodes.add(object_)
            return True
        return False

    def _delete_edge(self, subject: str, predicate: str, object_: str) -> bool:
        if (subject, predicate, object_) in self.current_edges:
            self.current_edges.remove((subject, predicate, object_))
            self._cleanup_isolated_nodes()
            return True
        return False

    def _cleanup_isolated_nodes(self):
        connected_nodes = set()
        for s, _, o in self.current_edges:
            connected_nodes.add(s)
            connected_nodes.add(o)
        isolated = self.current_nodes - connected_nodes
        self.current_nodes -= isolated

    def _should_stop(self) -> bool:
        return (
            self.operation_count >= self.max_operations or
            len(self.current_nodes) >= 200 or
            len(self.current_edges) >= 400
        )

    # --- scoring helpers for edge picking --- #
    def _node_sim(self, qv: np.ndarray, nid: str) -> float:
        i = self._node_index_map.get(nid)
        if i is None:
            return 0.0
        return float(np.dot(qv, self.node_pack.vecs[i]))

    def _rel_sim(self, qv: np.ndarray, rid: str) -> float:
        i = self._rel_index_map.get(rid)
        if i is None:
            return 0.0
        return float(np.dot(qv, self.rel_pack.vecs[i]))

    def _best_new_edge_from(self, qv: np.ndarray, s: str, cand_rels: Optional[Set[str]]):
        best = None
        # Outgoing edges
        for r, t in self.adj.get(s, []):
            if cand_rels and r not in cand_rels:
                continue
            if (s, r, t) in self.current_edges:
                continue
            score = 0.7 * self._node_sim(qv, t) + 0.3 * self._rel_sim(qv, r)
            if (best is None) or (score > best[0]):
                best = (score, s, r, t)
        # Incoming edges
        for r, h2 in self.adj_in.get(s, []):
            if cand_rels and r not in cand_rels:
                continue
            if (h2, r, s) in self.current_edges:
                continue
            score = 0.7 * self._node_sim(qv, h2) + 0.3 * self._rel_sim(qv, r)
            if (best is None) or (score > best[0]):
                best = (score, h2, r, s)
        return best

    # ---- New: Actor → Movie priority edge selection (soft guidance, not hard constraint) ---- #
    def _prefer_starred_movie_edge(
        self,
        qv: np.ndarray,
        actor_id: str,
        cand_rels: Optional[Set[str]],
        rel_name: str = "starred_actors",
    ) -> Optional[Tuple[float, str, str, str]]:
        """
        Preferentially select edges like movie --[starred_actors]--> actor from actor's incoming edges.
        Returns (score, head_movie, rel, tail_actor) or None
        """
        best = None
        for r, h in self.adj_in.get(actor_id, []):
            if r != rel_name:
                continue
            if cand_rels and r not in cand_rels:
                continue
            if (h, r, actor_id) in self.current_edges:
                continue
            # Bias to amplify the node similarity of head(movie) a bit
            score = 0.8 * self._node_sim(qv, h) + 0.2 * self._rel_sim(qv, r)
            if (best is None) or (score > best[0]):
                best = (score, h, r, actor_id)
        return best

    # ---------------- forward (distribution sampling + deletion strategy + correct log_prob/entropy) ---------------- #
    def forward(self, obs: Any) -> Dict[str, Any]:
        device = self.device_str
        question = self._query_text_from_obs(obs)

        # Early stop
        if not question or self._should_stop():
            return {
                "action": {
                    "operation_type": "stop",
                    "subject": None, "predicate": None, "object": None,
                    "nodes": list(self.current_nodes),
                    "edges": list(self.current_edges),
                    "num_nodes": len(self.current_nodes),
                    "num_edges": len(self.current_edges),
                },
                "log_prob": torch.tensor(0.0, device=device),
                "entropy": torch.tensor(0.0, device=device),
                "num_nodes": len(self.current_nodes),
                "num_edges": len(self.current_edges),
            }

        # Encoding + backbone
        # Use sanitized question embedding to reduce direct name bias
        with torch.no_grad():
            q_clean = self._sanitize_question(question)
            q_vec = self.embedder.encode([q_clean])[0].astype("float32")
        qv_t = torch.from_numpy(q_vec).to(device=device, dtype=torch.float32)
        h = self.backbone(qv_t)

        # ① Three-class operation distribution
        logits = self.operation_head(h)  # [3]
        op_dist = D.Categorical(logits=logits)
        op_idx = op_dist.sample()        # Can also change to logits.argmax() for greedy
        op_map = {0: "add_edge", 1: "delete_edge", 2: "stop"}
        op_type = op_map[int(op_idx.item())]

        # ② Candidates (consistent with original logic)
        ents = self._get_candidate_entities(question)
        rels = self._get_candidate_relations(question)
        # Build and stash context pool once per step for downstream reranking (dict pool)
        try:
            pool = self.build_context_pool(question, max_pool=40, max_neigh_per_node=2)
        except Exception:
            pool = []
        cand_rel_set = set(rels) if rels else None

        sources: List[str] = []
        sources.extend(ents[:3])   # Top seeds after strong anchor/fallback + soft reranking
        for n in self.current_nodes:
            if n not in sources:
                sources.append(n)
        if not sources:
            sources = list(self._node_index_map.keys())[:10]

        # Map "possible actor node ids" from NER spans (no strict role recognition, just as prior)
        ner_names = self._extract_ner_spans(question)
        actor_ids: List[str] = []
        seen_ids = set()
        for nm in ner_names:
            for nid in self._resolve_anchor_exact(nm, question, limit=3):
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    actor_ids.append(nid)

        subject = predicate = object_ = None

        if op_type == "add_edge":
            # First try: for each "suspected actor", do movie --[starred_actors]--> actor priority edge
            best = None
            for aid in actor_ids[:3]:
                cand = self._prefer_starred_movie_edge(q_vec, aid, cand_rel_set, rel_name="starred_actors")
                if cand and (best is None or cand[0] > best[0]):
                    best = cand

            # If none selected above, fallback to general greedy
            if best is None:
                for s in sources:
                    cand = self._best_new_edge_from(q_vec, s, cand_rel_set)
                    if cand and (best is None or cand[0] > best[0]):
                        best = cand
                if best is None and cand_rel_set:
                    for s in sources:
                        cand = self._best_new_edge_from(q_vec, s, None)
                        if cand and (best is None or cand[0] > best[0]):
                            best = cand

            if best is not None:
                _, s2, r2, t2 = best
                if (s2, r2, t2) in self.edge_set:
                    if self._add_edge(s2, r2, t2):
                        self.operation_count += 1
                    subject, predicate, object_ = s2, r2, t2
            else:
                # If unable to add edge, degrade to STOP and align log_prob with STOP action
                op_type = "stop"
                op_idx = torch.tensor(2, device=device)

        elif op_type == "delete_edge":
            # Delete "least relevant to question" edge from current subgraph (simple heuristic)
            if self.current_edges:
                qv = q_vec / (np.linalg.norm(q_vec) + 1e-8)
                worst = None
                for (s, r, t) in self.current_edges:
                    # Score = 0.7*min(sim(s),sim(t)) + 0.3*sim(r)
                    sv = self._node_sim(qv, s); tv = self._node_sim(qv, t); rv = self._rel_sim(qv, r)
                    score = 0.7 * min(sv, tv) + 0.3 * rv
                    if worst is None or score < worst[0]:
                        worst = (score, s, r, t)
                _, s2, r2, t2 = worst
                self._delete_edge(s2, r2, t2)
                self.operation_count += 1
                subject, predicate, object_ = s2, r2, t2
            else:
                op_type = "stop"
                op_idx = torch.tensor(2, device=device)

        # stop: no change to subgraph

        # ③ log_prob / entropy aligned with PPO
        # Ensure log_prob corresponds to the final executed action index
        if not torch.is_tensor(op_idx):
            op_idx = torch.tensor(int(op_idx), device=device)
        log_prob = op_dist.log_prob(op_idx)
        entropy = op_dist.entropy()

        return {
            "action": {
                "operation_type": op_type,
                "subject": subject, "predicate": predicate, "object": object_,
                "nodes": list(self.current_nodes),
                "edges": list(self.current_edges),
                "num_nodes": len(self.current_nodes),
                "num_edges": len(self.current_edges),
            },
            "log_prob": log_prob.to(device),
            "entropy": entropy.to(device),
            "num_nodes": len(self.current_nodes),
            "num_edges": len(self.current_edges),
            # Downstream reranker expects dict pool; prefer latest_outputs cache if present
            "context_pool": (self.latest_outputs.get("context_pool", pool) if question else []),
        }

    # ---------------- RL helpers / compatibility ---------------- #
    def reset_subgraph(self):
        self.current_nodes.clear()
        self.current_edges.clear()
        self.operation_count = 0

    def get_current_subgraph(self) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        return self.current_nodes.copy(), self.current_edges.copy()

    def set_subgraph(self, nodes: Set[str], edges: Set[Tuple[str, str, str]]):
        self.current_nodes = set(nodes)
        self.current_edges = set(edges)
        self.operation_count = 0

    def log_prob_action(self, obs: Any, action: Dict[str, Any]) -> torch.Tensor:
        device = self.device_str
        question = self._query_text_from_obs(obs)
        if not question:
            return torch.tensor(0.0, device=device)
        with torch.no_grad():
            q_vec = self.embedder.encode([question])[0]
        qv_t = torch.from_numpy(q_vec).to(device=device, dtype=torch.float32)

        h = self.backbone(qv_t)
        operation_logits = self.operation_head(h)

        operation_map = {"add_edge": 0, "delete_edge": 1, "stop": 2}
        operation_idx = operation_map.get(action.get("operation_type", "stop"), 2)

        operation_dist = D.Categorical(logits=operation_logits)
        return operation_dist.log_prob(torch.tensor(operation_idx, device=device))

    def _reward_add_edge(self, obs: Dict[str, Any]) -> float:
        return 0.1

    def _reward_delete_edge(self, obs: Dict[str, Any]) -> float:
        return 0.05

    def _evaluate_final_subgraph(self, obs: Dict[str, Any]) -> float:
        num_nodes = len(self.current_nodes)
        num_edges = len(self.current_edges)
        if 5 <= num_nodes <= 50 and 3 <= num_edges <= 100:
            return 1.0
        elif num_nodes > 0 and num_edges > 0:
            return 0.5
        else:
            return -0.5

    def reset(self):
        self.current_nodes.clear()
        self.current_edges.clear()
        self.operation_count = 0
