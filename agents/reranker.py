# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import requests, time, random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------- Helper -------------------------------
def _safe_norm(x: np.ndarray, axis=None, keepdims=False, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    if axis is None:
        return x / (n + eps)
    return x / (n + eps)

# =============================== DynamicReranker (Retained) ===============================
class DynamicReranker(nn.Module):
    def __init__(self,
                 embedder,
                 max_steps: int = 15,
                 stop_penalty: float = -0.2,
                 token_budget: float = 512.0,
                 device: Optional[str] = None,
                 use_lm: bool = False,
                 lm_model_name: str = "gpt2",
                 w_lm: float = 0.5,
                 min_k: int = 2):
        super().__init__()
        self.embedder = embedder
        self.max_steps = int(max_steps)
        self.token_budget = float(token_budget)
        self.register_buffer("stop_penalty_base", torch.tensor(float(stop_penalty)))
        self._device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_k = max(0, int(min_k))

        self.w_vec = nn.Parameter(torch.tensor(1.0 - float(w_lm)))
        self.w_lm  = nn.Parameter(torch.tensor(float(w_lm)))
        self.logit_temp = nn.Parameter(torch.tensor(1.0))
        self.stop_bias  = nn.Parameter(torch.tensor(0.0))

        self.stop_ratio_coef = nn.Parameter(torch.tensor(0.3))
        self.stop_step_coef  = nn.Parameter(torch.tensor(0.1))

        self._lad = None
        self.use_lm_flag = bool(use_lm)
        if self.use_lm_flag:
            try:
                from .decoder import LogicAwareDecoder as _LAD
                self._lad = _LAD(vocab_size=2048, max_rules=0, device=str(self._device),
                                 use_lm=True, lm_model_name=lm_model_name)
            except Exception as e:
                logger.warning(f"DynamicReranker: tiny LM scoring disabled ({e})")
                self._lad = None

        self._reset_runtime()
        super().to(self._device)

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        if len(args) and isinstance(args[0], torch.device):
            self._device = args[0]
        elif "device" in kwargs and kwargs["device"] is not None:
            self._device = kwargs["device"]
        return ret

    def _reset_runtime(self):
        self.question = ""
        self.pool: List[Dict[str, Any]] = []
        self.remaining_idx: List[int] = []
        self.picked_idx: List[int] = []
        self.step_count = 0
        self.tok_mass = 0.0
        self._scores_full: Optional[torch.Tensor] = None
        self._vec_score: Optional[torch.Tensor] = None
        self._lm_score: Optional[torch.Tensor] = None

    def _ensure_pool_dicts(self, pool_any: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, it in enumerate(pool_any or []):
            if isinstance(it, dict) and "text" in it:
                approx_tokens = int(it.get("approx_tokens", max(1, round(len(str(it["text"]).split())/0.75))))
                prior = float(it.get("prior", 0.0))
                out.append({"id": it.get("id", i), "text": str(it["text"]), "approx_tokens": approx_tokens, "prior": prior})
            else:
                txt = str(it)
                approx_tokens = int(max(1, round(len(txt.split())/0.75)))
                out.append({"id": f"str:{i}", "text": txt, "approx_tokens": approx_tokens, "prior": 0.0})
        return out

    @staticmethod
    def _sanitize_question(q: str) -> str:
        try:
            import re
            return re.sub(r"\[(.+?)\]", " ", q or "").strip()
        except Exception:
            return q or ""

    def start_episode(self, question: str, pool: List[Dict[str, Any]]):
        self._reset_runtime()
        self.question = question or ""
        self.pool = self._ensure_pool_dicts(pool)
        self.remaining_idx = list(range(len(self.pool)))

        if self.embedder is not None:
            try:
                q_clean = self._sanitize_question(self.question)
                qv = self.embedder.encode([q_clean])[0].astype("float32")
                qn = _safe_norm(qv)
                cand_vecs = self.embedder.encode([it["text"] for it in self.pool]).astype("float32")
                cn = _safe_norm(cand_vecs, axis=1, keepdims=True)
                dot = (cn @ qn.reshape(-1, 1)).reshape(-1)
                base = np.array([float(it.get("prior", 0.0)) for it in self.pool], dtype=np.float32)
                vec_score = 0.7*dot + 0.3*base
            except Exception:
                vec_score = np.zeros(len(self.pool), dtype=np.float32)
        else:
            base = np.array([float(it.get("prior", 0.0)) for it in self.pool], dtype=np.float32)
            vec_score = base

        self._vec_score = torch.tensor(vec_score, dtype=torch.float32, device=self._device)

        if self._lad is not None and hasattr(self._lad, "_score_candidate"):
            lm_vals = []
            for it in self.pool:
                try:
                    s = float(self._lad._score_candidate(self.question, it["text"], None))
                except Exception:
                    s = 0.0
                lm_vals.append(s)
            self._lm_score = torch.tensor(lm_vals, dtype=torch.float32, device=self._device)
        else:
            self._lm_score = torch.zeros_like(self._vec_score, device=self._device)

        def z(t: torch.Tensor) -> torch.Tensor:
            if t.numel() <= 1:
                return torch.zeros_like(t)
            m  = torch.nan_to_num(t.mean(), nan=0.0)
            sd = torch.nan_to_num(t.std(unbiased=False), nan=0.0).clamp_min(1e-6)
            out = (t - m) / sd
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        alpha_vec = torch.sigmoid(self.w_vec)
        alpha_lm  = torch.sigmoid(self.w_lm)
        self._scores_full = z(self._vec_score) * alpha_vec + z(self._lm_score) * alpha_lm
        self._scores_full = torch.nan_to_num(self._scores_full, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, obs: Optional[dict] = None) -> Dict[str, Any]:
        scores = self._scores_full
        M = len(self.remaining_idx)

        if scores is None or M == 0:
            logits = torch.tensor([0.0], device=self._device)
            dist = D.Categorical(logits=logits)
            a = torch.tensor(0, device=self._device)
            logp = dist.log_prob(a); H = dist.entropy()
            self.step_count += 1
            return {"action": {"type": "stop", "picked": self.picked_idx.copy()},
                    "log_prob": logp, "entropy": H, "k": len(self.picked_idx)}

        cand_logits = scores[self.remaining_idx]
        cand_logits = torch.nan_to_num(cand_logits, nan=0.0, posinf=0.0, neginf=0.0)

        ratio = (self.tok_mass / max(1.0, self.token_budget)) if self.token_budget > 0 else 0.0
        step_norm = float(self.step_count) / max(1.0, float(self.max_steps))
        stop_logit = self.stop_penalty_base + self.stop_bias \
                     + self.stop_ratio_coef * ratio + self.stop_step_coef * step_norm
        stop_logit = torch.nan_to_num(stop_logit, nan=0.0)

        temp = torch.nan_to_num(self.logit_temp, nan=1.0).clamp_min(1e-3)
        logits = torch.cat([cand_logits, stop_logit.view(1)], dim=0) / temp

        force_stop = (
            (self.step_count >= self.max_steps)
            or (M == 0)
            or (self.token_budget > 0 and self.tok_mass >= self.token_budget)
        )

        if (len(self.picked_idx) < self.min_k) and not force_stop:
            logits[-1] = torch.tensor(float("-inf"), device=self._device)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        dist = D.Categorical(logits=logits)
        a = torch.tensor(M, device=self._device) if force_stop else dist.sample()
        logp = dist.log_prob(a); H = dist.entropy()

        if int(a.item()) == M:
            self.step_count += 1
            return {"action": {"type": "stop", "picked": self.picked_idx.copy()},
                    "log_prob": logp, "entropy": H, "k": len(self.picked_idx)}

        pick_local = int(a.item())
        pick_idx = self.remaining_idx[pick_local]
        self.picked_idx.append(pick_idx)
        del self.remaining_idx[pick_local]
        self.step_count += 1
        self.tok_mass += float(self.pool[pick_idx].get("approx_tokens", 0.0))
        return {"action": {"type": "pick", "index": pick_idx, "picked": self.picked_idx.copy()},
                "log_prob": logp, "entropy": H, "k": len(self.picked_idx)}

    def log_prob_action(self, obs: Any, action: Dict[str, Any],
                        rule_masks: Optional[List[List[int]]] = None,
                        prev_tokens: Optional[List[int]] = None) -> torch.Tensor:
        try:
            scores = self._scores_full
            pool = self.pool
            if scores is None or not pool:
                return torch.tensor(0.0, device=self._device)
            if isinstance(action, dict) and isinstance(action.get("ordered_indices"), list):
                ordered = [int(i) for i in action["ordered_indices"]]
            elif isinstance(prev_tokens, list):
                ordered = [int(i) for i in prev_tokens]
            else:
                return torch.tensor(0.0, device=self._device)

            N = len(pool)
            remain = list(range(N))
            tok_mass = 0.0
            total_lp = torch.tensor(0.0, device=self._device)
            picked_so_far: List[int] = []

            for i in ordered:
                cand_logits = scores[remain]
                cand_logits = torch.nan_to_num(cand_logits, nan=0.0, posinf=0.0, neginf=0.0)
                ratio = (tok_mass / max(1.0, self.token_budget)) if self.token_budget > 0 else 0.0
                step_norm = float(len(picked_so_far)) / max(1.0, float(self.max_steps))
                stop_logit = self.stop_penalty_base + self.stop_bias \
                             + self.stop_ratio_coef * ratio + self.stop_step_coef * step_norm
                stop_logit = torch.nan_to_num(stop_logit, nan=0.0)
                temp = torch.nan_to_num(self.logit_temp, nan=1.0).clamp_min(1e-3)
                logits = torch.cat([cand_logits, stop_logit.view(1)], dim=0) / temp
                if len(picked_so_far) < self.min_k:
                    logits[-1] = torch.tensor(float("-inf"), device=self._device)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                dist = D.Categorical(logits=logits)
                if 0 <= i < N and (i in remain):
                    local_idx = remain.index(i)
                    total_lp = total_lp + dist.log_prob(torch.tensor(local_idx, device=self._device))
                    picked_so_far.append(i)
                    tok_mass += float(pool[i].get("approx_tokens", 0.0))
                    remain.pop(local_idx)
                else:
                    total_lp = total_lp + dist.log_prob(torch.tensor(len(remain), device=self._device))
                    break

            cand_logits = scores[remain]
            cand_logits = torch.nan_to_num(cand_logits, nan=0.0, posinf=0.0, neginf=0.0)
            ratio = (tok_mass / max(1.0, self.token_budget)) if self.token_budget > 0 else 0.0
            step_norm = float(len(picked_so_far)) / max(1.0, self.max_steps)
            stop_logit = self.stop_penalty_base + self.stop_bias \
                         + self.stop_ratio_coef * ratio + self.stop_step_coef * step_norm
            stop_logit = torch.nan_to_num(stop_logit, nan=0.0)
            temp = torch.nan_to_num(self.logit_temp, nan=1.0).clamp_min(1e-3)
            logits = torch.cat([cand_logits, stop_logit.view(1)], dim=0) / temp
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            dist = D.Categorical(logits=logits)
            total_lp = total_lp + dist.log_prob(torch.tensor(len(remain), device=self._device))
            return total_lp.reshape(())
        except Exception:
            return torch.tensor(0.0, device=self._device)

    def finalize(self) -> List[Dict[str, Any]]:
        return [self.pool[i] for i in self.picked_idx if 0 <= i < len(self.pool)]

# =============================== API Reranker (New) ===============================
SILICONFLOW_BASE = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_KEY  = os.getenv("SILICONFLOW_API_KEY", "")

class APIReranker(nn.Module):
    """
    Uses SiliconFlow /v1/rerank to let LLM rerank candidates in batch, returns indices order:
      - During training forward() follows deterministic "pick by pre-ranking order + conditional STOP"
      - Retains min_k / token_budget / max_steps constraints
      - Falls back to embedder cosine ranking or prior ranking if API fails

    Environment variables (recommended):
      SILICONFLOW_API_KEY   - Required
      SILICONFLOW_BASE_URL  - Optional, defaults to https://api.siliconflow.cn
    """

    def __init__(self,
                 model: str = "BAAI/bge-reranker-v2-m3",
                 token_budget: float = 512.0,
                 max_steps: int = 15,
                 stop_penalty: float = -0.2,     # For record only; this class forward is deterministic, doesn't use logit
                 device: Optional[str] = None,
                 min_k: int = 2,
                 request_timeout: float = 60.0,
                 min_interval_sec: float = 1.2,
                 embedder_fallback=None):
        super().__init__()
        self.model = model
        self.token_budget = float(token_budget)
        self.max_steps = int(max_steps)
        self.min_k = max(0, int(min_k))
        self.register_buffer("stop_penalty_base", torch.tensor(float(stop_penalty)))

        self._device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # HTTP session (disable system proxy to avoid accidental proxy in container)
        self._sess = requests.Session()
        self._sess.trust_env = False
        
        keya = (SILICONFLOW_KEY or os.getenv("SILICONFLOW_API_KEY", "")).strip()
        roota = (SILICONFLOW_BASE or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn")).rstrip("/")
        self.url = f"{roota}/rerank"
        self.key = keya
        self.timeout = float(request_timeout)
        self.min_interval = float(min_interval_sec)
        self._last_ts = 0.0

        self.embedder_fallback = embedder_fallback  # Optional fallback embedder

        self._reset_runtime()
        super().to(self._device)

    # ---------- Runtime State ----------
    def _reset_runtime(self):
        self.question: str = ""
        self.pool: List[Dict[str, Any]] = []
        self.remaining_idx: List[int] = []
        self.picked_idx: List[int] = []
        self.step_count: int = 0
        self.tok_mass: float = 0.0
        self._order: List[int] = []  # Global order obtained from API or fallback ranking

    def _ensure_pool_dicts(self, pool_any: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, it in enumerate(pool_any or []):
            if isinstance(it, dict) and "text" in it:
                txt = str(it["text"])
                approx_tokens = int(it.get("approx_tokens", max(1, round(len(txt.split()) / 0.75))))
                prior = float(it.get("prior", 0.0))
                out.append({"id": it.get("id", i), "text": txt, "approx_tokens": approx_tokens, "prior": prior})
            else:
                txt = str(it)
                approx_tokens = int(max(1, round(len(txt.split()) / 0.75)))
                out.append({"id": f"str:{i}", "text": txt, "approx_tokens": approx_tokens, "prior": 0.0})
        return out

    def _throttle(self):
        now = time.monotonic()
        wait = self.min_interval - (now - self._last_ts)
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.12))
        self._last_ts = time.monotonic()

    # ---------- API Call ----------
    def _rank_with_api(self, question: str, cands: List[str]) -> List[int]:
        if not self.key:
            raise RuntimeError("APIReranker: missing SILICONFLOW_API_KEY")

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        # Official format: model/query/documents/top_n
        payload = {
            "model": self.model,
            "query": question,
            "documents": cands,  # Supports dict; more stable
            # "top_n": len(cands),
        }

        self._throttle()
        r = self._sess.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            proxies={"http": None, "https": None},
        )
        # Avoid latin-1 errors
        r.encoding = "utf-8"

        if r.status_code != 200:
            # Print server JSON error (if any)
            try:
                err = r.json()
            except Exception:
                err = r.text[:500]
            raise RuntimeError(f"APIReranker HTTP {r.status_code}: {err}")

        data = r.json()
        # Common return structure: {"data":[{"index":i,"relevance_score":x}, ...]}
        results = data.get("data") or data.get("results") or data.get("scores")
        if not isinstance(results, list):
            raise RuntimeError(f"unexpected response shape: {data}")

        scored: List[tuple[int, float]] = []
        for obj in results:
            try:
                idx = int(obj.get("index"))
                sc = float(
                    obj.get("relevance_score", obj.get("score", obj.get("relevance", 0.0)))
                )
                scored.append((idx, sc))
            except Exception:
                continue

        if not scored:
            raise RuntimeError("empty_scores")

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        order_raw = [i for (i, _) in scored]

        # Filter invalid indices + deduplicate + complete
        seen, order = set(), []
        for i in order_raw:
            if 0 <= i < len(cands) and i not in seen:
                seen.add(i)
                order.append(i)
        if len(order) < len(cands):
            for i in range(len(cands)):
                if i not in seen:
                    order.append(i)

        return order

    # ---------- Fallback Ranking ----------
    def _fallback_order(self, question: str, pool: List[Dict[str, Any]]) -> List[int]:
        # 1) embedder cosine similarity
        try:
            if self.embedder_fallback is not None and len(pool) > 0:
                import re as _re
                q_clean = _re.sub(r"\[(.+?)\]", " ", question or "").strip()
                qv = self.embedder_fallback.encode([q_clean])[0].astype("float32")
                qn = _safe_norm(qv)
                cand_mat = self.embedder_fallback.encode([it["text"] for it in pool]).astype("float32")
                cn = _safe_norm(cand_mat, axis=1, keepdims=True)
                cos = (cn @ qn.reshape(-1, 1)).reshape(-1)
                order = list(np.argsort(-cos))
                return order
        except Exception:
            pass
        # 2) Sort by prior descending
        try:
            order = list(sorted(range(len(pool)),
                                key=lambda i: float(pool[i].get("prior", 0.0)),
                                reverse=True))
            if order:
                return order
        except Exception:
            pass
        # 3) Original order
        return list(range(len(pool)))

    # ========== 统一对外 API ==========
    def start_episode(self, question: str, pool: List[Dict[str, Any]]):
        self._reset_runtime()
        self.question = question or ""
        self.pool = self._ensure_pool_dicts(pool)
        N = len(self.pool)
        if N == 0:
            self.remaining_idx = []
            self._order = []
            return

        try:
            cand_texts = [it["text"] for it in self.pool]
            self._order = self._rank_with_api(self.question, cand_texts)
            if not self._order or len(self._order) != N:
                raise RuntimeError("empty/bad order")
        except Exception as e:
            logger.warning(f"APIReranker: API ranking failed, fallback. ({e})")
            self._order = self._fallback_order(self.question, self.pool)

        # Pre-ranking order -> initial remaining
        self.remaining_idx = self._order.copy()

    def forward(self, obs: Optional[dict] = None) -> Dict[str, Any]:
        M = len(self.remaining_idx)

        # No candidates => STOP
        if M == 0:
            self.step_count += 1
            return {
                "action": {"type": "stop", "picked": self.picked_idx.copy()},
                "log_prob": torch.tensor(0.0, device=self._device),
                "entropy": torch.tensor(0.0, device=self._device),
                "k": len(self.picked_idx),
            }

        # Whether to force stop
        force_stop = (
            (self.step_count >= self.max_steps)
            or (self.token_budget > 0 and self.tok_mass >= self.token_budget)
        )
        if (len(self.picked_idx) >= self.min_k) and force_stop:
            self.step_count += 1
            return {
                "action": {"type": "stop", "picked": self.picked_idx.copy()},
                "log_prob": torch.tensor(0.0, device=self._device),
                "entropy": torch.tensor(0.0, device=self._device),
                "k": len(self.picked_idx),
            }

        # Haven't reached min_k or haven't triggered force stop => take next in pre-ranking order
        pick_idx = self.remaining_idx.pop(0)
        self.picked_idx.append(pick_idx)
        self.step_count += 1
        self.tok_mass += float(self.pool[pick_idx].get("approx_tokens", 0.0))

        return {
            "action": {"type": "pick", "index": pick_idx, "picked": self.picked_idx.copy()},
            "log_prob": torch.tensor(0.0, device=self._device),
            "entropy": torch.tensor(0.0, device=self._device),
            "k": len(self.picked_idx),
        }

    def log_prob_action(self, obs: Any, action: Dict[str, Any],
                        rule_masks: Optional[List[List[int]]] = None,
                        prev_tokens: Optional[List[int]] = None) -> torch.Tensor:
        # Deterministic strategy, no gradient source during training -> return 0
        return torch.tensor(0.0, device=self._device)

    def finalize(self) -> List[Dict[str, Any]]:
        return [self.pool[i] for i in self.picked_idx if 0 <= i < len(self.pool)]

# =============================== Pure LLM Reranker (Old wrapper, retained for compatibility) ===============================
try:
    from .llm_reranker_agent import LLMRerankerAgent as _LLM_INTERNAL
except Exception:
    _LLM_INTERNAL = None

class LLMRerankerAgent(nn.Module):
    def __init__(self, lm_model_name: str = "gpt2", token_budget: float = 512.0,
                 max_steps: int = 15, stop_temp: float = 1.0,
                 stop_penalty: float = 0.0, device: Optional[str] = None):
        super().__init__()
        self._device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_temp = float(stop_temp)
        self.stub = nn.Linear(1, 1, bias=False)
        if _LLM_INTERNAL is not None:
            self._impl = _LLM_INTERNAL(lm_model_name=lm_model_name, token_budget=token_budget,
                                       max_steps=max_steps, stop_temp=stop_temp,
                                       stop_penalty=stop_penalty, device=device)
            self._fallback = None
        else:
            self._impl = None
            self._fallback = DynamicReranker(embedder=None, token_budget=token_budget, max_steps=max_steps,
                                             stop_penalty=stop_penalty, device=device)
        super().to(self._device)

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        if len(args) and isinstance(args[0], torch.device):
            self._device = args[0]
        elif "device" in kwargs and kwargs["device"] is not None:
            self._device = kwargs["device"]
        return ret

    def start_episode(self, question: str, pool: List[Dict[str, Any]]):
        if self._impl is not None: return self._impl.start_episode(question, pool)
        return self._fallback.start_episode(question, pool)
    def forward(self, obs: Optional[dict] = None):
        _ = self.stub.weight * 1.0
        if self._impl is not None: return self._impl.forward(obs)
        return self._fallback.forward(obs)
    def log_prob_action(self, obs, action, rule_masks=None, prev_tokens=None):
        if self._impl is not None and hasattr(self._impl, "log_prob_action"):
            return self._impl.log_prob_action(obs, action, rule_masks, prev_tokens)
        return self._fallback.log_prob_action(obs, action, rule_masks, prev_tokens)
    def finalize(self):
        if self._impl is not None and hasattr(self._impl, "finalize"):
            return self._impl.finalize()
        return self._fallback.finalize()

# =============================== Groq Answerer (Retained) ===============================
GROQ_BASE_URL_INLINE = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
GROQ_API_KEY_INLINE  = os.getenv("SILICONFLOW_API_KEY", "")

class BaseAnswerer:
    def answer(self, question: str, contexts: List[str]) -> str:
        raise NotImplementedError

class GroqAnswerer(BaseAnswerer):
    _SYS = (
        "You will be given a user question and a list CANDIDATES (one item per line).\n"
        "Rules:\n"
        "• You MUST pick exactly one item from CANDIDATES as the final answer.\n"
        "• If several look plausible, pick the single most specific and relevant title/name.\n"
        "• You are FORBIDDEN to output anything not exactly present in CANDIDATES.\n"
        "Output: return ONLY the chosen candidate text. No explanations. No punctuation."
    )
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "Qwen/Qwen3-32B",
                 base_url: Optional[str] = None,
                 request_timeout: float = 120.0,
                 min_interval_sec: float = 2.5,
                 enable_thinking: bool = False):
        key = (api_key or GROQ_API_KEY_INLINE or os.getenv("SILICONFLOW_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("GroqAnswerer: missing API key. Set SILICONFLOW_API_KEY environment variable or pass api_key parameter.")
        root = (base_url or GROQ_BASE_URL_INLINE or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
        self.url = f"{root}/chat/completions"
        self.model = model
        self.timeout = float(request_timeout)
        self.api_key = key
        self.min_interval = float(min_interval_sec); self._last_ts = 0.0
        self._sess = requests.Session(); self._sess.trust_env = False
        self.enable_thinking = bool(enable_thinking)

    def _throttle(self):
        now = time.monotonic()
        wait = self.min_interval - (now - self._last_ts)
        if wait > 0: time.sleep(wait + random.uniform(0, 0.15))
        self._last_ts = time.monotonic()

    def answer(self, question: str, contexts: List[str]) -> str:
        self._throttle()
        cand_text = "\n".join(f"- {c}" for c in contexts[:32])
        user = (
            f"Question:\n{question}\n\n"
            f"CANDIDATES:\n{cand_text}\n\n"
            "Pick exactly one item from CANDIDATES."
        )
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._SYS},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "stream": False,
            "max_tokens": 1024,
            "enable_thinking": self.enable_thinking,
        }
        try:
            r = self._sess.post(self.url, headers=headers, json=payload, timeout=self.timeout, proxies={"http": None, "https": None})
        except Exception as e:
            logger.warning(f"GroqAnswerer request error: {e}")
            return ""
        if r.status_code != 200:
            logger.warning(f"GroqAnswerer HTTP {r.status_code}: {r.text[:500]}")
            return ""
        try:
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            return " ".join(text.split())[:128]
        except Exception:
            logger.warning(f"GroqAnswerer: unexpected response: {r.text[:500]}")
            return ""

def build_answerer(provider: str = "groq", **kw) -> BaseAnswerer:
    return GroqAnswerer(enable_thinking=False, request_timeout=240.0, **kw)

__all__ = [
    "build_answerer", "GroqAnswerer", "BaseAnswerer",
    "DynamicReranker", "LLMRerankerAgent", "APIReranker",
]

# =============================== Factories ===============================
def build_reranker(use_llm_reranker: bool,
                   embedder=None,
                   token_budget: float = 512.0,
                   max_steps: int = 15,
                   stop_penalty: float = -0.2,
                   device: Optional[str] = None,
                   rerank_lm_name: str = "gpt2",
                   w_lm: float = 0.5,
                   use_lm_in_dynamic: bool = False,
                   min_k: int = 2,
                   use_llm_reranker_api: bool = False,
                   llm_reranker_model: Optional[str] = None):
    """
    - use_llm_reranker_api=True -> APIReranker (Recommended)
    - use_llm_reranker=True -> Old LLMRerankerAgent (if exists)
    - Otherwise -> DynamicReranker
    """
    if use_llm_reranker_api:
        return APIReranker(
            model=(llm_reranker_model or rerank_lm_name),
            token_budget=token_budget,
            max_steps=max_steps,
            stop_penalty=stop_penalty,
            device=device,
            min_k=int(min_k),
            embedder_fallback=embedder,
        )
    if use_llm_reranker:
        return LLMRerankerAgent(
            lm_model_name=rerank_lm_name,
            token_budget=token_budget,
            max_steps=max_steps,
            stop_temp=1.0,
            stop_penalty=stop_penalty,
            device=device,
        )
    return DynamicReranker(
        embedder=embedder,
        max_steps=max_steps,
        stop_penalty=stop_penalty,
        token_budget=token_budget,
        device=device,
        use_lm=bool(use_lm_in_dynamic),
        lm_model_name=rerank_lm_name,
        w_lm=float(w_lm),
        min_k=int(min_k),
    )

# =============================== One-shot helper (Retained) ===============================
def rerank_and_answer(question: str,
                      pool: List[Dict[str, Any]],
                      reranker,
                      answerer: Optional[BaseAnswerer] = None,
                      k_ctx: int = 6) -> Tuple[List[int], List[str], str]:
    reranker.start_episode(question, pool)
    steps = []
    max_steps = getattr(reranker, "max_steps", 15)
    for _ in range(int(max_steps) + 1):
        o = reranker.forward({})
        steps.append(o)
        if o["action"]["type"] == "stop":
            break
    picked_idx = steps[-1]["action"].get("picked", []) if steps else []
    picked_texts = []
    for i in picked_idx:
        try:
            it = pool[i] if isinstance(pool[i], dict) else {"text": str(pool[i])}
            picked_texts.append(str(it.get("text", "")))
        except Exception:
            picked_texts.append("")
    final = ""
    if answerer is not None and picked_texts:
        final = answerer.answer(question, picked_texts[:max(1, int(k_ctx))]) or ""
    return picked_idx, picked_texts, final
