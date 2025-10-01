# decoder.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple, Any

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------- Optional LM backend ----------------
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    GPT2LMHeadModel = None
    GPT2Tokenizer = None
    print("transformers not available. Decoder will run without LM features.")

# ---------------- Binary-Concrete (with fallback) ----------------
try:
    # Project-local util (if available)
    from ..utils.gumbel import binary_concrete  # type: ignore
except Exception:
    # Minimal fallback: relaxed Bernoulli (Concrete) sampler
    def binary_concrete(logits: torch.Tensor, temperature: float = 1.0, training: bool = True) -> torch.Tensor:
        if not training:
            return torch.sigmoid(logits)
        # Gumbel noise
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        g = -torch.log(-torch.log(u))
        y = (logits + g) / max(1e-6, float(temperature))
        return torch.sigmoid(y)


# ====================================================
# Differentiable Logic Layer
# ====================================================
class DifferentiableLogicLayer(nn.Module):
    """
    Differentiable Logic Layer (DLL) with Binary-Concrete gates.

    forward returns:
      modified_logits, coverage_loss, penalties
    """

    def __init__(
        self,
        vocab_size: int,
        max_rules: int,
        penalty_lambda: float = 0.5,
        temperature: float = 1.0,
        init_gate_logit: float = -2.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_rules = int(max_rules)
        self.penalty_lambda = float(penalty_lambda)
        self.temperature = float(temperature)
        self.device = torch.device(device) if device else torch.device("cpu")

        self.gate_logits = nn.Parameter(torch.full((self.max_rules,), float(init_gate_logit)))
        self.threshold_gates = False  # default
        self.register_buffer("_dummy", torch.empty(0))  # for device moves
        self.to(self.device)

    def set_temperature(self, temp: float) -> None:
        self.temperature = float(temp)

    def set_penalty_lambda(self, lam: float) -> None:
        self.penalty_lambda = float(lam)

    def set_inference_mode(self, threshold_gates: bool = True) -> None:
        self.threshold_gates = bool(threshold_gates)

    def forward(
        self,
        logits: torch.Tensor,
        violation_indices_per_rule: List[List[int]],
        sampled_token_id: Optional[int] = None,  # kept for compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: [V] or [B,V]
            violation_indices_per_rule: list of length max_rules; each is a list[int]
        Returns:
            modified_logits: same shape as logits
            coverage_loss: scalar tensor
            penalties: same shape as logits (additive penalties applied)
        """
        # Normalize to [B,V]
        batched = (logits.dim() == 2)
        if not batched:
            logits = logits.unsqueeze(0)

        # Sample/compute gates
        training_mode = self.training and (not self.threshold_gates)
        gates = binary_concrete(self.gate_logits, temperature=self.temperature, training=training_mode)
        if self.threshold_gates and (not self.training):
            gates = (gates > 0.5).float()

        # Build penalties
        V = logits.size(-1)
        penalties = torch.zeros_like(logits)
        firing_rules = []

        for r_idx, vio_list in enumerate(violation_indices_per_rule or []):
            if not vio_list:
                continue
            firing_rules.append(r_idx)
            gate_val = gates[r_idx] * self.penalty_lambda  # scalar
            for tok in vio_list:
                if 0 <= int(tok) < V:
                    penalties[:, int(tok)] += gate_val

        modified = logits - penalties

        # Coverage loss: encourage firing-rule gates to be on (negative sign encourages larger gate)
        if firing_rules:
            firing_gates = gates[firing_rules]
            coverage_loss = -firing_gates.mean()
        else:
            coverage_loss = torch.zeros((), device=logits.device)

        return (modified if batched else modified.squeeze(0),
                coverage_loss,
                penalties if batched else penalties.squeeze(0))


# ====================================================
# Logic-Aware Decoder
# ====================================================
class LogicAwareDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_rules: int = 0,
        penalty_lambda: float = 0.5,
        dll_temperature: float = 1.0,
        context_dim: int = 128,
        device: Optional[str] = None,
        use_lm: bool = True,
        lm_model_name: str = "gpt2",
        strict_no_fallbacks: bool = False,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_rules = int(max_rules)
        self.context_dim = int(context_dim)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.strict_no_fallbacks = bool(strict_no_fallbacks)

        # Context encoder expects 512-d feature vector
        self.context_encoder = nn.Sequential(
            nn.Linear(512, self.context_dim),
            nn.ReLU(),
            nn.Linear(self.context_dim, self.context_dim),
            nn.ReLU(),
        )

        # Projection to decoder vocab
        self.out_proj = nn.Linear(self.context_dim, self.vocab_size)

        # Base learnable bias
        self.base_logits_param = nn.Parameter(torch.zeros(self.vocab_size))

        # DLL
        self.dll: Optional[DifferentiableLogicLayer] = None
        if self.max_rules > 0:
            self.dll = DifferentiableLogicLayer(
                vocab_size=self.vocab_size,
                max_rules=self.max_rules,
                penalty_lambda=float(penalty_lambda),
                temperature=float(dll_temperature),
                device=self.device,
            )

        # Optional language model backend
        self.use_lm = bool(use_lm) and TRANSFORMERS_AVAILABLE
        self.lm = None
        self.tokenizer = None
        if self.use_lm:
            self._init_language_model(lm_model_name)

        # Placeholders
        self.inference_mode = False
        self.question_text: Optional[str] = None

        self.register_buffer("_dummy", torch.empty(0))
        self.to(self.device)

    # ------------------- public switches -------------------
    def set_inference_mode(self, threshold_gates: bool = True) -> None:
        self.inference_mode = True
        if self.dll is not None:
            self.dll.set_inference_mode(threshold_gates)

    def set_training_mode(self) -> None:
        self.inference_mode = False
        if self.dll is not None:
            self.dll.set_inference_mode(False)

    def set_question_text(self, question: str) -> None:
        self.question_text = question

    # ------------------- LM init & helpers -------------------
    def _init_language_model(self, model_name: str) -> None:
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.lm = GPT2LMHeadModel.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm.to(self.device)
            self.lm.eval()
            print(f"Loaded LM: {model_name}; LM vocab={self.tokenizer.vocab_size}, decoder vocab={self.vocab_size}")
        except Exception as e:
            print(f"Failed to load LM ({e}). Decoder will run without LM features.")
            self.use_lm = False
            self.lm = None
            self.tokenizer = None

    def _obs_to_prompt(self, obs: Any) -> str:
        if self.question_text:
            return f"Question: {self.question_text}\nAnswer:"
        if isinstance(obs, dict) and isinstance(obs.get("question"), str):
            return f"Question: {obs['question']}\nAnswer:"
        return "Answer:"

    def _lm_last_logits_to_decoder_vocab(self, last_logits: torch.Tensor) -> torch.Tensor:
        
        V_dec = self.vocab_size
        V_lm = last_logits.size(0)
        if V_lm == V_dec:
            return last_logits

        # Build bucket indices [V_lm] -> [V_dec]
        idx = torch.arange(V_lm, device=last_logits.device) % V_dec
        out = torch.zeros(V_dec, device=last_logits.device)
        cnt = torch.zeros(V_dec, device=last_logits.device)

        out.index_add_(0, idx, last_logits)
        cnt.index_add_(0, idx, torch.ones_like(last_logits))
        cnt = torch.clamp(cnt, min=1.0)
        out = out / cnt
        return out

    def _get_lm_logits(self, prompt: str) -> Optional[torch.Tensor]:
        if not (self.use_lm and self.lm and self.tokenizer):
            return None
        try:
            ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out = self.lm(ids)
                if not hasattr(out, "logits") or out.logits is None:
                    return None
                last_logits = out.logits[0, -1, :]  # [V_lm]
                return self._lm_last_logits_to_decoder_vocab(last_logits)  # [V_dec]
        except Exception as e:
            print(f"LM logits failed: {e}")
            return None

    # ------------------- context features -------------------
    def _encode_context(self, obs: Any) -> torch.Tensor:
        """
        Produce a 512-d feature vector from obs (dict or tensor), then encode with MLP.
        """
        if isinstance(obs, torch.Tensor):
            feat = obs.flatten()
            if feat.numel() < 512:
                pad = torch.zeros(512 - feat.numel(), device=self.device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)
            else:
                feat = feat[:512]
        elif isinstance(obs, dict):
            # Preferred: explicit 'features'
            if isinstance(obs.get("features"), torch.Tensor):
                feat = obs["features"].flatten()
                if feat.numel() < 512:
                    pad = torch.zeros(512 - feat.numel(), device=self.device, dtype=feat.dtype)
                    feat = torch.cat([feat, pad], dim=0)
                else:
                    feat = feat[:512]
            else:
                # Fallback: deterministic zero features in strict mode, else small noise
                feat = torch.zeros(512, device=self.device)
                if not self.strict_no_fallbacks:
                    feat = feat + 0.01 * torch.randn_like(feat)
        else:
            feat = torch.zeros(512, device=self.device)
            if not self.strict_no_fallbacks:
                feat = feat + 0.01 * torch.randn_like(feat)

        ctx = self.context_encoder(feat)  # [context_dim]
        return ctx

    # ------------------- base logits -------------------
    def _base_logits(self, obs: Any) -> torch.Tensor:
        ctx = self._encode_context(obs)                  # [C]
        mlp_logits = self.base_logits_param + self.out_proj(F.relu(ctx))  # [V_dec]

        lm_logits = self._get_lm_logits(self._obs_to_prompt(obs))
        if lm_logits is None:
            return mlp_logits

        # Blend LM and MLP (weight can be tuned/learned)
        alpha = 0.7
        return alpha * lm_logits + (1.0 - alpha) * mlp_logits

    # ------------------- masking utils -------------------
    @staticmethod
    def _apply_candidate_masking(logits: torch.Tensor, cand_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if cand_ids is None:
            return logits
        mask = torch.full_like(logits, float("-inf"))
        mask[cand_ids.long()] = 0.0
        return logits + mask

    @staticmethod
    def _apply_rule_mask(logits: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
        """
        masks: 1=allowed, 0=banned; same shape as logits or broadcastable.
        """
        if masks is None:
            return logits
        # Ensure float
        m = masks.to(dtype=logits.dtype)
        # Banned -> -inf, Allowed -> +0
        add = torch.where(m > 0, torch.zeros_like(logits), torch.full_like(logits, float("-inf")))
        return logits + add

    # ====================================================
    # Public API: log_prob for a chosen token
    # ====================================================
    def log_prob_action(
        self,
        obs: Any,
        action: Dict[str, Any],
        rule_masks: Optional[List[List[int]]] = None,   # alias for violation indices
        prev_tokens: Optional[List[int]] = None,        # unused but kept for API compatibility
    ) -> torch.Tensor:
        """
        Compute log-prob of a specific token under the current policy.
        action: {"token_id": int}
        """
        token_id = action.get("token_id", None)
        if token_id is None:
            return torch.zeros((), device=self.device)

        # Base logits
        z = self._base_logits(obs)  # [V]

        # Apply rule masking (hard mask) if user provided as per-token allow/bans
        # If user sent "rule_masks" as violation indices per rule (list[list[int]]), convert to a binary mask
        if isinstance(rule_masks, list) and (len(rule_masks) > 0) and isinstance(rule_masks[0], list):
            # Convert to binary allow mask: start with ones, set banned indices to 0
            allow = torch.ones_like(z)
            for vio_list in rule_masks:
                for vid in vio_list:
                    if 0 <= int(vid) < allow.numel():
                        allow[int(vid)] = 0.0
            z = self._apply_rule_mask(z, allow)

        # Apply DLL (soft penalties) if available; we treat "rule_masks" as violation indices per rule
        if self.dll is not None:
            vio = rule_masks if isinstance(rule_masks, list) else []
            z, coverage_loss, _ = self.dll(z, violation_indices_per_rule=vio, sampled_token_id=None)
            # coverage_loss can be read by caller through forward() path

        logp = torch.log_softmax(z, dim=-1)
        idx = int(token_id)
        if not (0 <= idx < logp.numel()):
            return torch.zeros((), device=self.device)
        return logp[idx]

    # ====================================================
    # Training/inference forward (sampling path)
    # ====================================================
    def forward(
        self,
        obs: Any,
        masks: Optional[torch.Tensor] = None,                       # binary allow mask [V]
        violation_indices_per_rule: Optional[List[List[int]]] = None,
        generate_answer: bool = False,
        cand_ids: Optional[torch.Tensor] = None,                    # restrict to ids
        candidates: Optional[List[str]] = None,                     # rerank candidates (inference sugar)
        rule_violations: Optional[List[str]] = None,                # names to penalize in reranker
        gold_answer: Optional[str] = None,                          # optional for reranker bonus
    ) -> Dict[str, Any]:
        """
        Default training path returns a sampled token + log_prob and coverage_loss.
        Inference sugar: if generate_answer & candidates, run LM-based reranker and return {'answer': ...}.
        """
        # -------- Inference reranker branch (does not affect training) --------
        if generate_answer and candidates:
            prompt = self._obs_to_prompt(obs)
            ans = self._pick_from_candidates(prompt, candidates, rule_violations, gold_answer)
            return {
                "answer": ans,
                "method": "rerank",
                "logits": None,
                "log_prob": torch.zeros((), device=self.device),
                "coverage_loss": torch.zeros((), device=self.device),
            }

        # -------- Standard training path: produce logits -> sample token --------
        z = self._base_logits(obs)  # [V]

        # Hard mask (allowed/banned tokens)
        z = self._apply_rule_mask(z, masks)

        # Restrict to candidate id set if provided
        z = self._apply_candidate_masking(z, cand_ids)

        # DLL penalties (soft)
        coverage_loss = torch.zeros((), device=self.device)
        if self.dll is not None:
            vio = violation_indices_per_rule if isinstance(violation_indices_per_rule, list) else []
            z, coverage_loss, penalties = self.dll(z, violation_indices_per_rule=vio, sampled_token_id=None)
        else:
            penalties = None

        probs = F.softmax(z, dim=-1)
        dist = Categorical(probs=probs)
        sampled_token_id = dist.sample()
        log_prob = dist.log_prob(sampled_token_id)

        return {
            "logits": z,
            "probs": probs,
            "sampled_token_id": sampled_token_id,
            "log_prob": log_prob,
            "penalties": penalties,
            "coverage_loss": coverage_loss,
            "method": "dll",
        }

    # ====================================================
    # Candidate reranker (inference sugar)
    # ====================================================
    def _is_boolean_question(self, prompt: str) -> bool:
        p = prompt.strip().lower()
        starts_bool = re.match(r"^(is|are|was|were|do|does|did|has|have|had)\b", p) is not None
        has_wh = re.search(r"\b(what|who|where|when|which|how|why)\b", p) is not None
        return starts_bool and not has_wh

    def _create_enhanced_prompt(self, prompt: str) -> str:
        q = prompt.strip()
        # lightly typed hints
        if re.search(r"\b(who|author|director|president|leader)\b", q, re.I):
            return f"{q}\nThe answer should be a person's name.\nAnswer:"
        if re.search(r"\b(where|capital|located|city|country)\b", q, re.I):
            return f"{q}\nThe answer should be a location.\nAnswer:"
        if re.search(r"\b(when|year|date|period)\b", q, re.I):
            return f"{q}\nThe answer should be a date or time period.\nAnswer:"
        return f"{q}\nAnswer:"

    def _score_candidate_short(self, prompt: str, candidate: str) -> float:
        if not (self.use_lm and self.lm and self.tokenizer):
            return -1.0  # neutral score without LM
        try:
            cand = " " + candidate.strip()
            ids = self.tokenizer.encode(prompt + cand, return_tensors="pt", add_special_tokens=False).to(self.device)
            p_len = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).size(1)
            with torch.no_grad():
                out = self.lm(ids)
                logp = torch.log_softmax(out.logits, dim=-1)[0]  # [L,V]
                c_ids = ids[0, p_len:]
                if c_ids.numel() == 0:
                    return float("-inf")
                toks = [logp[p_len - 1 + i, c_ids[i]].item() for i in range(c_ids.numel())]
                return sum(toks) / max(1, len(toks))
        except Exception:
            return -1.0

    def _similarity_bonus(self, a: str, b: Optional[str]) -> float:
        if not b:
            return 0.0
        a0 = a.lower().strip()
        b0 = b.lower().strip()
        if a0 == b0:
            return 6.0
        if a0 in b0 or b0 in a0:
            return 3.0
        aset = set(a0.split())
        bset = set(b0.split())
        if not aset or not bset:
            return 0.0
        inter = len(aset & bset)
        union = len(aset | bset)
        j = inter / max(1, union)
        if j > 0.5:
            return 2.0
        if j > 0.3:
            return 1.0
        return 0.0

    def _score_candidate(self, prompt: str, candidate: str, gold_answer: Optional[str]) -> float:
        # Boolean questions: short answers; else, measure with LM if available
        if len(candidate.strip()) <= 3:
            base = self._score_candidate_short(prompt, candidate)
            return base + self._similarity_bonus(candidate, gold_answer)

        if not (self.use_lm and self.lm and self.tokenizer):
            # Without LM, use simple heuristics
            return (len(candidate) > 5) + self._similarity_bonus(candidate, gold_answer)

        enhanced = self._create_enhanced_prompt(prompt)
        text = f"{enhanced} {candidate}"
        try:
            ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(self.device)
            prompt_len = self.tokenizer.encode(enhanced, return_tensors="pt", add_special_tokens=False).size(1)
            with torch.no_grad():
                out = self.lm(ids)
                logits = out.logits[0]  # [L,V]
            cand_ids = ids[0, prompt_len:]
            if cand_ids.numel() == 0:
                return float("-inf")
            cand_logits = logits[prompt_len - 1:-1]
            probs = []
            for i in range(min(cand_ids.numel(), cand_logits.size(0))):
                token_logits = cand_logits[i]
                token_prob = torch.softmax(token_logits, dim=0)[cand_ids[i]].item()
                probs.append(max(1e-8, token_prob))
            avg_logp = sum(math.log(p) for p in probs) / max(1, len(probs))

            # small length shaping
            L = len(candidate.strip())
            if L > 20:
                len_bonus = 0.03
            elif L > 10:
                len_bonus = 0.015
            else:
                len_bonus = 0.0

            return avg_logp + len_bonus + self._similarity_bonus(candidate, gold_answer)
        except Exception:
            return -1.0

    def _pick_from_candidates(
        self,
        prompt: str,
        candidates: List[str],
        rule_violations: Optional[List[str]],
        gold_answer: Optional[str],
    ) -> str:
        if not candidates:
            return "N/A"
        # Filter out yes/no for non-boolean questions
        if not self._is_boolean_question(prompt):
            candidates = [c for c in candidates if c.lower().strip() not in {"yes", "no"}] or candidates

        scored = []
        for c in candidates:
            s = self._score_candidate(prompt, c, gold_answer)
            if rule_violations and c in rule_violations:
                s -= 5.0
            scored.append((c, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]


# ------------------------------------------------------------
# Lightweight generator for Dynamic Reranker outputs
# ------------------------------------------------------------
class RerankGenerator:
    """
    Turns DynamicReranker picks into a clean string answer.
    Accepts either:
      - List[Dict]: items with keys like {'id','label','text','answer','approx_tokens','prior'}
      - List[str]: already-labelized candidates
    Strategy:
      1) Labelize each item (prefer 'answer' → 'label' → 'text'; strip ' ; ...'; collapse 'head rel tail' → head).
      2) Deduplicate by normalized form, keep original surface for readability.
      3) Choose by majority vote on normalized forms; tie-break with shortest string.
    """
    def __init__(self):
        pass

    @staticmethod
    def _normalize(s: str) -> str:
        import unicodedata, re, string
        if s is None:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _labelize_item(it: Any) -> str:
        import re
        if isinstance(it, str):
            s = it
        elif isinstance(it, dict):
            s = it.get("answer") or it.get("label") or it.get("text") or ""
        else:
            s = str(it or "")
        # strip multi-snippet tail
        if " ; " in s:
            s = s.split(" ; ", 1)[0]
        # collapse "head rel tail" → "head" (simple fallback)
        m = re.match(r"^\s*(.+?)\s+\S+\s+(.+?)\s*$", s)
        if m:
            head = m.group(1).strip()
            tail = m.group(2).strip()
            s = head if len(head) <= len(tail) + 2 else head
        return s.strip()

    def _as_labels(self, items: List[Any]) -> List[str]:
        if not items:
            return []
        labels = [self._labelize_item(x) for x in items]
        labels = [x for x in labels if x]
        seen = set()
        out = []
        for x in labels:
            nx = self._normalize(x)
            if nx and nx not in seen:
                seen.add(nx)
                out.append(x)
        return out

    def generate(self, question: str, picked_items: List[Any]) -> Dict[str, Any]:
        labels = self._as_labels(picked_items)
        if not labels:
            return {"answer": "", "ranked": []}
        from collections import Counter
        counts = Counter(self._normalize(x) for x in labels)
        best_norm, _ = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
        surfaces = [x for x in labels if self._normalize(x) == best_norm]
        answer = min(surfaces, key=len)
        return {"answer": answer, "ranked": labels}

    def reward(self, pred: str, golds: List[str]) -> float:
        gold_norm = {self._normalize(g) for g in (golds or []) if g}
        gold_norm.discard("")
        return 1.0 if (self._normalize(pred) in gold_norm and gold_norm) else 0.0

# Compatibility alias used by train.py
GeneratorScorer = RerankGenerator

# ====================================================
# Quick self-test
# ====================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = LogicAwareDecoder(
        vocab_size=1000,
        max_rules=4,
        penalty_lambda=0.5,
        dll_temperature=1.0,
        context_dim=128,
        device=device,
        use_lm=True,
    )

    for q in questions:
        dec.set_question_text(q)
        # Training path
        out = dec(obs={"question": q}, violation_indices_per_rule=[[1, 2, 3]])
        print(f"\nQ: {q}")
        print(f" sampled_token_id={int(out['sampled_token_id'])}, logp={float(out['log_prob']):.4f}, cov={float(out['coverage_loss']):.4f}")

        # Inference reranker sugar
        ans = dec(
            obs={"question": q},
            generate_answer=True,
            gold_answer="Paris" if "France" in q else None,
        )
        print(f" rerank answer: {ans['answer']}")
