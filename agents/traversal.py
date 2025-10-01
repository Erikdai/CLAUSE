# traversal.py
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import hashlib
import torch
import torch.nn as nn
from torch.distributions import Categorical


PathElem = Union[str, Tuple[str, str, str]]


class EnhancedTraversalAgent(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        vocab_buckets: int = 65536,
        hidden: int = 128,
        max_hops: int = 4,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_buckets = int(vocab_buckets)
        self.max_hops = int(max_hops)
        self._device = device or "cpu"
        self._path: List[PathElem] = []

        # Hash embedding for string tokens
        self.hash_emb = nn.Embedding(self.vocab_buckets, self.emb_dim)
        nn.init.normal_(self.hash_emb.weight, std=0.1)

        # Termination head (stop vs continue)
        self.termination_head = nn.Sequential(
            nn.Linear(self.emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # [stop, continue]
        )

        # Candidate selection head (if continue)
        self.candidate_head = nn.Sequential(
            nn.Linear(self.emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # score each candidate
        )
        # Ensure module parameters reside on the requested device (CUDA-safe).
        self.to(self._device)

    def _hash_token(self, token: str) -> int:
        """Convert string to hash bucket index."""
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vocab_buckets

    def _pool_embeddings(self, tokens: List[str]) -> torch.Tensor:
        """Convert list of tokens to pooled embedding."""
        if not tokens:
            return torch.zeros(self.emb_dim, device=self._device)
        indices = [self._hash_token(t) for t in tokens]
        indices_tensor = torch.tensor(indices, device=self._device)
        embeddings = self.hash_emb(indices_tensor)  # [N, emb_dim]
        return embeddings.mean(dim=0)  # [emb_dim]

    def _maybe_reset_path(self, obs: Any) -> None:
        """
        Reset internal path buffer at episode start or on explicit reset.
        """
        should_reset = False
        if isinstance(obs, dict):
            if obs.get("reset") is True:
                should_reset = True
            t = obs.get("t", obs.get("step", None))
            if isinstance(t, int) and t <= 0:
                should_reset = True
        if should_reset:
            self._path = []

    # Optional: textualize a path (nodes/edges) into a short snippet for reranking
    def textualize_path(self, path: List[PathElem]) -> str:
        parts: List[str] = []
        for elem in path:
            if isinstance(elem, tuple) and len(elem) == 3:
                h, r, t = elem
                parts.append(f"{h} {r} {t}")
            elif isinstance(elem, str):
                parts.append(elem)
        return " ; ".join(parts)

    # -------------------------- forward -------------------------- #

    def forward(self, obs: Any) -> Dict[str, Any]:
        """
        Decide to STOP or CONTINUE; if continue, choose a candidate (node/edge).

        Returns: dict with keys `action`, `log_prob`, `entropy`, `path`, `path_len`.
        """
        device = self._device
        self._maybe_reset_path(obs)

        # Extract context from observation
        question = ""
        current_node = ""
        candidates = []
        path = []

        if isinstance(obs, dict):
            question = obs.get("question", "")
            current_node = obs.get("current_node", "")
            candidates = obs.get("neighbors", obs.get("candidates", obs.get("candidate_nodes", obs.get("candidate_edges", []))))
            path = obs.get("path", self._path)

        # Update internal path
        self._path = path

        # Build context tokens
        context_tokens = []
        if question:
            context_tokens.extend(question.split())
        if current_node:
            context_tokens.append(current_node)
        if path:
            for elem in path[-3:]:  # Last 3 path elements
                if isinstance(elem, str):
                    context_tokens.append(elem)
                elif isinstance(elem, tuple) and len(elem) == 3:
                    h, r, t = elem
                    context_tokens.extend([h, r, t])

        # Get context embedding
        context_emb = self._pool_embeddings(context_tokens)

        # Termination decision
        term_logits = self.termination_head(context_emb)
        term_dist = Categorical(logits=term_logits)
        term_action = term_dist.sample()
        term_log_prob = term_dist.log_prob(term_action)
        term_entropy = term_dist.entropy()

        stop = bool(term_action.item() == 0)
        # Forced STOP guards (keep log-prob aligned with executed STOP)
        forced_stop = False
        if len(self._path) >= self.max_hops:
            forced_stop = True

        # If stopping by sample or by force, return STOP with correct log-prob for STOP (index 0)
        if stop or forced_stop:
            term_log_prob = term_dist.log_prob(torch.tensor(0, device=self._device))
            return {
                "action": {"stop": True, "next_edge": None, "next_node": None, "path": self._path},
                "log_prob": term_log_prob,
                "entropy": term_entropy,
                "path": self._path,
                "path_len": len(self._path),
            }

        # Continue: select candidate
        if not candidates:
            # No candidates available -> force STOP, align log-prob with STOP
            term_log_prob = term_dist.log_prob(torch.tensor(0, device=self._device))
            return {
                "action": {"stop": True, "next_edge": None, "next_node": None, "path": self._path},
                "log_prob": term_log_prob,
                "entropy": term_entropy,
                "path": self._path,
                "path_len": len(self._path),
            }

        # Score candidates
        candidate_scores = []
        for cand in candidates:
            if isinstance(cand, str):
                cand_tokens = [cand]
            elif isinstance(cand, tuple) and len(cand) == 3:
                h, r, t = cand
                cand_tokens = [h, r, t]
            elif isinstance(cand, dict):
                # Extract text from dict format
                cand_tokens = []
                for key in ["head", "relation", "tail", "text", "name"]:
                    if key in cand and isinstance(cand[key], str):
                        cand_tokens.extend(cand[key].split())
            else:
                cand_tokens = [str(cand)]

            cand_emb = self._pool_embeddings(cand_tokens)
            # Combine context and candidate
            combined_emb = context_emb + cand_emb
            score = self.candidate_head(combined_emb)
            candidate_scores.append(score)

        # Select best candidate
        if candidate_scores:
            candidate_tensor = torch.stack(candidate_scores).squeeze(-1)
            cand_dist = Categorical(logits=candidate_tensor)
            cand_action = cand_dist.sample()
            cand_log_prob = cand_dist.log_prob(cand_action)
            cand_entropy = cand_dist.entropy()

            selected_cand = candidates[cand_action.item()]
            next_edge = None
            next_node = None

            if isinstance(selected_cand, tuple) and len(selected_cand) == 3:
                next_edge = selected_cand
                next_node = selected_cand[2]  # tail node
            elif isinstance(selected_cand, str):
                next_node = selected_cand
            elif isinstance(selected_cand, dict):
                if "tail" in selected_cand:
                    next_node = selected_cand["tail"]
                if all(k in selected_cand for k in ["head", "relation", "tail"]):
                    next_edge = (selected_cand["head"], selected_cand["relation"], selected_cand["tail"])

            # Update path
            if next_edge:
                self._path.append(next_edge)
            elif next_node:
                self._path.append(next_node)

            total_log_prob = term_log_prob + cand_log_prob
            total_entropy = term_entropy + cand_entropy

            return {
                "action": {
                    "stop": False,
                    "next_edge": next_edge,
                    "next_node": next_node,
                    "path": self._path,
                },
                "log_prob": total_log_prob,
                "entropy": total_entropy,
                "path": self._path,
                "path_len": len(self._path),
            }
        else:
            # No valid candidates -> force STOP, align log-prob with STOP
            term_log_prob = term_dist.log_prob(torch.tensor(0, device=self._device))
            return {
                "action": {"stop": True, "next_edge": None, "next_node": None, "path": self._path},
                "log_prob": term_log_prob,
                "entropy": term_entropy,
                "path": self._path,
                "path_len": len(self._path),
            }

    def log_prob_action(self, obs: Any, action: Dict[str, Any]) -> torch.Tensor:
        """
        Compute log probability of taking a specific action given observation.
        """
        device = self._device
        self._maybe_reset_path(obs)

        # Extract context from observation
        question = ""
        current_node = ""
        candidates = []
        path = []

        if isinstance(obs, dict):
            question = obs.get("question", "")
            current_node = obs.get("current_node", "")
            candidates = obs.get("neighbors", obs.get("candidates", obs.get("candidate_nodes", obs.get("candidate_edges", []))))
            path = obs.get("path", self._path)

        # Build context tokens
        context_tokens = []
        if question:
            context_tokens.extend(question.split())
        if current_node:
            context_tokens.append(current_node)
        if path:
            for elem in path[-3:]:  # Last 3 path elements
                if isinstance(elem, str):
                    context_tokens.append(elem)
                elif isinstance(elem, tuple) and len(elem) == 3:
                    h, r, t = elem
                    context_tokens.extend([h, r, t])

        # Get context embedding
        context_emb = self._pool_embeddings(context_tokens)

        # Termination decision
        term_logits = self.termination_head(context_emb)
        term_dist = Categorical(logits=term_logits)
        
        # Check if action is stop
        if action.get("stop", False):
            return term_dist.log_prob(torch.tensor(0, device=device))  # 0 = stop
        
        # Continue requested, but forward() would have forced STOP if no candidates or max hops exceeded
        if not candidates or len(self._path) >= self.max_hops:
            return term_dist.log_prob(torch.tensor(0, device=device))  # 0 = stop
        
        # Score candidates
        candidate_scores = []
        for cand in candidates:
            if isinstance(cand, str):
                cand_tokens = [cand]
            elif isinstance(cand, tuple) and len(cand) == 3:
                h, r, t = cand
                cand_tokens = [h, r, t]
            elif isinstance(cand, dict):
                cand_tokens = []
                for key in ["head", "relation", "tail", "text", "name"]:
                    if key in cand and isinstance(cand[key], str):
                        cand_tokens.extend(cand[key].split())
            else:
                cand_tokens = [str(cand)]

            cand_emb = self._pool_embeddings(cand_tokens)
            combined_emb = context_emb + cand_emb
            score = self.candidate_head(combined_emb)
            candidate_scores.append(score)

        if candidate_scores:
            candidate_tensor = torch.stack(candidate_scores).squeeze(-1)
            cand_dist = Categorical(logits=candidate_tensor)
            
            # Find which candidate was selected
            selected_cand = action.get("next_edge") or action.get("next_node")
            cand_idx = 0
            if selected_cand:
                for i, cand in enumerate(candidates):
                    if (isinstance(cand, tuple) and len(cand) == 3 and cand == selected_cand) or \
                       (isinstance(cand, str) and cand == selected_cand) or \
                       (isinstance(cand, dict) and cand.get("tail") == selected_cand):
                        cand_idx = i
                        break
            
            term_log_prob = term_dist.log_prob(torch.tensor(1, device=device))  # 1 = continue
            cand_log_prob = cand_dist.log_prob(torch.tensor(cand_idx, device=device))
            return term_log_prob + cand_log_prob
        else:
            # Align with STOP when candidates couldn't be scored (matches forward())
            return term_dist.log_prob(torch.tensor(0, device=device))  # 0 = stop


# -----------------------------
# Path-to-snippet utility (optional)
# -----------------------------
def paths_to_snippets(paths: List[List[Tuple[str, str, str]]], node_label_func: Callable[[str], str], max_per_path: int = 2) -> List[str]:
    """
    Turn traversal paths [(h,r,t), ...] into short textual snippets.
    node_label_func: nid -> label string
    """
    snippets: List[str] = []
    for p in paths[:max_per_path]:
        s_parts: List[str] = []
        for triple in p:
            try:
                h, r, t = triple
            except Exception:
                # Skip non-triple entries defensively
                continue
            s_parts.append(f"{node_label_func(h)} {r} {node_label_func(t)}")
        if s_parts:
            snippets.append(" ; ".join(s_parts))
    return snippets
