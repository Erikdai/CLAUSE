# kg_env.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import random

Triple = Tuple[str, str, str]


class KGQAEnvironment:
    def __init__(
        self,
        ttl_file: str = "/path/to/datasets/",
        cache_dir: str = "/path/",

        # ---- Composite-reward weights (new) ----
        w_acc: float = 1.0,
        w_logic: Optional[float] = None,   # if None, will use logic_bonus or default
        w_latency: Optional[float] = None, # if None, will use latency_alpha or default
        w_edge: float = 0.02,
        w_node: float = 0.00,
        w_hop: float = 0.01,  # Per-hop penalty weight

        # ---- Normalizers / references ----
        token_budget_ref: int = 1500,  # same scale as the builder's estimate (12*|V| + 6*|E|)
        edge_ref: int = 400,           # reference #edges for normalization
        node_ref: int = 200,           # reference #nodes for normalization

        # ---- Legacy args (kept for backward compatibility) ----
        logic_bonus: Optional[float] = None,
        latency_alpha: Optional[float] = None,

        # ---- Misc ----
        vocab_size: int = 2048,
        max_dec_steps: int = 6,
        max_hops: int = 4,  # Maximum traversal hops
        max_path_length: int = 10,  # Maximum path length
        seed: int = 13,
        debug_logging: bool = False,  # Enable debug logging
    ):
        self.ttl_file = ttl_file
        self.cache_dir = cache_dir
        self.vocab_size = int(vocab_size)
        self.max_dec_steps = int(max_dec_steps)
        self.max_hops = int(max_hops)
        self.max_path_length = int(max_path_length)
        self._debug_logging = bool(debug_logging)

        # Map legacy args if present
        if w_logic is None:
            w_logic = 0.05 if logic_bonus is None else float(logic_bonus)
        if w_latency is None:
            w_latency = 0.03 if latency_alpha is None else float(latency_alpha)

        # Reward weights
        self.w_acc = float(w_acc)
        self.w_logic = float(w_logic)
        self.w_latency = float(w_latency)
        self.w_edge = float(w_edge)
        self.w_node = float(w_node)
        self.w_hop = float(w_hop)

        # Normalizers
        self.token_budget_ref = int(max(1, token_budget_ref))
        self.edge_ref = int(max(1, edge_ref))
        self.node_ref = int(max(1, node_ref))

        self.rng = random.Random(seed)

        # Runtime state
        self.reset_state()

        # Expose last scalar reward and detailed breakdown
        self.last_reward: float = 0.0
        self.last_components: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #

    def reset_state(self) -> None:
        self.question: str = ""
        self.answer_token_id: int = 0  # placeholder GT token id

        # Subgraph chosen by GB
        self.nodes: Set[str] = set()
        self.edges: Set[Triple] = set()
        self.adj: Dict[str, List[Triple]] = {}

        # Traversal/Decoding state
        self.current_node: Optional[str] = None
        self.path: List[Union[str, Triple]] = []
        self.tr_stopped: bool = False
        self.dec_steps: int = 0

        # Bookkeeping
        self.t: int = 0  # step counter within episode
        self.gb_done: bool = False

        # Reward components
        self.last_components = {}
        
        # Token mappings for termination logic
        self.token_to_id = {
            "<bos>": 0,
            "<pad>": 1,
            "<unk>": 2,
            "<eos>": 3,
            "yes": 4,
            "no": 5
        }

    def reset(self, question: str, gold_answer: str = None) -> Dict[str, Any]:
        """Start a new episode with a question. Returns initial observation for GB."""
        self.reset_state()
        self.question = str(question or "")
        
        # Use ground-truth answer if provided, otherwise use a placeholder
        if gold_answer:
            # Map the ground-truth answer to a token ID using the decoder's vocabulary
            self.answer_token_id = self._map_answer_to_token(gold_answer)
        else:
            # Fallback to a default token ID if no ground-truth answer is provided
            self.answer_token_id = 0

        obs = {
            "gb_observation": {
                "question": self.question,
                "gold_answer": gold_answer,  # Include gold answer in observation
                "reset": True,
                "t": 0,
            }
        }
        return obs

    # ------------------------------------------------------------------ #
    # Step functions for GB/TR/DEC
    # ------------------------------------------------------------------ #

    def step_gb(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, float]]:
        """
        Graph-Builder step with incremental graph edits.
        Expects `action` to contain:
            - operation_type: str ("add_edge", "delete_edge", "stop")
            - subject: Optional[str] (for edge operations)
            - predicate: Optional[str] (for edge operations)
            - object: Optional[str] (for edge operations)
            - nodes: List[str] (current subgraph state)
            - edges: List[Tuple[str,str,str]] (current subgraph state)
            - num_nodes: int
            - num_edges: int
        Returns (next_obs_for_TR, reward, done=False, cost_components).
        """
        operation_type = action.get("operation_type", "stop")
        
        # Remember previous sizes to detect structural changes
        prev_nodes_len = len(self.nodes)
        prev_edges_len = len(self.edges)

        # Handle incremental graph edits
        if operation_type == "add_edge":
            subject = action.get("subject")
            predicate = action.get("predicate")
            object_ = action.get("object")
            
            if subject and predicate and object_:
                # Add the edge to current subgraph
                if (subject, predicate, object_) not in self.edges:
                    self.edges.add((subject, predicate, object_))
                    self.nodes.add(subject)
                    self.nodes.add(object_)
        
        elif operation_type == "delete_edge":
            subject = action.get("subject")
            predicate = action.get("predicate")
            object_ = action.get("object")
            
            if subject and predicate and object_:
                # Remove the edge from current subgraph
                if (subject, predicate, object_) in self.edges:
                    self.edges.remove((subject, predicate, object_))
                    # Clean up isolated nodes
                    self._cleanup_isolated_nodes()
        
        # Update subgraph state from action (in case of inconsistencies)
        V = self._as_set(action.get("nodes", []))
        E = self._as_set(action.get("edges", []))
        
        # Use the action's reported state if it's more complete
        if len(V) > len(self.nodes) or len(E) > len(self.edges):
            self.nodes = V
            self.edges = E
        
        self._build_adjacency()

        # Counts (prefer reported counts; fall back to recompute)
        num_nodes = int(action.get("num_nodes", len(self.nodes)))
        num_edges = int(action.get("num_edges", len(self.edges)))

        # Choose a deterministic starting node if any
        self.current_node = self._pick_start_node()

        # Estimate tokens based on current subgraph size
        est_tokens = 12 * len(self.nodes) + 6 * len(self.edges)

        # Normalized costs
        lat_cost = (est_tokens / float(self.token_budget_ref)) if est_tokens > 0 else 0.0
        edge_cost = (num_edges / float(self.edge_ref)) if self.edge_ref > 0 else 0.0
        node_cost = (num_nodes / float(self.node_ref)) if self.node_ref > 0 else 0.0

        # Composite reward (GB) - reward for incremental improvements
        r_latency = -self.w_latency * lat_cost
        r_edge = -self.w_edge * edge_cost
        r_node = -self.w_node * node_cost
        
        # Additional reward for successful operations (structural change-based)
        operation_reward = 0.0
        grew_nodes = len(self.nodes) > prev_nodes_len
        grew_edges = len(self.edges) > prev_edges_len
        shrunk_edges = len(self.edges) < prev_edges_len
        if operation_type == "add_edge":
            if grew_edges or grew_nodes:
                operation_reward = 0.1  # positive reward for growth
        elif operation_type == "delete_edge":
            if shrunk_edges:
                operation_reward = 0.05  # smaller reward for cleanup operations
        
        # Add subgraph building bonus to encourage meaningful graph construction
        subgraph_bonus = 0.0
        if num_nodes > 0 and num_edges > 0:
            # Reward for building non-empty subgraphs
            subgraph_bonus = 0.02
            # Additional bonus for building well-connected subgraphs
            if num_edges >= num_nodes:  # Good connectivity
                subgraph_bonus += 0.01
        
        reward = float(r_latency + r_edge + r_node + operation_reward + subgraph_bonus)

        # Save breakdown
        self.last_components = {
            "acc": 0.0,
            "logic": 0.0,
            "latency": float(r_latency),
            "edge": float(r_edge),
            "node": float(r_node),
            "operation": float(operation_reward),
            "subgraph": float(subgraph_bonus),
            "hop": 0.0,  # No hop penalty in graph builder step
            "total": float(reward),
        }

        self.last_reward = reward
        
        # Check if graph building is complete
        if operation_type == "stop" or num_nodes >= 200 or num_edges >= 400:
            self.gb_done = True
            self.t = 0
        else:
            self.gb_done = False

        obs = {
            "tr_observation": self._build_tr_obs(reset=True),
        }
        done = False
        
        # Return cost components alongside reward
        costs = {
            'edge': float(edge_cost),
            'latency': float(lat_cost),
            'logic': 0.0,  # No logic cost in graph builder step
        }
        
        return obs, reward, done, costs

    def step_tr(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, float]]:
        """
        Traversal step.
        Expects action dict as produced by traversal agent:
            - "stop": bool
            - "next_edge": Optional[Triple]
            - "next_node": Optional[str]
            - "path": List[Union[str, Triple]] (agent-maintained; optional)
        Returns (next_obs_for_DEC, reward, done_flag_after_TR, cost_components)
        """
        # Compute hop penalty for this step
        hop_penalty = -self.w_hop if not action.get("stop", False) else 0.0
        
        # Add exploration bonus to encourage continued traversal
        exploration_bonus = 0.0
        if not action.get("stop", False):
            # Reward for taking a step (encouraging exploration)
            exploration_bonus = 0.05
            # Additional bonus for building longer paths
            if len(self.path) > 0:
                exploration_bonus += 0.02 * min(len(self.path), 3)  # Cap at 3 steps
        
        reward = float(hop_penalty + exploration_bonus)
        
        # Debug logging
        if hasattr(self, '_debug_logging') and self._debug_logging:
            print(f"🔍 TR DEBUG: hop_penalty={hop_penalty:.6f}, exploration_bonus={exploration_bonus:.6f}, total_reward={reward:.6f}")
        
        done = False

        stop = bool(action.get("stop", False))
        self.tr_stopped = stop

        # Advance along chosen candidate if not stopping
        if not stop:
            next_edge: Optional[Triple] = action.get("next_edge")
            next_node: Optional[str] = action.get("next_node")

            if next_edge and self._triple_like(next_edge):
                self.path.append(next_edge)
                self.current_node = next_edge[2]
            elif isinstance(next_node, str):
                self.path.append(next_node)
                self.current_node = next_node

        # Enhanced termination conditions for traversal
        done = bool(
            stop or  # Agent chose to stop
            self.t >= self.max_hops or  # Maximum hops reached
            len(self.path) >= self.max_path_length or  # Maximum path length reached
            (self.current_node is None and len(self.path) > 0)  # Invalid state
        )

        obs = {
            "dec_observation": self._build_dec_obs(),
        }
        self.t += 1
        
        # Save breakdown for traversal step
        self.last_components = {
            "acc": 0.0,
            "logic": 0.0,
            "latency": 0.0,
            "edge": 0.0,
            "node": 0.0,
            "hop": float(hop_penalty),
            "exploration": float(exploration_bonus),
            "total": float(reward),
        }
        
        self.last_reward = float(reward)
        
        # Return cost components alongside reward
        costs = {
            'edge': 0.0,  # No edge cost in traversal step
            'latency': 0.0,  # No latency cost in traversal step
            'logic': 0.0,  # No logic cost in traversal step
        }
        
        return obs, float(reward), done, costs

    def step_dec(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, float]]:
        """
        Decoder step.
        Expects action dict:
            - "token_id": int
            - "active_rules_ok": bool

        Returns (next_obs_for_TR, reward, done_flag_after_DEC, cost_components)
        """
        token_id = int(action.get("token_id", -1))
        active_rules_ok = bool(action.get("active_rules_ok", True))

        # Terms
        acc_term = self.w_acc * (1.0 if token_id == self.answer_token_id else 0.0)
        logic_term = self.w_logic * (1.0 if active_rules_ok else 0.0)
        
        # Add generation bonus to encourage non-bos token generation
        generation_bonus = 0.0
        if token_id != 0:  # Not <bos> token
            generation_bonus = 0.03  # Small bonus for generating actual tokens
            # Additional bonus for generating meaningful tokens (not special tokens)
            if token_id > 10:  # Assuming special tokens are in range 0-10
                generation_bonus += 0.02

        reward = float(acc_term + logic_term + generation_bonus)
        
        # Debug logging
        if hasattr(self, '_debug_logging') and self._debug_logging:
            print(f"🔍 DEC DEBUG: acc_term={acc_term:.6f}, logic_term={logic_term:.6f}, generation_bonus={generation_bonus:.6f}, total_reward={reward:.6f}")
        
        self.dec_steps += 1

        # Enhanced termination conditions for decoder
        done = bool(
            self.tr_stopped or  # Traversal agent stopped
            (acc_term > 0.0) or  # Correct answer generated
            (self.dec_steps >= self.max_dec_steps) or  # Maximum decoder steps reached
            (token_id == self.token_to_id.get("<eos>", -1)) or  # End-of-sequence token
            (len(self.path) == 0 and self.dec_steps > 0)  # No traversal path available
        )

        # Next observation (if not done)
        next_obs = {"tr_observation": self._build_tr_obs(reset=False)} if not done else {}

        # Save breakdown (these replace the GB breakdown for the last step)
        self.last_components = {
            "acc": float(acc_term),
            "logic": float(logic_term),
            "latency": 0.0,
            "edge": 0.0,
            "node": 0.0,
            "hop": 0.0,  # No hop penalty in decoder step
            "generation": float(generation_bonus),
            "total": float(reward),
        }

        self.last_reward = reward
        
        # Return cost components alongside reward
        costs = {
            'edge': 0.0,  # No edge cost in decoder step
            'latency': 0.0,  # No latency cost in decoder step
            'logic': float(1.0 if not active_rules_ok else 0.0),  # Logic cost based on rule violations
        }
        
        return next_obs, reward, done, costs

    def should_terminate_episode(self) -> bool:
        """
        Check if the episode should terminate early based on current state.
        
        Returns:
            bool: True if episode should terminate, False otherwise
        """
        return bool(
            self.tr_stopped or  # Traversal agent stopped
            self.t >= self.max_hops or  # Maximum traversal hops reached
            len(self.path) >= self.max_path_length or  # Maximum path length reached
            self.dec_steps >= self.max_dec_steps or  # Maximum decoder steps reached
            (self.current_node is None and len(self.path) > 0) or  # Invalid traversal state
            (len(self.path) == 0 and self.dec_steps > 0)  # No path available for decoding
        )

    # ------------------------------------------------------------------ #
    # Observation builders
    # ------------------------------------------------------------------ #

    def _build_tr_obs(self, reset: bool) -> Dict[str, Any]:
        neighbors: List[Triple] = []
        if self.current_node is not None:
            neighbors = list(self.adj.get(self.current_node, []))
        if len(neighbors) > 128:
            neighbors = neighbors[:128]
        return {
            "question": self.question,
            "current_node": self.current_node,
            "neighbors": neighbors,
            "path": list(self.path),
            "t": int(self.t),
            "reset": bool(reset),
        }

    def _build_dec_obs(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "path": list(self.path),
            "t": int(self.t),
        }

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_set(x: Any) -> Set:
        try:
            return set(x) if x is not None else set()
        except Exception:
            return set()

    @staticmethod
    def _triple_like(x: Any) -> bool:
        return isinstance(x, (tuple, list)) and len(x) == 3 and all(isinstance(xx, str) for xx in x)

    def _build_adjacency(self) -> None:
        """Construct adjacency from self.edges {head: [triples...]}."""
        self.adj = {}
        for e in self.edges:
            if not self._triple_like(e):
                continue
            h, r, t = e  # type: ignore
            self.adj.setdefault(h, []).append((h, r, t))

    def _cleanup_isolated_nodes(self):
        """Remove nodes that are no longer connected to any edges."""
        connected_nodes = set()
        for s, p, o in self.edges:
            if self._triple_like((s, p, o)):
                connected_nodes.add(s)
                connected_nodes.add(o)
        
        isolated = self.nodes - connected_nodes
        self.nodes -= isolated

    def _pick_start_node(self) -> Optional[str]:
        """Heuristic: pick the node with the most outgoing edges; fallback to any node."""
        if not self.nodes:
            return None
        degree = {n: 0 for n in self.nodes}
        for h, _, _ in self.edges:
            if h in degree:
                degree[h] += 1
        if degree:
            return max(degree.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return sorted(list(self.nodes))[0]

    def _map_answer_to_token(self, answer: str) -> int:
        """Map a ground-truth answer to a token ID using the decoder's vocabulary."""
        if not answer:
            return 0
    
        answer_lower = answer.lower().strip()
        
        # Check for common answer patterns
        if answer_lower in self.token_to_id:
            return self.token_to_id[answer_lower]
        
        # Check for variations 
        answer_title = answer.title().strip()
        if answer_title in self.token_to_id:
            return self.token_to_id[answer_title]
        
        # Check for partial matches in the vocabulary
        for token, token_id in self.token_to_id.items():
            if token.lower() in answer_lower or answer_lower in token.lower():
                return token_id
        
        import hashlib
        h = hashlib.md5(answer.encode("utf-8")).hexdigest()
        return int(h, 16) % max(1, self.vocab_size)
