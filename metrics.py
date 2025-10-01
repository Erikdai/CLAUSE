# -*- coding: utf-8 -*-
import time
from typing import Dict, List, Any, Optional, Tuple, Set
import torch
import numpy as np
from dataclasses import dataclass
import re
from collections import Counter


@dataclass
class LogicConsistencyMetrics:
    """Metrics for logic consistency evaluation"""
    total_steps: int
    consistent_steps: int
    logic_consistency: float
    rule_violations: List[Dict[str, Any]]
    had_rule_steps: int


@dataclass
class TraceF1Metrics:
    """Metrics for trace-F1 evaluation"""
    traversal_edges: Set[Tuple[str, str, str]]
    fired_rule_edges: Set[Tuple[str, str, str]]
    trace_f1: float
    precision: float
    recall: float


@dataclass
class PerformanceMetrics:
    """Performance metrics including latency and budget"""
    latency_ms: float
    latency_normalized: float  # relative to GraphRAG baseline
    edge_budget: float
    node_budget: float
    token_budget: float
    edge_budget_normalized: float  # relative to GraphRAG
    node_budget_normalized: float  # relative to GraphRAG


class AdvancedMetricsComputer:
    """Computes advanced evaluation metrics for LogiKG-MAPPO"""
    
    def __init__(self, graphrag_baseline: Optional[Dict[str, float]] = None):
        """
        Initialize the metrics computer
        
        Args:
            graphrag_baseline: Baseline metrics from GraphRAG for normalization
                Should contain: {'latency_ms': float, 'edges': float, 'nodes': float}
        """
        self.graphrag_baseline = graphrag_baseline or {
            'latency_ms': 100.0,  # Default baseline
            'edges': 100.0,
            'nodes': 50.0
        }

    # -------------------------------
    # Text normalization utilities
    # -------------------------------
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize text: lowercase, remove punctuation/articles, collapse whitespace.

        Mirrors common QA evaluation normalization (SQuAD-style):
        - Lowercase
        - Remove punctuation
        - Remove articles (a, an, the)
        - Collapse multiple spaces
        """
        if text is None:
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation (keep alphanumerics and whitespace)
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Remove articles
        tokens = text.split()
        tokens = [tok for tok in tokens if tok not in {"a", "an", "the"}]

        # Rejoin and collapse whitespace
        return " ".join(tokens)

    # -------------------------------
    # Text similarity metrics
    # -------------------------------
    @staticmethod
    def _ngram_counts(tokens: List[str], n: int) -> Counter:
        if n <= 0:
            return Counter()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def compute_bleu_scores(predicted: str, gold: str) -> Dict[str, float]:
        """Compute BLEU-1/2/4 with brevity penalty (simple implementation)."""
        pred = AdvancedMetricsComputer.normalize_answer(predicted).split()
        ref = AdvancedMetricsComputer.normalize_answer(gold).split()

        if len(pred) == 0 and len(ref) == 0:
            return {"bleu1": 1.0, "bleu2": 1.0, "bleu4": 1.0}
        if len(pred) == 0 or len(ref) == 0:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

        def precision_n(n: int) -> float:
            pred_counts = AdvancedMetricsComputer._ngram_counts(pred, n)
            ref_counts = AdvancedMetricsComputer._ngram_counts(ref, n)
            overlap = sum((pred_counts & ref_counts).values())
            total = max(sum(pred_counts.values()), 1)
            return overlap / total

        p1 = precision_n(1)
        p2 = precision_n(2)
        p4 = precision_n(4)

        # Brevity penalty
        ref_len = len(ref)
        pred_len = len(pred)
        if pred_len > 0 and pred_len < ref_len:
            bp = np.exp(1 - ref_len / pred_len)
        else:
            bp = 1.0

        bleu1 = bp * p1
        # For BLEU-2 and BLEU-4, use geometric mean of precisions up to n
        bleu2 = bp * (p1 * p2) ** 0.5
        # If p4 is zero with short strings, this will be zero; acceptable for sanity metric
        bleu4 = bp * (p1 * max(p2, 1e-9) * max(precision_n(3), 1e-9) * max(p4, 1e-9)) ** 0.25

        return {"bleu1": float(bleu1), "bleu2": float(bleu2), "bleu4": float(bleu4)}

    @staticmethod
    def compute_rouge_l(predicted: str, gold: str) -> float:
        """Compute ROUGE-L F1 using LCS between token sequences."""
        pred = AdvancedMetricsComputer.normalize_answer(predicted).split()
        ref = AdvancedMetricsComputer.normalize_answer(gold).split()
        if len(pred) == 0 and len(ref) == 0:
            return 1.0
        if len(pred) == 0 or len(ref) == 0:
            return 0.0

        # LCS dynamic programming
        m, n = len(pred), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        precision = lcs / m if m > 0 else 0.0
        recall = lcs / n if n > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        rouge_l = 2 * precision * recall / (precision + recall)
        return float(rouge_l)
    
    def compute_logic_consistency(
        self,
        decoded_sequence: List[str],
        violation_sets_per_step: List[Set[int]],
        rule_active_per_step: List[bool],
        token_to_id_mapping: Dict[str, int]
    ) -> LogicConsistencyMetrics:
        """
        Compute Logic Consistency (Logic-Cons) metric
        
        For a decoded sequence (y_1,...,y_T), let V_t be the union of rule-violation 
        token sets provided to DLL at step t, and let HadRule_t indicate whether at 
        least one rule mask was present at step t. We count a step as consistent if 
        either no rule was active or the sampled token is not in the violation set:
        
        Cons(t) = I[¬HadRule_t] ∨ I[y_t ∉ V_t]
        
        Logic-Cons is the fraction of steps satisfying Cons(t), averaged over examples.
        
        Args:
            decoded_sequence: List of decoded tokens [y_1, ..., y_T]
            violation_sets_per_step: List of violation token sets per step [V_1, ..., V_T]
            rule_active_per_step: List of booleans indicating if rules were active [HadRule_1, ..., HadRule_T]
            token_to_id_mapping: Mapping from tokens to token IDs
            
        Returns:
            LogicConsistencyMetrics object
        """
        total_steps = len(decoded_sequence)
        consistent_steps = 0
        rule_violations = []
        had_rule_steps = sum(rule_active_per_step)
        
        for t, (token, violation_set, had_rule) in enumerate(zip(
            decoded_sequence, violation_sets_per_step, rule_active_per_step
        )):
            # Get token ID
            token_id = token_to_id_mapping.get(token, -1)
            
            # Check consistency: either no rule active OR token not in violation set
            if not had_rule:
                # No rule active - automatically consistent
                is_consistent = True
                violation_type = "no_rule_active"
            else:
                # Rule active - check if token violates it
                is_consistent = token_id not in violation_set
                violation_type = "rule_violation" if not is_consistent else "rule_compliant"
            
            if is_consistent:
                consistent_steps += 1
            
            # Record rule violations for analysis
            if had_rule and not is_consistent:
                rule_violations.append({
                    'step': t,
                    'token': token,
                    'token_id': token_id,
                    'violation_set': list(violation_set),
                    'violation_type': violation_type
                })
        
        logic_consistency = consistent_steps / total_steps if total_steps > 0 else 0.0
        
        return LogicConsistencyMetrics(
            total_steps=total_steps,
            consistent_steps=consistent_steps,
            logic_consistency=logic_consistency,
            rule_violations=rule_violations,
            had_rule_steps=had_rule_steps
        )
    
    def compute_trace_f1(
        self,
        traversal_evidence_edges: Set[Tuple[str, str, str]],
        fired_rule_instances: Set[Tuple[str, str, str]]
    ) -> TraceF1Metrics:
        """
        Compute Trace-F1 metric
        
        Micro-F1 between the set of traversal evidence edges used by the TR agent 
        and the set of fired rule instances reconstructed from DLL masks over the 
        sequence. Trace-F1 measures whether the logic expert focuses on the same 
        evidence used by traversal.
        
        Args:
            traversal_evidence_edges: Set of edges used by traversal agent
            fired_rule_instances: Set of edges from fired rule instances
            
        Returns:
            TraceF1Metrics object
        """
        # Convert to sets for set operations
        traversal_set = set(traversal_evidence_edges)
        rule_set = set(fired_rule_instances)
        
        # Compute intersection
        intersection = traversal_set & rule_set
        
        # Compute precision and recall
        precision = len(intersection) / len(rule_set) if len(rule_set) > 0 else 0.0
        recall = len(intersection) / len(traversal_set) if len(traversal_set) > 0 else 0.0
        
        # Compute F1
        if precision + recall > 0:
            trace_f1 = 2 * (precision * recall) / (precision + recall)
        else:
            trace_f1 = 0.0
        
        return TraceF1Metrics(
            traversal_edges=traversal_set,
            fired_rule_edges=rule_set,
            trace_f1=trace_f1,
            precision=precision,
            recall=recall
        )
    
    def compute_performance_metrics(
        self,
        total_time_ms: float,
        subgraph_edges: int,
        subgraph_nodes: int,
        token_consumption: int,
        graphrag_comparison: Optional[Dict[str, float]] = None
    ) -> PerformanceMetrics:
        """
        Compute performance metrics including latency and budget
        
        Args:
            total_time_ms: Total wall-clock time in milliseconds
            subgraph_edges: Number of edges in subgraph
            subgraph_nodes: Number of nodes in subgraph
            token_consumption: Number of tokens consumed
            graphrag_comparison: Optional specific comparison metrics
            
        Returns:
            PerformanceMetrics object
        """
        # Use provided comparison or default baseline
        baseline = graphrag_comparison or self.graphrag_baseline
        
        # Normalize latency to GraphRAG baseline
        latency_normalized = total_time_ms / baseline['latency_ms']
        
        # Normalize edge and node budgets to GraphRAG baseline
        edge_budget_normalized = subgraph_edges / baseline['edges']
        node_budget_normalized = subgraph_nodes / baseline['nodes']
        
        return PerformanceMetrics(
            latency_ms=total_time_ms,
            latency_normalized=latency_normalized,
            edge_budget=subgraph_edges,
            node_budget=subgraph_nodes,
            token_budget=token_consumption,
            edge_budget_normalized=edge_budget_normalized,
            node_budget_normalized=node_budget_normalized
        )
    
    def compute_exact_match_f1(
        self,
        predicted_answer: str,
        gold_answer: str
    ) -> Tuple[bool, float]:
        """
        Compute Exact Match (EM) and F1 following standard QA evaluation
        
        Args:
            predicted_answer: Model's predicted answer
            gold_answer: Ground truth answer
            
        Returns:
            Tuple of (exact_match: bool, f1_score: float)
        """
        # Normalize both predicted and gold answers
        normalized_pred = self.normalize_answer(predicted_answer)
        normalized_gold = self.normalize_answer(gold_answer)

        # Exact Match on normalized strings
        exact_match = normalized_pred == normalized_gold

        # F1 Score using token frequency overlap (bag-of-words with counts)
        pred_tokens = normalized_pred.split()
        gold_tokens = normalized_gold.split()

        if len(pred_tokens) == 0 and len(gold_tokens) == 0:
            f1_score = 1.0
        elif len(pred_tokens) == 0 or len(gold_tokens) == 0:
            f1_score = 0.0
        else:
            pred_counts = Counter(pred_tokens)
            gold_counts = Counter(gold_tokens)

            # Overlap count is sum of minimum counts per token
            overlap_counts = pred_counts & gold_counts  # intersection with min counts
            overlap = sum(overlap_counts.values())

            precision = overlap / sum(pred_counts.values()) if pred_counts else 0.0
            recall = overlap / sum(gold_counts.values()) if gold_counts else 0.0

            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
        
        return exact_match, f1_score
    
    def compute_all_metrics(
        self,
        decoded_sequence: List[str],
        violation_sets_per_step: List[Set[int]],
        rule_active_per_step: List[bool],
        token_to_id_mapping: Dict[str, int],
        traversal_evidence_edges: Set[Tuple[str, str, str]],
        fired_rule_instances: Set[Tuple[str, str, str]],
        total_time_ms: float,
        subgraph_edges: int,
        subgraph_nodes: int,
        token_consumption: int,
        predicted_answer: str,
        gold_answer: str,
        graphrag_comparison: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compute all advanced evaluation metrics
        
        Args:
            All parameters needed for individual metric computations
            
        Returns:
            Dictionary containing all computed metrics
        """
        # Compute individual metrics
        logic_cons = self.compute_logic_consistency(
            decoded_sequence, violation_sets_per_step, 
            rule_active_per_step, token_to_id_mapping
        )
        
        trace_f1 = self.compute_trace_f1(
            traversal_evidence_edges, fired_rule_instances
        )
        
        performance = self.compute_performance_metrics(
            total_time_ms, subgraph_edges, subgraph_nodes, 
            token_consumption, graphrag_comparison
        )
        
        exact_match, f1_score = self.compute_exact_match_f1(predicted_answer, gold_answer)
        bleu = self.compute_bleu_scores(predicted_answer, gold_answer)
        rouge_l = self.compute_rouge_l(predicted_answer, gold_answer)

        # Logic extras
        total_steps = len(decoded_sequence)
        active_steps = sum(1 for b in rule_active_per_step if b)
        rule_coverage = (active_steps / total_steps) if total_steps > 0 else 0.0
        total_violations = 0
        for vset in violation_sets_per_step:
            total_violations += len(vset)
        violation_rate = (total_violations / total_steps) if total_steps > 0 else 0.0
        
        # Compile all metrics
        all_metrics = {
            # Standard QA metrics
            'exact_match': exact_match,
            'f1_score': f1_score,
            'bleu1': bleu['bleu1'],
            'bleu2': bleu['bleu2'],
            'bleu4': bleu['bleu4'],
            'rouge_l': rouge_l,
            
            # Logic consistency metrics
            'logic_consistency': logic_cons.logic_consistency,
            'total_steps': logic_cons.total_steps,
            'consistent_steps': logic_cons.consistent_steps,
            'had_rule_steps': logic_cons.had_rule_steps,
            'rule_violations': logic_cons.rule_violations,
            
            # Trace-F1 metrics
            'trace_f1': trace_f1.trace_f1,
            'trace_precision': trace_f1.precision,
            'trace_recall': trace_f1.recall,
            'traversal_edges_count': len(trace_f1.traversal_edges),
            'fired_rule_edges_count': len(trace_f1.fired_rule_edges),
            # Logic extras
            'rule_coverage': rule_coverage,
            'violation_rate': violation_rate,
            
            # Performance metrics
            'latency_ms': performance.latency_ms,
            'latency_normalized': performance.latency_normalized,
            'edge_budget': performance.edge_budget,
            'node_budget': performance.node_budget,
            'token_budget': performance.token_budget,
            'edge_budget_normalized': performance.edge_budget_normalized,
            'node_budget_normalized': performance.node_budget_normalized,
        }
        
        return all_metrics
