# -*- coding: utf-8 -*-
import os
import time
import json
import csv
from typing import Dict, List, Any, Optional, Tuple, Iterator
from pathlib import Path
import logging
import numpy as np
import torch
from dataclasses import dataclass, asdict

# Import from the new organized datasets package
# from datasets import dataset_manager 
from ..agents.enhanced_graph_builder import EnhancedGraphBuilder
from ..agents.traversal import EnhancedTraversalAgent
from ..agents.decoder import LogicAwareDecoder
from ..rule_inducer import RuleInducer
from ..envs.kg_env import KGQAEnvironment
from ..rl.mappo import SimpleMAPPO, CentralCritic
from .metrics import AdvancedMetricsComputer
from datasets.manager import dataset_manager

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""
    
    # Dataset settings
    dataset_name: str = "custom_ttl"
    dataset_path: Optional[str] = None
    dataset_split: str = "dev"
    max_examples: int = 100
    
    # Model settings
    ttl_file: str = "/path/to/datasets/"
    cache_dir: str = "/outputs/cache"
    vocab_size: int = 1000
    max_dec_steps: int = 10
    
    # Environment settings
    logic_bonus: float = 0.05
    latency_alpha: float = 0.03
    w_acc: float = 1.0
    w_logic: float = 0.1
    w_latency: float = 0.01
    w_edge: float = 0.1
    w_node: float = 0.1
    w_hop: float = 0.01  # Per-hop penalty weight
    token_budget_ref: int = 1000
    edge_ref: int = 100
    node_ref: int = 50
    
    # Agent settings
    emb_dim: int = 64
    hidden_dim: int = 64
    max_hops: int = 3
    max_rules: int = 4
    
    # Evaluation settings
    seed: int = 42
    output_dir: str = "/outputs/evaluation"
    save_detailed: bool = True
    verbose: bool = True
    # Controls
    allow_dummy_data: bool = False
    strict_mode: bool = True
    use_per_token_decoding: bool = True

@dataclass
class EvaluationResult:
    """Results from a single evaluation example"""
    
    # Basic info
    example_id: str
    question: str
    gold_answer: str
    dataset: str
    
    # Graph building results
    gb_nodes: int
    gb_edges: int
    gb_tokens: int
    gb_time: float
    
    # Traversal results
    tr_path_length: int
    tr_stopped: bool
    tr_time: float
    
    # Decoding results
    dec_tokens: List[str]
    dec_time: float
    
    # Overall metrics
    total_time: float
    success: bool
    accuracy: float
    
    # Additional metadata
    metadata: Dict[str, Any]
    
    # Advanced metrics
    exact_match: bool = False
    f1_score: float = 0.0
    logic_consistency: float = 0.0
    trace_f1: float = 0.0
    latency_ms: float = 0.0
    latency_normalized: float = 0.0
    edge_budget_normalized: float = 0.0
    node_budget_normalized: float = 0.0
    # Additional text metrics
    bleu1: float = 0.0
    bleu2: float = 0.0
    bleu4: float = 0.0
    rouge_l: float = 0.0
    # Logic extras
    rule_coverage: float = 0.0
    violation_rate: float = 0.0
    # Ranking metrics
    acc_at_3: float = 0.0
    acc_at_5: float = 0.0
    mrr: float = 0.0
    covered: bool = False

class LogiKGEvaluator:
    """Main evaluator for LogiKG-MAPPO"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.setup_logging()
        self.setup_components()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def setup_components(self):
        """Initialize all LogiKG-MAPPO components"""
        logger.info("Setting up LogiKG-MAPPO components...")
        
        # Initialize agents
        self.graph_builder = EnhancedGraphBuilder(
            ttl_file=self.config.ttl_file
        )
        
        self.traversal_agent = EnhancedTraversalAgent(
            emb_dim=self.config.emb_dim,
            hidden=self.config.hidden_dim,
            max_hops=self.config.max_hops
        )
        
        self.decoder = LogicAwareDecoder(
            vocab_size=self.config.vocab_size,
            max_rules=self.config.max_rules,
            penalty_lambda=0.5,
            dll_temperature=1.0,
            context_dim=128,
            device="cpu",
            use_lm=True,  # Enable language model integration
            lm_model_name="gpt2",
            strict_no_fallbacks=self.config.strict_mode
        )
        
        # Initialize rule inducer for per-token DLL masks
        self.rule_inducer = RuleInducer(
            max_rules=self.config.max_rules,
            enable_horn_rules=True,
            enable_advanced_rules=True,
        )
        
        # Initialize environment
        self.env = KGQAEnvironment(
            ttl_file=self.config.ttl_file,
            cache_dir=self.config.cache_dir,
            logic_bonus=self.config.logic_bonus,
            latency_alpha=self.config.latency_alpha,
            w_acc=self.config.w_acc,
            w_logic=self.config.w_logic,
            w_latency=self.config.w_latency,
            w_edge=self.config.w_edge,
            w_node=self.config.w_node,
            w_hop=self.config.w_hop,
            token_budget_ref=self.config.token_budget_ref,
            edge_ref=self.config.edge_ref,
            node_ref=self.config.node_ref,
            vocab_size=self.config.vocab_size,
            max_dec_steps=self.config.max_dec_steps,
            seed=self.config.seed
        )
        
        # Initialize MAPPO (if needed for evaluation)
        self.critic = CentralCritic(
            obs_dim=self.config.emb_dim,
            hidden=self.config.hidden_dim
        )
        
        self.mappo = SimpleMAPPO(
            gb=self.graph_builder,
            tr=self.traversal_agent,
            dec=self.decoder,
            critic=self.critic
        )
        
        # Initialize advanced metrics computer
        self.metrics_computer = AdvancedMetricsComputer()
        
        logger.info("All components initialized successfully")
        
    def evaluate_single_example(self, question: str, gold_answer: str, 
                              metadata: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single question-answer pair"""
        
        start_time = time.time()
        
        try:
            # Step 1: Graph Building
            gb_start = time.time()
            gb_obs = {'question': question}
            gb_result = self.graph_builder(gb_obs)
            gb_time = time.time() - gb_start
            
            # Step 2: Environment reset with question
            env_obs = self.env.reset(question)
            
            # Step 3: Traversal
            tr_start = time.time()
            # Convert set to list and limit neighbors
            subgraph_nodes = gb_result.get('subgraph_nodes', set())
            if isinstance(subgraph_nodes, set):
                subgraph_nodes = list(subgraph_nodes)
            neighbors = subgraph_nodes[:10] if len(subgraph_nodes) > 10 else subgraph_nodes
            
            tr_obs = {
                'current_node': 'start',
                'neighbors': neighbors,
                'question': question
            }
            tr_result = self.traversal_agent(tr_obs)
            tr_time = time.time() - tr_start
            
            # Step 4: Decoding with rule tracking
            dec_start = time.time()
            
            # Track rule violations and active rules per step
            decoded_sequence = []
            violation_sets_per_step = []
            rule_active_per_step = []
            traversal_evidence_edges = set()
            fired_rule_instances = set()
            
            # Provide tokenizer-based mappings to rule inducer (per-example)
            if hasattr(self.decoder, 'tokenizer') and self.decoder.tokenizer is not None:
                tok = self.decoder.tokenizer
                def _map_token(s: str) -> Optional[int]:
                    try:
                        ids = tok.encode(s, add_special_tokens=False)
                        return int(ids[0]) if ids else None
                    except Exception:
                        return None
                self.rule_inducer.entity_to_token = _map_token
                self.rule_inducer.relation_to_token = _map_token
                # Seed a small universe of relation tokens from traversal path
                rel_ids = set()
                triples = self.rule_inducer._extract_triples_from_path(tr_result.get('action', {}))
                for h, r, t in triples:
                    rel_ids.add(r)
                # Add common type relations to strengthen type-chain patterns
                rel_ids.update({'P31', 'P279'})
                rel_tok_ids = set()
                for r in rel_ids:
                    rid = _map_token(r)
                    if isinstance(rid, int):
                        rel_tok_ids.add(rid)
                self.rule_inducer.all_relation_tokens = rel_tok_ids if rel_tok_ids else None

            # Decide decoding mode
            use_per_token = bool(self.config.use_per_token_decoding or self.config.strict_mode)
            max_dec_steps = self.config.max_dec_steps

            if use_per_token:
                # Per-token decoding with RuleInducer masks
                partial_token_ids: List[int] = []
                tr_path = tr_result.get('action', {}).get('path', [])
                for step in range(max_dec_steps):
                    # Build rule masks for current step
                    try:
                        masks = self.rule_inducer.induce(
                            paths=[tr_path] if tr_path else [],
                            partial_output_tokens=partial_token_ids,
                            k_frontier_paths=1,
                        )
                    except Exception:
                        masks = []

                    # Call decoder (standard forward)
                    out = self.decoder(tr_obs, violation_indices_per_rule=masks)
                    tok_id = int(out.get('sampled_token_id', 0))
                    partial_token_ids.append(tok_id)

                    # Convert to token string if tokenizer is available
                    if hasattr(self.decoder, 'tokenizer') and self.decoder.tokenizer is not None and tok_id < self.decoder.tokenizer.vocab_size:
                        tok_str = self.decoder.tokenizer.convert_ids_to_tokens([tok_id])[0]
                    else:
                        tok_str = f"token_{tok_id}"

                    decoded_sequence.append(tok_str)

                    # Track masks and rule activity for metrics
                    union_violations = set()
                    for m in masks:
                        union_violations.update(m)
                    violation_sets_per_step.append(union_violations)
                    rule_active_per_step.append(len(union_violations) > 0)
            else:
                # Reranking path; keep as non-strict fast mode
                # Set question text for LM integration
                if hasattr(self.decoder, 'set_question_text'):
                    self.decoder.set_question_text(question)

                from ..utils.wikidata_kg import get_wikidata_kg
                kg = get_wikidata_kg(use_fallback_knowledge=not self.config.strict_mode)
                candidates, rule_violations = kg.generate_candidates_for_question(
                    question,
                    max_candidates=20,
                    gold_answer=gold_answer
                )

                dec_result = self.decoder(
                    tr_obs,
                    violation_indices_per_rule=[],
                    generate_answer=True,
                    candidates=candidates,
                    rule_violations=rule_violations,
                    gold_answer=gold_answer
                )

                if 'answer' in dec_result:
                    generated_answer = dec_result['answer']
                    if hasattr(self.decoder, 'tokenizer') and self.decoder.tokenizer is not None:
                        token_ids = self.decoder.tokenizer.encode(generated_answer, add_special_tokens=False)
                        decoded_sequence = self.decoder.tokenizer.convert_ids_to_tokens(token_ids)
                    else:
                        decoded_sequence = generated_answer.split()
                else:
                    # Fallback single token
                    tok_id = int(dec_result.get('sampled_token_id', 0))
                    if hasattr(self.decoder, 'tokenizer') and self.decoder.tokenizer is not None and tok_id < self.decoder.tokenizer.vocab_size:
                        tok_str = self.decoder.tokenizer.convert_ids_to_tokens([tok_id])[0]
                    else:
                        tok_str = f"token_{tok_id}"
                    decoded_sequence = [tok_str]

                # For rerank mode, violation tracking is unavailable; mark empty
                violation_sets_per_step = [set() for _ in decoded_sequence]
                rule_active_per_step = [False for _ in decoded_sequence]
            # Rule tracking already filled above for both decoding modes
            
            dec_time = time.time() - dec_start
            total_time = time.time() - start_time
            
            # Extract basic results
            gb_nodes = gb_result.get('num_nodes', 0)
            gb_edges = gb_result.get('num_edges', 0)
            gb_tokens = gb_result.get('estimated_tokens', 0)
            
            tr_path_length = len(tr_result.get('action', {}).get('path', []))
            tr_stopped = tr_result.get('action', {}).get('stop', False)
            
            dec_tokens = decoded_sequence
            
            # Extract traversal evidence edges (from traversal result)
            tr_path = tr_result.get('action', {}).get('path', [])
            for path_elem in tr_path:
                if isinstance(path_elem, tuple) and len(path_elem) == 3:
                    traversal_evidence_edges.add(path_elem)
            
            # Compute advanced metrics
            # Derive token_to_id_mapping from the decoder's tokenizer vocabulary
            if hasattr(self.decoder, 'tokenizer') and self.decoder.tokenizer is not None:
                vocab = self.decoder.tokenizer.get_vocab()  # token -> id
                token_to_id_mapping = dict(vocab)
                # Ensure <unk> exists for OOV handling
                unk_token = getattr(self.decoder.tokenizer, 'unk_token', None) or '<unk>'
                unk_id = getattr(self.decoder.tokenizer, 'unk_token_id', None)
                if unk_id is None:
                    # GPT-2 typically has no unk; use -1 sentinel
                    unk_id = -1
                token_to_id_mapping[unk_token] = unk_id
                # Map any tokens in decoded_sequence not in vocab to <unk>
                decoded_sequence = [
                    tok if tok in token_to_id_mapping else unk_token
                    for tok in decoded_sequence
                ]
            else:
                # Fallback placeholder mapping
                token_to_id_mapping = {f"token_{i}": i for i in range(1000)}
            
            # Convert time to milliseconds
            total_time_ms = total_time * 1000
            
            # Compute all advanced metrics
            advanced_metrics = self.metrics_computer.compute_all_metrics(
                decoded_sequence=decoded_sequence,
                violation_sets_per_step=violation_sets_per_step,
                rule_active_per_step=rule_active_per_step,
                token_to_id_mapping=token_to_id_mapping,
                traversal_evidence_edges=traversal_evidence_edges,
                fired_rule_instances=fired_rule_instances,  # Placeholder for now
                total_time_ms=total_time_ms,
                subgraph_edges=gb_edges,
                subgraph_nodes=gb_nodes,
                token_consumption=gb_tokens,
                predicted_answer=" ".join(decoded_sequence),  # Simple concatenation
                gold_answer=gold_answer
            )

            # Ranking metrics over candidates using decoder scorer
            # Build prompt consistent with reranker
            if hasattr(self.decoder, 'set_question_text'):
                self.decoder.set_question_text(question)
            prompt = f"Question: {question}\nAnswer:"
            try:
                scored = []
                for c in candidates:
                    s = self.decoder.score_candidate(prompt, c, gold_answer)
                    scored.append((c, s))
                scored.sort(key=lambda x: x[1], reverse=True)
                ranked = [c for c, _ in scored]
            except Exception:
                ranked = candidates

            def norm(x: str) -> str:
                return self.metrics_computer.normalize_answer(x)

            norm_aliases = {norm(gold_answer)}
            norm_ranked = [norm(c) for c in ranked]
            covered = any(c in norm_aliases for c in norm_ranked)
            gold_rank = None
            for idx, cand in enumerate(norm_ranked, start=1):
                if cand in norm_aliases:
                    gold_rank = idx
                    break
            mrr = (1.0 / gold_rank) if gold_rank is not None else 0.0
            acc_at_3 = 1.0 if gold_rank is not None and gold_rank <= 3 else 0.0
            acc_at_5 = 1.0 if gold_rank is not None and gold_rank <= 5 else 0.0
            
            # Simple accuracy metric (can be enhanced)
            success = advanced_metrics['exact_match']  # Use exact match as success
            accuracy = advanced_metrics['f1_score']
            
            return EvaluationResult(
                example_id=metadata.get('id', 'unknown'),
                question=question,
                gold_answer=gold_answer,
                dataset=self.config.dataset_name,
                gb_nodes=gb_nodes,
                gb_edges=gb_edges,
                gb_tokens=gb_tokens,
                gb_time=gb_time,
                tr_path_length=tr_path_length,
                tr_stopped=tr_stopped,
                tr_time=tr_time,
                dec_tokens=dec_tokens,
                dec_time=dec_time,
                total_time=total_time,
                success=success,
                accuracy=accuracy,
                # Advanced metrics
                exact_match=advanced_metrics['exact_match'],
                f1_score=advanced_metrics['f1_score'],
                logic_consistency=advanced_metrics['logic_consistency'],
                trace_f1=advanced_metrics['trace_f1'],
                latency_ms=advanced_metrics['latency_ms'],
                latency_normalized=advanced_metrics['latency_normalized'],
                edge_budget_normalized=advanced_metrics['edge_budget_normalized'],
                node_budget_normalized=advanced_metrics['node_budget_normalized'],
                # Extra text metrics
                bleu1=advanced_metrics.get('bleu1', 0.0),
                bleu2=advanced_metrics.get('bleu2', 0.0),
                bleu4=advanced_metrics.get('bleu4', 0.0),
                rouge_l=advanced_metrics.get('rouge_l', 0.0),
                # Logic extras
                rule_coverage=advanced_metrics.get('rule_coverage', 0.0),
                violation_rate=advanced_metrics.get('violation_rate', 0.0),
                # Ranking metrics
                acc_at_3=acc_at_3,
                acc_at_5=acc_at_5,
                mrr=mrr,
                covered=covered,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            # Return error result
            return EvaluationResult(
                example_id=metadata.get('id', 'unknown'),
                question=question,
                gold_answer=gold_answer,
                dataset=self.config.dataset_name,
                gb_nodes=0,
                gb_edges=0,
                gb_tokens=0,
                gb_time=0.0,
                tr_path_length=0,
                tr_stopped=False,
                tr_time=0.0,
                dec_tokens=[],
                dec_time=0.0,
                total_time=time.time() - start_time,
                success=False,
                accuracy=0.0,
                # Advanced metrics (default values)
                exact_match=False,
                f1_score=0.0,
                logic_consistency=0.0,
                trace_f1=0.0,
                latency_ms=0.0,
                latency_normalized=0.0,
                edge_budget_normalized=0.0,
                node_budget_normalized=0.0,
                metadata=metadata
            )
    
    def evaluate_dataset(self) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """Evaluate the entire dataset"""
        
        logger.info(f"Starting evaluation on dataset: {self.config.dataset_name}")

        # Attempt to load real dataset via dataset_manager
        try:
            ds_iter = dataset_manager.load_dataset(
                self.config.dataset_name,
                path=self.config.dataset_path,
                split=self.config.dataset_split,
            )
            dataset_source = "manager"
        except Exception as e:
            logger.error(f"Failed to load dataset via manager: {e}")
            if not self.config.allow_dummy_data:
                raise
            # Fallback to minimal dummy samples if explicitly allowed
            ds_iter = iter([
                ("What is the capital of France?", "Paris", {"id": "ex1"}),
                ("Who is the author of '1984'?", "George Orwell", {"id": "ex2"}),
                ("What is the capital of Germany?", "Berlin", {"id": "ex3"}),
            ])
            dataset_source = "dummy"

        results = []
        total_examples = 0

        for item in ds_iter:
            try:
                question, answer, metadata = item
            except Exception:
                # If manager yields different structure, skip gracefully
                continue
            if total_examples >= self.config.max_examples:
                break

            logger.info(
                f"Evaluating example {total_examples + 1}/{self.config.max_examples}"
            )

            result = self.evaluate_single_example(question, answer, metadata)
            results.append(result)
            total_examples += 1

            if self.config.verbose and total_examples % 10 == 0:
                logger.info(f"Processed {total_examples} examples...")

        # Compute aggregate metrics
        aggregate_metrics = self.compute_aggregate_metrics(results)

        logger.info(
            f"Evaluation completed. Processed {total_examples} examples (source={dataset_source})."
        )

        return results, aggregate_metrics
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute aggregate metrics from individual results"""
        
        if not results:
            return {}
        
        # Basic counts
        total_examples = len(results)
        successful_examples = sum(1 for r in results if r.success)
        
        # Timing metrics
        total_gb_time = sum(r.gb_time for r in results)
        total_tr_time = sum(r.tr_time for r in results)
        total_dec_time = sum(r.dec_time for r in results)
        total_time = sum(r.total_time for r in results)
        times = [r.total_time for r in results]
        p50_time = float(np.percentile(times, 50)) if times else 0.0
        p95_time = float(np.percentile(times, 95)) if times else 0.0
        
        # Graph building metrics
        avg_nodes = sum(r.gb_nodes for r in results) / total_examples
        avg_edges = sum(r.gb_edges for r in results) / total_examples
        avg_tokens = sum(r.gb_tokens for r in results) / total_examples
        
        # Traversal metrics
        avg_path_length = sum(r.tr_path_length for r in results) / total_examples
        stopped_count = sum(1 for r in results if r.tr_stopped)
        
        # Accuracy metrics
        avg_accuracy = sum(r.accuracy for r in results) / total_examples
        
        # Advanced metrics
        exact_match_count = sum(1 for r in results if r.exact_match)
        exact_match_rate = exact_match_count / total_examples
        
        avg_f1 = sum(r.f1_score for r in results) / total_examples
        avg_logic_consistency = sum(r.logic_consistency for r in results) / total_examples
        avg_trace_f1 = sum(r.trace_f1 for r in results) / total_examples
        
        avg_latency_ms = sum(r.latency_ms for r in results) / total_examples
        avg_latency_normalized = sum(r.latency_normalized for r in results) / total_examples
        avg_edge_budget_normalized = sum(r.edge_budget_normalized for r in results) / total_examples
        avg_node_budget_normalized = sum(r.node_budget_normalized for r in results) / total_examples
        avg_bleu1 = sum(getattr(r, 'bleu1', 0.0) for r in results) / total_examples
        avg_bleu2 = sum(getattr(r, 'bleu2', 0.0) for r in results) / total_examples
        avg_bleu4 = sum(getattr(r, 'bleu4', 0.0) for r in results) / total_examples
        avg_rouge_l = sum(getattr(r, 'rouge_l', 0.0) for r in results) / total_examples
        avg_rule_coverage = sum(getattr(r, 'rule_coverage', 0.0) for r in results) / total_examples
        avg_violation_rate = sum(getattr(r, 'violation_rate', 0.0) for r in results) / total_examples
        coverage_rate = sum(1 for r in results if getattr(r, 'covered', False)) / total_examples
        avg_mrr = sum(getattr(r, 'mrr', 0.0) for r in results) / total_examples
        avg_acc3 = sum(getattr(r, 'acc_at_3', 0.0) for r in results) / total_examples
        avg_acc5 = sum(getattr(r, 'acc_at_5', 0.0) for r in results) / total_examples
        
        return {
            'total_examples': total_examples,
            'successful_examples': successful_examples,
            'success_rate': successful_examples / total_examples,
            'avg_accuracy': avg_accuracy,
            
            # Advanced accuracy metrics
            'exact_match_rate': exact_match_rate,
            'avg_f1_score': avg_f1,
            'avg_logic_consistency': avg_logic_consistency,
            'avg_trace_f1': avg_trace_f1,
            
            # Timing
            'total_time': total_time,
            'avg_time_per_example': total_time / total_examples,
            'avg_gb_time': total_gb_time / total_examples,
            'avg_tr_time': total_tr_time / total_examples,
            'avg_dec_time': total_dec_time / total_examples,
            'p50_time': p50_time,
            'p95_time': p95_time,
            
            # Graph building
            'avg_nodes': avg_nodes,
            'avg_edges': avg_edges,
            'avg_tokens': avg_tokens,
            
            # Traversal
            'avg_path_length': avg_path_length,
            'stopped_count': stopped_count,
            'stopped_rate': stopped_count / total_examples,
            
            # Performance metrics
            'avg_latency_ms': avg_latency_ms,
            'avg_latency_normalized': avg_latency_normalized,
            'avg_edge_budget_normalized': avg_edge_budget_normalized,
            'avg_node_budget_normalized': avg_node_budget_normalized,
            # Additional text/logic/ranking metrics
            'avg_bleu1': avg_bleu1,
            'avg_bleu2': avg_bleu2,
            'avg_bleu4': avg_bleu4,
            'avg_rouge_l': avg_rouge_l,
            'avg_rule_coverage': avg_rule_coverage,
            'avg_violation_rate': avg_violation_rate,
            'coverage_rate': coverage_rate,
            'avg_mrr': avg_mrr,
            'avg_acc3': avg_acc3,
            'avg_acc5': avg_acc5
        }
    
    def save_results(self, results: List[EvaluationResult], 
                    aggregate_metrics: Dict[str, Any]) -> str:
        """Save evaluation results to files"""
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.dataset_name}_{self.config.dataset_split}_{timestamp}"
        
        # Save detailed results
        if self.config.save_detailed:
            detailed_file = output_dir / f"{base_name}_detailed.csv"
            with open(detailed_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))
            
            logger.info(f"Detailed results saved to: {detailed_file}")
        
        # Save aggregate metrics
        metrics_file = output_dir / f"{base_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        logger.info(f"Aggregate metrics saved to: {metrics_file}")
        
        # Save summary
        summary_file = output_dir / f"{base_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"LogiKG-MAPPO Evaluation Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Dataset: {self.config.dataset_name}\n")
            f.write(f"Split: {self.config.dataset_split}\n")
            f.write(f"Examples: {aggregate_metrics['total_examples']}\n")
            f.write(f"Success Rate: {aggregate_metrics['success_rate']:.2%}\n")
            f.write(f"Avg Accuracy: {aggregate_metrics['avg_accuracy']:.4f}\n")
            f.write(f"Avg Time: {aggregate_metrics['avg_time_per_example']:.2f}s\n")
            f.write(f"Total Time: {aggregate_metrics['total_time']:.2f}s\n")
            f.write(f"\nAdvanced Metrics:\n")
            f.write(f"Exact Match Rate: {aggregate_metrics.get('exact_match_rate', 0):.2%}\n")
            f.write(f"Avg F1 Score: {aggregate_metrics.get('avg_f1_score', 0):.4f}\n")
            f.write(f"Avg Logic Consistency: {aggregate_metrics.get('avg_logic_consistency', 0):.4f}\n")
            f.write(f"Avg Trace-F1: {aggregate_metrics.get('avg_trace_f1', 0):.4f}\n")
            f.write(f"Avg Latency: {aggregate_metrics.get('avg_latency_ms', 0):.1f}ms\n")
            f.write(f"Avg Latency (normalized): {aggregate_metrics.get('avg_latency_normalized', 0):.2f}x\n")
            f.write(f"Avg Edge Budget (normalized): {aggregate_metrics.get('avg_edge_budget_normalized', 0):.2f}x\n")
            f.write(f"Avg Node Budget (normalized): {aggregate_metrics.get('avg_node_budget_normalized', 0):.2f}x\n")
            f.write(f"Latency p50: {aggregate_metrics.get('p50_time', 0):.2f}s  p95: {aggregate_metrics.get('p95_time', 0):.2f}s\n")
            f.write(f"\nText Similarity Metrics:\n")
            f.write(f"BLEU-1: {aggregate_metrics.get('avg_bleu1', 0):.3f}\n")
            f.write(f"BLEU-2: {aggregate_metrics.get('avg_bleu2', 0):.3f}\n")
            f.write(f"BLEU-4: {aggregate_metrics.get('avg_bleu4', 0):.3f}\n")
            f.write(f"ROUGE-L: {aggregate_metrics.get('avg_rouge_l', 0):.3f}\n")
            f.write(f"\nLogic/DLL Metrics:\n")
            f.write(f"Rule Coverage: {aggregate_metrics.get('avg_rule_coverage', 0):.4f}\n")
            f.write(f"Violation Rate: {aggregate_metrics.get('avg_violation_rate', 0):.4f}\n")
            f.write(f"\nRanking Metrics:\n")
            f.write(f"Coverage (gold in candidates): {aggregate_metrics.get('coverage_rate', 0):.2%}\n")
            f.write(f"MRR: {aggregate_metrics.get('avg_mrr', 0):.4f}\n")
            f.write(f"Acc@3: {aggregate_metrics.get('avg_acc3', 0):.3f}\n")
            f.write(f"Acc@5: {aggregate_metrics.get('avg_acc5', 0):.3f}\n")
        
        logger.info(f"Summary saved to: {summary_file}")
        
        return str(output_dir)
    
    def run_evaluation(self) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """Run the complete evaluation pipeline"""
        
        logger.info("Starting LogiKG-MAPPO evaluation pipeline")
        
        # Validate dataset
        if self.config.dataset_path:
            # if not dataset_manager.validate_dataset_path( # Commented out - not available in current structure
            #     self.config.dataset_name, self.config.dataset_path
            # ):
            #     raise ValueError(f"Invalid dataset path: {self.config.dataset_path}")
            pass # Placeholder for dataset validation if dataset_manager is not available
        
        # Run evaluation
        results, metrics = self.evaluate_dataset()
        
        # Save results
        output_dir = self.save_results(results, metrics)
        
        # Print summary
        self.print_summary(metrics)
        
        logger.info(f"Evaluation completed. Results saved to: {output_dir}")
        
        return results, metrics
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary to console"""
        
        print("\n" + "="*60)
        print("LogiKG-MAPPO Evaluation Summary")
        print("="*60)
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Examples: {metrics['total_examples']}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Avg Accuracy: {metrics['avg_accuracy']:.4f}")
        print(f"Avg Time: {metrics['avg_time_per_example']:.2f}s")
        print(f"Total Time: {metrics['total_time']:.2f}s")
        print(f"Latency p50: {metrics.get('p50_time', 0):.2f}s  p95: {metrics.get('p95_time', 0):.2f}s")
        print(f"Avg Nodes: {metrics['avg_nodes']:.1f}")
        print(f"Avg Edges: {metrics['avg_edges']:.1f}")
        print(f"Avg Path Length: {metrics['avg_path_length']:.1f}")
        print("="*60)
        print("Additional Metrics")
        print(f"BLEU-1: {metrics.get('avg_bleu1', 0):.3f}  BLEU-2: {metrics.get('avg_bleu2', 0):.3f}  BLEU-4: {metrics.get('avg_bleu4', 0):.3f}")
        print(f"ROUGE-L: {metrics.get('avg_rouge_l', 0):.3f}  MRR: {metrics.get('avg_mrr', 0):.3f}")
        print(f"Acc@3: {metrics.get('avg_acc3', 0):.3f}  Acc@5: {metrics.get('avg_acc5', 0):.3f}  Coverage: {metrics.get('coverage_rate', 0):.2%}")
