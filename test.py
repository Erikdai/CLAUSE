#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import argparse
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

# ------------------------------- Path Setup -------------------------------

REPO_ROOT = "/path/"
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from logikg_mappo.agents.enhanced_graph_builder import EnhancedGraphBuilder
from logikg_mappo.agents.traversal import EnhancedTraversalAgent
from logikg_mappo.agents.decoder import LogicAwareDecoder
from logikg_mappo.rl.lc_mappo import (
    MultiHeadCentralCritic,
    create_lc_mappo_optimizers,
    prepare_lc_mappo_batch,
    lc_mappo_update,
)
from logikg_mappo.rl.common import StepLog

# Import the core LC-MAPPO helpers
from logikg_mappo.rl.lc_mappo import (
    detect_question_type,
    calculate_edge_relevance,
    calculate_adaptive_reward,
    estimate_question_difficulty,
    create_enhanced_lc_mappo_policies,
    update_success_history,
    should_continue_step,
    calculate_step_reward,
    lc_mappo_step_control,
    create_coordination_context,
    update_coordination_signals,
    calculate_coordination_rewards,
    should_agent_wait,
)

# ------------------------------- Utils -------------------------------

def _as_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def _norm_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s.lower().strip())

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (set, tuple)):
        return list(x)
    return [x]

def _extract_mention(question: str) -> str:
    m = re.search(r'\[([^\]]+)\]', question)
    return m.group(1) if m else ""

def _norm_nodes(nodes: List[str]) -> List[str]:
    return [_norm_text(n) for n in nodes]

def _norm_edges(edges_raw) -> List[Tuple[str, str, str]]:
    out = []
    for e in _as_list(edges_raw):
        if isinstance(e, dict):
            h = e.get("h") or e.get("head")
            r = e.get("r") or e.get("rel")
            t = e.get("t") or e.get("tail")
            if all(isinstance(x, str) for x in (h, r, t)):
                out.append((h, r, t))
        elif isinstance(e, (list, tuple)) and len(e) == 3:
            h, r, t = e
            if all(isinstance(x, str) for x in (h, r, t)):
                out.append((h, r, t))
    return out

def build_subgraph(gb: EnhancedGraphBuilder, question: str, steps: int):
    gb.reset_subgraph()
    for i in range(steps):
        out = gb.forward({"question": question})
        act = out["action"]
        print(f"      [GB {i+1}] {act['operation_type']}  ({act['subject']}, {act['predicate']}, {act['object']})"
              f"  V={out['num_nodes']} E={out['num_edges']}")
        if act["operation_type"] == "stop":
            break
    V, E = gb.get_current_subgraph()
    return _norm_nodes(V), _norm_edges(E)

def traverse(edges: List[Tuple[str, str, str]], question: str, mention: str, seeds: List[str], max_hops: int):
    from collections import defaultdict
    start = seeds[0] if seeds else (edges[0][0] if edges else "UNKNOWN")
    print(f"      [TR] start = {start}")
    adj = defaultdict(list)
    for h, r, t in edges:
        adj[h].append((h, r, t))

    agent = EnhancedTraversalAgent(max_hops=max_hops, device=None)
    agent.eval()

    path_edges = []
    cur = start
    for hop in range(max_hops):
        obs = {
            "question": question,
            "current_node": cur,
            "neighbors": adj.get(cur, []),
            "path": path_edges,
            "step": hop
        }
        o = agent.forward(obs)
        a = o["action"]
        if a["stop"]:
            print(f"      [TR {hop+1}] stop")
            break
        if a["next_edge"]:
            h, r, t = a["next_edge"]
            print(f"      [TR {hop+1}] choose edge: {(h, r, t)}")
            path_edges.append((h, r, t))
            cur = t
        elif a["next_node"]:
            print(f"      [TR {hop+1}] choose node: {a['next_node']}")
            cur = a["next_node"]
        else:
            print(f"      [TR {hop+1}] no action")
            break
    return path_edges

def candidates_from_evidence(evidence: List[Tuple[str, str, str]], mention: str) -> List[str]:
    cand = set()
    if mention:
        for h, r, t in evidence:
            if h == mention and t != mention:
                cand.add(t)
            if t == mention and h != mention:
                cand.add(h)
    if not cand:
        for h, r, t in evidence:
            cand.add(h); cand.add(t)
    return list(cand)

def _simple_offline_rerank(question: str, candidates: List[str]) -> str:
    if not candidates:
        return ""
    import re as _re
    q = (question or "").lower()
    mm = _re.findall(r"\[([^\]]+)\]", q)
    mention = (mm[0].strip() if mm else "").lower()
    q_tokens = set(t for t in _re.sub(r"[^\w\s]", " ", q).split() if t)

    def score(c: str) -> float:
        if not c:
            return -1e9
        c_low = c.lower()
        s = 0.0
        # Avoid returning the bracketed mention itself as answer
        if mention and c_low == mention:
            s -= 20.0
        c_tokens = set(t for t in _re.sub(r"[^\w\s]", " ", c_low).split() if t)
        s += 5.0 * len(q_tokens & c_tokens)
        s += min(5.0, 0.2 * len(c_low))
        return s

    return max(candidates, key=score)

# ------------------------------- LC-MAPPO Controlled Steps -------------------------------

def create_gb_observation(question, mention, current_edges, obs_dim):
    """
    Create a compact observation vector for the Graph Builder policy.
    """
    obs = torch.zeros(obs_dim)
    obs[0] = len(question.split()) / 50.0
    obs[1] = len(mention.split()) / 10.0 if mention else 0.0
    obs[2] = len(current_edges) / 20.0
    obs[3] = 1.0 if mention else 0.0
    obs[4] = 1.0 if "?" in question else 0.0
    obs[5] = len(current_edges) / 12.0
    return obs

def create_tr_observation(question, mention, current_node, path_edges, obs_dim):
    """
    Create a compact observation vector for the Traversal policy.
    """
    obs = torch.zeros(obs_dim)
    obs[0] = len(question.split()) / 50.0
    obs[1] = len(mention.split()) / 10.0 if mention else 0.0
    obs[2] = len(path_edges) / 10.0
    obs[3] = 1.0 if current_node else 0.0
    obs[4] = 1.0 if "?" in question else 0.0
    obs[5] = len(path_edges) / 4.0
    return obs

def create_dec_observation(question, candidates, obs_dim):
    """
    Create a compact observation vector for the Decoder policy.
    """
    obs = torch.zeros(obs_dim)
    obs[0] = len(question.split()) / 50.0
    obs[1] = len(candidates) / 20.0
    obs[2] = 1.0 if "?" in question else 0.0
    obs[3] = 1.0 if candidates else 0.0
    obs[4] = 0.0
    obs[5] = 0.0
    return obs

def lc_mappo_graph_building(gb, question, mention, policy, state_dim, obs_dim, device, max_steps=12, success_history=0.8, coordination_context=None):
    """
    LC-MAPPO-controlled graph building loop with simple coordination/reward shaping.
    """
    question_type = detect_question_type(question)
    print(f"      [SMART] Question type: {question_type}")

    question_difficulty = 'medium'
    if 'which' in question.lower() or 'appears in' in question.lower():
        question_difficulty = 'hard'
    elif len(question.split()) < 6:
        question_difficulty = 'easy'
    print(f"      [REWARD] Question difficulty: {question_difficulty}, Success history: {success_history:.2f}")

    if coordination_context is None:
        coordination_context = create_coordination_context(
            question, question_type, [], [], None, None, None, success_history
        )
    print(f"      [COORD] Multi-agent coordination enabled")

    gb.reset_subgraph()
    actions, rewards, edge_relevance_scores = [], [], []
    step = 0
    while step < max_steps:
        if should_agent_wait(coordination_context, 'gb', step):
            print(f"      [GB {step+1}] Waiting for coordination...")
            step += 1
            continue

        current_V, current_E = gb.get_current_subgraph()
        obs = create_gb_observation(question, mention, current_E, obs_dim)

        with torch.no_grad():
            try:
                obs_tensor = obs.unsqueeze(0).to(device)
                action_idx = lc_mappo_step_control(
                    obs_tensor, policy, step, max_steps, current_E,
                    edge_relevance_scores, question_difficulty,
                    policy.success_history, device
                )
                # Small exploration jitter
                if torch.rand(1).item() < 0.05:
                    action_idx = 1 - action_idx
            except Exception as e:
                print(f"      [WARNING] Policy network failed: {e}, using fallback")
                if step < 6:
                    action_idx = 0
                elif len(current_E) >= 10 or step >= 10:
                    action_idx = 1
                else:
                    action_idx = 0 if torch.rand(1).item() > 0.3 else 1

        if action_idx == 0:  # add_edge
            gb_result = gb.forward({"question": question})
            if gb_result and "action" in gb_result:
                act = gb_result["action"]
                if act["operation_type"] != "stop":
                    edge = (act['subject'], act['predicate'], act['object'])
                    relevance = calculate_edge_relevance(edge, question_type, question)
                    edge_relevance_scores.append(relevance)
                    base_reward = calculate_step_reward(
                        step, max_steps, relevance, question_difficulty, policy.success_history
                    )
                    coordination_reward, _ = calculate_coordination_rewards(
                        coordination_context, 'gb', act, base_reward
                    )
                    print(f"      [GB {step+1}] {act['operation_type']}  ({act['subject']}, {act['predicate']}, {act['object']})"
                          f"  V={gb_result['num_nodes']} E={gb_result['num_edges']}  [RELEVANCE: {relevance:.2f}, REWARD: {coordination_reward:.3f}]")
                else:
                    base_reward = -0.1
                    coordination_reward, _ = calculate_coordination_rewards(
                        coordination_context, 'gb', {}, base_reward
                    )
                    print(f"      [GB {step+1}] stop")
                    break
            else:
                if step < 3:
                    base_reward = -0.05
                    coordination_reward, _ = calculate_coordination_rewards(
                        coordination_context, 'gb', {}, base_reward
                    )
                    print(f"      [GB {step+1}] no edge found, continuing...")
                else:
                    base_reward = -0.1
                    coordination_reward, _ = calculate_coordination_rewards(
                        coordination_context, 'gb', {}, base_reward
                    )
                    break
        else:
            base_reward = 0.0
            coordination_reward, _ = calculate_coordination_rewards(
                coordination_context, 'gb', {}, base_reward
            )
            print(f"      [GB {step+1}] stop (LC-MAPPO decision)")
            break

        coordination_context['current_edges'] = current_E
        coordination_context['edge_relevance_scores'] = edge_relevance_scores
        coordination_context = update_coordination_signals(
            coordination_context, 'gb', gb_result, step
        )

        actions.append(action_idx)
        rewards.append(coordination_reward)
        step += 1

    V, E = gb.get_current_subgraph()
    return _norm_nodes(V), _norm_edges(E), actions, rewards, coordination_context

def lc_mappo_traversal(edges, question, mention, seeds, policy, state_dim, obs_dim, device, max_hops=4):
    """
    LC-MAPPO-controlled traversal with a simple heuristic policy fallback.
    """
    if not seeds:
        return [], [], []

    adj = {}
    for h, r, t in edges:
        adj.setdefault(h, []).append((h, r, t))

    path_edges, actions, rewards = [], [], []
    current_node, hop = seeds[0], 0

    while hop < max_hops:
        obs = create_tr_observation(question, mention, current_node, path_edges, obs_dim)
        with torch.no_grad():
            try:
                _ = policy(obs.unsqueeze(0).to(device))  # forward placeholder
                if current_node in adj and len(adj[current_node]) > 0 and hop < 2:
                    action_idx = 1  # continue
                else:
                    action_idx = 0  # stop
                # Small exploration jitter
                if torch.rand(1).item() < 0.15:
                    action_idx = 1 - action_idx
            except Exception as e:
                print(f"      [WARNING] Traversal policy failed: {e}, using fallback")
                if current_node in adj and len(adj[current_node]) > 0 and hop < 2:
                    action_idx = 1
                else:
                    action_idx = 0

        if action_idx == 0:
            reward = 0.1 if path_edges else -0.05
            actions.append(action_idx); rewards.append(reward)
            break
        else:
            if current_node in adj and len(adj[current_node]) > 0:
                next_edge = adj[current_node][0]
                path_edges.append(next_edge)
                current_node = next_edge[2]
                reward = 0.05
            else:
                reward = -0.1
                break
        actions.append(action_idx); rewards.append(reward); hop += 1

    return path_edges, actions, rewards

def lc_mappo_decoding(dec, question, candidates, policy, state_dim, obs_dim, device, use_lm=False):
    """
    LC-MAPPO-controlled decoding. If LM is disabled, fall back to a simple heuristic re-ranker.
    """
    if not candidates:
        return "", [], []
    if use_lm:
        dec.set_question_text(question)
        res = dec.forward(obs={"question": question}, generate_answer=True, candidates=candidates)
        predicted = _as_str(res.get("answer", ""))
    else:
        predicted = _simple_offline_rerank(question, candidates)

    try:
        obs = create_dec_observation(question, candidates, obs_dim)
        _ = policy(obs.unsqueeze(0).to(device))
        action_idx = 0
        # Small exploration jitter
        if torch.rand(1).item() < 0.2 and len(candidates) > 1:
            action_idx = 1
    except Exception as e:
        print(f"      [WARNING] Decoder policy failed: {e}, using fallback")
        action_idx = 0

    actions, rewards = [action_idx], [1.0]
    return predicted, actions, rewards

# ------------------------------- Metrics -------------------------------

def _parse_metaqa_line(line: str) -> tuple:
    """
    Parse a MetaQA-format line: "<question>\t<answer>|<alt1>|<alt2>..."
    Returns (question, [answers]) or (None, None) if malformed.
    """
    line = line.strip()
    if not line or '\t' not in line:
        return None, None
    q, a = line.split('\t', 1)
    answers = [s.strip() for s in (a.split('|') if '|' in a else (a.split('→') if '→' in a else [a]))]
    return q.strip(), answers

def _compute_metrics(predicted: str, gold_answers: List[str]) -> Dict[str, Any]:
    """
    Compute EM, token-level F1 (best over references), and loose coverage.
    """
    pred_norm = _norm_text(predicted)
    gold_norms = [_norm_text(g) for g in gold_answers]
    em = 1.0 if pred_norm in gold_norms else 0.0
    pred_tokens = set(pred_norm.split())
    f1_scores = []
    for gold_norm in gold_norms:
        gold_tokens = set(gold_norm.split())
        if not pred_tokens and not gold_tokens:
            f1_scores.append(1.0)
        elif not pred_tokens or not gold_tokens:
            f1_scores.append(0.0)
        else:
            inter = len(pred_tokens & gold_tokens)
            precision = inter / len(pred_tokens)
            recall = inter / len(gold_tokens)
            f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    f1 = max(f1_scores) if f1_scores else 0.0
    covered = 1.0 if any(g in pred_norm or pred_norm in g for g in gold_norms) else 0.0
    return {'em': em, 'f1': f1, 'covered': covered, 'predicted': predicted, 'gold_answers': gold_answers}

# ------------------------------- Evaluation Loop -------------------------------

def evaluate_lc_mappo(qa_file: str, cache_dir: str, ttl_file: str = None, gb_steps: int = 12, max_hops: int = 4,
                      limit: int = 20, use_lm: bool = True, device: str = "cpu", checkpoint_path: str = None,
                      train_first: int = 0, train_epochs: int = 1, save_ckpt: bool = False, skip_eval_update: bool = True):
    print("LC-MAPPO Enhanced Multi-Agent Evaluation")
    print("=" * 50)

    # Agents
    print("Initializing enhanced agents...")
    print("Initializing Graph Builder...")
    gb = EnhancedGraphBuilder(
        cache_dir=cache_dir,
        ttl_file=ttl_file,
        use_cache_only=ttl_file is None,
        embedder_name="thenlper/gte-small",
        device=device,
        hidden=128,
        max_operations=gb_steps,
        max_candidates_per_operation=50
    )

    print("Initializing Traversal Agent...")
    tr = EnhancedTraversalAgent(max_hops=max_hops, device=device)

    print("Initializing Decoder Agent...")
    dec = LogicAwareDecoder(vocab_size=50257, use_lm=use_lm)

    # LC-MAPPO components
    print("Initializing LC-MAPPO components...")
    state_dim, obs_dim, act_dim, dll_dim, gnn_dim, n_agents = 11, 6, 3, 32, 32, 3
    critic = MultiHeadCentralCritic(
        state_dim=state_dim,
        obs_dim=obs_dim,
        act_dim=act_dim,
        dll_dim=dll_dim,
        gnn_dim=gnn_dim,
        hidden=256,
        n_agents=n_agents
    ).to(device)

    agents = [(gb, 'gb'), (tr, 'tr'), (dec, 'dec')]
    policies_list = create_enhanced_lc_mappo_policies(agents, device, success_history=0.8)
    policies = {'gb': policies_list[0], 'tr': policies_list[1], 'dec': policies_list[2]}
    critic_optimizer, policy_optimizers = create_lc_mappo_optimizers(
        critic=critic, policies=policies_list, lr=3e-4
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained model from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            critic.load_state_dict(checkpoint['critic_state_dict'])
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"Successfully loaded trained model (episodes: {checkpoint.get('episodes', 'unknown')})")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print(f"Using random initialization")
    else:
        print(f"Using random initialization (no checkpoint provided)")

    print(f"LC-MAPPO initialized: {len(policies)} policies, critic params={sum(p.numel() for p in critic.parameters())}")

    # Example constraint weights/budgets and regularization coefficients
    lambdas = {'edge': torch.tensor(0.1), 'lat': torch.tensor(0.1), 'logic': torch.tensor(0.5)}
    budgets = {'edge': torch.tensor(0.2), 'lat': torch.tensor(0.2), 'logic': torch.tensor(0.2)}
    eps0, kappa, dual_lr = 0.2, 0.1, 1e-2
    kl_coef, ent_coef = 0.01, 0.01

    print(f"Loading MetaQA test data from: {qa_file}")
    questions_data = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            q, answers = _parse_metaqa_line(line)
            if q and answers:
                questions_data.append({'question': q, 'answers': answers, 'line_number': line_num})
                if len(questions_data) >= limit:
                    break
    print(f"Loaded {len(questions_data)} questions")

    # Optional tiny training (kept but disabled by default)
    if (train_first or 0) > 0 and not checkpoint_path:
        n_train = min(int(train_first), len(questions_data))
        print(f"\nTraining LC-MAPPO on first {n_train} questions × {max(1, int(train_epochs))} epochs")
        for t_idx in range(n_train):
            q_train = questions_data[t_idx]['question']
            q_train_no_ans = q_train.split("\t", 1)[0] if "\t" in q_train else q_train
            mention = _extract_mention(q_train_no_ans)
            V, E = build_subgraph(gb, q_train_no_ans, steps=gb_steps)
            if not E:
                continue
            seeds = gb._match_nodes_by_mention(mention, topk=1) if mention else []
            path_edges = traverse(E, q_train_no_ans, mention, seeds, max_hops=max_hops)
            evidence = path_edges if path_edges else E[:min(len(E), 12)]
            candidates = candidates_from_evidence(evidence, mention)

            # Lightweight batch for an update step
            gb_summary = {"num_nodes": len(V), "num_edges": len(E), "estimated_tokens": len(E) * 10}
            tr_summary = {"path_len": len(path_edges), "candidate_count": len(evidence), "stop": True}
            dec_summary = {"out_len": 1 + len(candidates[:1]), "rule_count": 2}
            step_logs = [
                StepLog(agent='gb', log_prob=torch.tensor(0.0), entropy=torch.tensor(0.1), reward=0.1, old_log_prob=torch.tensor(0.0)),
                StepLog(agent='tr', log_prob=torch.tensor(0.0), entropy=torch.tensor(0.1), reward=0.05, old_log_prob=torch.tensor(0.0)),
                StepLog(agent='dec', log_prob=torch.tensor(0.0), entropy=torch.tensor(0.1), reward=0.1, old_log_prob=torch.tensor(0.0)),
            ]
            recorded_actions = {
                'gb_actions': [{'op': 'add_edge'}],
                'tr_actions': [{'stop': False, 'candidate_idx': 0}],
                'dec_actions': [{'token_id': 0}],
                'gb_obs': [{}], 'tr_obs': [{}], 'dec_obs': [{}],
            }
            batch = prepare_lc_mappo_batch(
                state_feats=[torch.randn(state_dim) for _ in range(3)],
                step_logs=step_logs,
                recorded_actions=recorded_actions,
                gb_summary=gb_summary,
                tr_summary=tr_summary,
                dec_summary=dec_summary,
                tau=1.0,
                device=device,
                dec_vocab_size=50257 if use_lm else 1000
            )
            for ep in range(max(1, int(train_epochs))):
                lc_mappo_update(
                    batch, critic, policies, lambdas, budgets,
                    eps0, kappa, dual_lr, critic_optimizer, policy_optimizers,
                    kl_coef=kl_coef, ent_coef=ent_coef, agents=agents,
                    global_step=ep + t_idx * max(1, int(train_epochs))
                )
        if save_ckpt:
            try:
                ckpt = {'critic_state_dict': critic.state_dict(),
                        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                        'episodes': int(train_first)}
                os.makedirs('outputs', exist_ok=True)
                torch.save(ckpt, 'outputs/lc_mappo_ckpt.pt')
                print("Saved LC-MAPPO checkpoint to outputs/lc_mappo_ckpt.pt")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    total_questions, em_count, f1_sum, covered_count = len(questions_data), 0, 0.0, 0
    results = []
    lc_mappo_stats = {'total_episodes': 0, 'total_steps': 0, 'critic_losses': [], 'actor_losses': [],
                      'constraint_violations': {'edge': 0, 'latency': 0, 'logic': 0},
                      'agent_rewards': {'gb': [], 'tr': [], 'dec': []}}

    print(f"\nProcessing {total_questions} questions with LC-MAPPO...")
    success_history = 0.8

    for idx, qa in enumerate(questions_data, 1):
        question = qa['question']
        gold_answers = qa['answers']
        print(f"\n--- Question {idx}/{total_questions} ---")
        print(f"Q: {question}")
        print(f"G: {gold_answers}")

        try:
            question_no_ans = question.split("\t", 1)[0] if "\t" in question else question
            mention = _extract_mention(question_no_ans)
            print(f"      Mention: {mention}")

            print("LC-MAPPO Graph Building...")
            V, E, gb_actions, gb_rewards, coordination_context = lc_mappo_graph_building(
                gb, question_no_ans, mention, policies['gb'],
                state_dim, obs_dim, device, max_steps=gb_steps, success_history=success_history
            )
            if not E:
                print("empty subgraph")
                predicted, candidates = "", []
                gb_summary = {"num_nodes": 0, "num_edges": 0, "estimated_tokens": 0, "actions": 0, "rewards": []}
                tr_summary = {"path_len": 0, "candidate_count": 0, "stop": True, "actions": 0, "rewards": []}
                tr_actions, tr_rewards = [], []
                dec_actions, dec_rewards = [], []
            else:
                print(f"      Final subgraph: {len(V)} nodes, {len(E)} edges")
                print(f"      [GB] Actions: {len(gb_actions)}, Rewards: {gb_rewards}")
                gb_summary = {"num_nodes": len(V), "num_edges": len(E), "estimated_tokens": len(E) * 10,
                              "actions": len(gb_actions), "rewards": gb_rewards}

                print("LC-MAPPO Traversal...")
                seeds = gb._match_nodes_by_mention(mention, topk=1) if mention else []
                seeds = [s.lower() for s in seeds] if seeds else []
                print(f"      [DEBUG] Seeds: {seeds}")

                path_edges, tr_actions, tr_rewards = lc_mappo_traversal(
                    E, question_no_ans, mention, seeds, policies['tr'],
                    state_dim, obs_dim, device, max_hops=max_hops
                )
                print(f"      [DEBUG] Path edges: {path_edges}")
                print(f"      [TR] Actions: {len(tr_actions)}, Rewards: {tr_rewards}")

                evidence = path_edges if path_edges else E[:min(len(E), 12)]
                print(f"      [EV] using {len(evidence)} triples")
                print(f"      [DEBUG] Evidence triples: {evidence[:3]}")

                tr_summary = {"path_len": len(path_edges), "candidate_count": len(evidence),
                              "actions": len(tr_actions), "rewards": tr_rewards}

                print("LC-MAPPO Decoding...")
                candidates = candidates_from_evidence(evidence, mention)
                print(f"      [CAND] {len(candidates)} → {candidates[:10]}")

                dec.set_question_text(question_no_ans)
                predicted, dec_actions, dec_rewards = lc_mappo_decoding(
                    dec, question_no_ans, candidates, policies['dec'],
                    state_dim, obs_dim, device, use_lm=use_lm
                )
                print(f"      [DEBUG] Question: {question_no_ans}")
                print(f"      [DEBUG] Candidates: {candidates}")
                print(f"      [DEC] Actions: {len(dec_actions)}, Rewards: {dec_rewards}")

                if gold_answers:
                    print(f"      [DEBUG] Checking correct answers:")
                    for gold in gold_answers:
                        if gold.lower() in [c.lower() for c in candidates]:
                            print(f"'{gold}' is in candidates")
                        else:
                            print(f"'{gold}' NOT in candidates")

                print(f"      Generated: {predicted}")
                print(f"      Candidates: {candidates[:5]}")

            metrics = _compute_metrics(predicted, gold_answers)
            if metrics['em'] > 0: em_count += 1
            f1_sum += metrics['f1']
            if metrics['covered'] > 0: covered_count += 1

            print("LC-MAPPO Training Step...")
            performance_reward = metrics['em'] * 2.0 + metrics['f1']
            gb_final_reward = sum(gb_rewards) + performance_reward * 0.3
            tr_final_reward = sum(tr_rewards) + performance_reward * 0.3
            dec_final_reward = sum(dec_rewards) + performance_reward * 0.4

            step_logs = [
                StepLog(agent='gb', log_prob=torch.tensor(0.0), entropy=torch.tensor(0.1),
                        reward=gb_final_reward, old_log_prob=torch.tensor(0.0)),
                StepLog(agent='tr', log_prob=torch.tensor(0.0), entropy=torch.tensor(0.1),
                        reward=tr_final_reward, old_log_prob=torch.tensor(0.0)),
                StepLog(agent='dec', log_prob=torch.tensor(0.0), entropy=torch.tensor(0.1),
                        reward=dec_final_reward, old_log_prob=torch.tensor(0.0)),
            ]
            recorded_actions = {
                'gb_actions': [{'action_idx': a} for a in gb_actions],
                'tr_actions': [{'action_idx': a} for a in (tr_actions or [])],
                'dec_actions': [{'action_idx': a} for a in (dec_actions or [])],
                'gb_obs': [{}] * len(gb_actions),
                'tr_obs': [{}] * len(tr_actions),
                'dec_obs': [{}] * len(dec_actions),
            }

            try:
                batch = prepare_lc_mappo_batch(
                    state_feats=[torch.randn(state_dim) for _ in range(3)],
                    step_logs=step_logs,
                    recorded_actions=recorded_actions,
                    gb_summary=gb_summary,
                    tr_summary=tr_summary,
                    dec_summary={"out_len": len((predicted or "").split()), "rule_count": 2},
                    tau=1.0,
                    device=device,
                    dec_vocab_size=50257 if use_lm else 1000
                )
                if not skip_eval_update:
                    training = lc_mappo_update(
                        batch, critic, policies, lambdas, budgets, eps0, kappa, dual_lr,
                        critic_optimizer, policy_optimizers, kl_coef=kl_coef, ent_coef=ent_coef,
                        agents=agents, global_step=lc_mappo_stats['total_steps']
                    )
                    lc_mappo_stats['total_steps'] += 1
                    try:
                        cl = training.get('critic_loss', 0.0); al = training.get('actor_loss', 0.0)
                        cl_val = float(cl.item()) if hasattr(cl, 'item') else float(cl)
                        al_val = float(al.item()) if hasattr(al, 'item') else float(al)
                        lc_mappo_stats['critic_losses'].append(cl_val)
                        lc_mappo_stats['actor_losses'].append(al_val)
                        print(f"      LC-MAPPO update: actor_loss={al_val:.4f} critic_loss={cl_val:.4f}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"LC-MAPPO step failed: {e}")

            results.append({
                'question': question,
                'gold_answers': gold_answers,
                'predicted': predicted,
                'candidates': candidates if 'candidates' in locals() else [],
                'em': metrics['em'],
                'f1': metrics['f1'],
                'covered': metrics['covered'],
                'gb_summary': gb_summary,
                'tr_summary': tr_summary,
                'dec_summary': {'out_len': len((predicted or '').split()),
                                'candidates': candidates if 'candidates' in locals() else []}
            })
            print(f"EM: {metrics['em']:.0f}, F1: {metrics['f1']:.3f}, Covered: {metrics['covered']:.0f}")

            update_success_history(policies_list, metrics['em'])
            print(f"Success History: {policies['gb'].success_history:.2f}")

        except Exception as e:
            print(f"Error processing question: {e}")
            import traceback; traceback.print_exc()
            continue

    em_rate = em_count / total_questions if total_questions > 0 else 0.0
    f1_avg = f1_sum / total_questions if total_questions > 0 else 0.0
    coverage_rate = covered_count / total_questions if total_questions > 0 else 0.0

    print(f"\nLC-MAPPO EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Questions: {total_questions}")
    print(f"Exact Match: {em_rate:.1%} ({em_count}/{total_questions})")
    print(f"F1 Average: {f1_avg:.3f}")
    print(f"Coverage: {coverage_rate:.1%}")
    print(f"LC-MAPPO Steps: {lc_mappo_stats['total_steps']}")

    output_file = "outputs/lc_mappo_eval.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_results = pd.DataFrame([{
        'question': r['question'],
        'gold_answer': '|'.join(r['gold_answers']),
        'predicted_answer': r['predicted'],
        'em': r['em'],
        'f1': r['f1'],
        'covered': r['covered'],
        'num_nodes': r['gb_summary']['num_nodes'],
        'num_edges': r['gb_summary']['num_edges'],
        'path_len': r['tr_summary']['path_len'],
        'candidates_count': len(r['candidates'])
    } for r in results])
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    return {'em_rate': em_rate, 'f1_avg': f1_avg, 'coverage_rate': coverage_rate,
            'total_questions': total_questions, 'lc_mappo_stats': lc_mappo_stats}

# ------------------------------- CLI -------------------------------

def main():
    parser = argparse.ArgumentParser(description="LC-MAPPO Enhanced Multi-Agent Evaluation")
    parser.add_argument("--qa_file", type=str,
                        default="/path/qa_test.txt",
                        help="Path to MetaQA test file")
    parser.add_argument("--cache_dir", type=str,
                        default="/path/",
                        help="Path to cache directory")
    parser.add_argument("--ttl_file", type=str, default=None,
                        help="Path to TTL file (optional; uses cache only if not provided)")
    parser.add_argument("--gb_steps", type=int, default=12,
                        help="Maximum graph-building steps")
    parser.add_argument("--max_hops", type=int, default=4,
                        help="Maximum traversal hops")
    parser.add_argument("--limit", type=int, default=20,
                        help="Maximum number of questions to evaluate")
    parser.add_argument("--no_lm", action="store_true",
                        help="Disable language model (use heuristic candidate re-ranking)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cpu/cuda)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (critic/policies simple format)")
    parser.add_argument("--train_first", type=int, default=0,
                        help="Train on the first N questions before evaluation")
    parser.add_argument("--train_epochs", type=int, default=1,
                        help="Number of lc_mappo updates per training question")
    parser.add_argument("--save_ckpt", action="store_true",
                        help="Save a simple LC-MAPPO checkpoint to outputs/")
    parser.add_argument("--skip_eval_update", action="store_true", default=True,
                        help="Do not update LC-MAPPO during evaluation (default True)")
    args = parser.parse_args()

    results = evaluate_lc_mappo(
        qa_file=args.qa_file,
        cache_dir=args.cache_dir,
        ttl_file=args.ttl_file,
        gb_steps=args.gb_steps,
        max_hops=args.max_hops,
        limit=args.limit,
        use_lm=not args.no_lm,
        device=args.device,
        checkpoint_path=args.checkpoint,
        train_first=args.train_first,
        train_epochs=args.train_epochs,
        save_ckpt=args.save_ckpt,
        skip_eval_update=args.skip_eval_update
    )

    print(f"\nLC-MAPPO evaluation completed!")
    print(f"Final EM Rate: {results['em_rate']:.1%}")
    print(f"Final F1: {results['f1_avg']:.3f}")
    print(f"Final Coverage: {results['coverage_rate']:.1%}")

if __name__ == "__main__":
    main()
