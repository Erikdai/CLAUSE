# lc_mappo.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging
import re  # kept if other modules import and expect it around

# Feature specs for obs/action encodings
from .action_encoding import (
    PATH_LEN_NORM_MAX, CAND_NORM_MAX, DEC_COMMON_ID_THRESHOLD, DEC_SHORT_OUTPUT_LEN,
)

# Default fixed length for DLL/GNN auxiliary feature vectors
AUX_FEAT_DIM = 32

# Optional: expose RERANK/Generator for trainers that register agents here
try:
    from ..agents.dynamic_reranker import DynamicReranker  # noqa: F401
except Exception:
    DynamicReranker = None  # type: ignore
try:
    # decoder exports GeneratorScorer alias internally
    from ..agents.decoder import GeneratorScorer  # noqa: F401
except Exception:
    GeneratorScorer = None  # type: ignore



logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Centralized critic with four monotone heads
# ----------------------------------------------------------------------
class MultiHeadCentralCritic(nn.Module):
    def __init__(self, state_dim, obs_dim, act_dim, dll_dim, gnn_dim, hidden=256, n_agents: int = 3):
        super().__init__()
        in_dim = state_dim + obs_dim + act_dim + dll_dim + gnn_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        # four monotone heads: task, edge, latency, logic
        self.heads = nn.ModuleList([MonotoneMixer(hidden) for _ in range(4)])
        
        # Pre-initialize mixers so they're part of critic.parameters() for training
        ctx_dim = state_dim + dll_dim + gnn_dim
        self._mixers = nn.ModuleList([QMixer(n_agents, ctx_dim) for _ in range(4)])
        self._mix_n = n_agents
        self._mix_ctx = ctx_dim

    def forward(self, state, obs, acts, dll_feat, gnn_summary):
        """
        Forward pass that returns per-agent Q-values [B, n_agents, 4].
        """
        # Ensure batch dimension exists
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if dll_feat.dim() == 1:
            dll_feat = dll_feat.unsqueeze(0)
        if gnn_summary.dim() == 1:
            gnn_summary = gnn_summary.unsqueeze(0)
        
        # Handle obs and acts to ensure [B, n_agents, features] format
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if acts.dim() == 1:
            acts = acts.unsqueeze(0).unsqueeze(0)
        elif acts.dim() == 2:
            acts = acts.unsqueeze(1)
        
        B, n_agents = obs.shape[:2]
        assert acts.shape[0] == B and acts.shape[1] == n_agents, \
            f"acts shape {tuple(acts.shape)} must match obs batch/agent dims {(B, n_agents)}"

        # Ensure state/dll/gnn have batch dimension B
        if state.size(0) != B:
            state = state.expand(B, -1).contiguous()
        if dll_feat.size(0) != B:
            dll_feat = dll_feat.expand(B, -1).contiguous()
        if gnn_summary.size(0) != B:
            gnn_summary = gnn_summary.expand(B, -1).contiguous()
        
        # Per-agent processing
        agent_q_values = []
        for i in range(n_agents):
            obs_i = obs[:, i]   # [B, obs_dim]
            acts_i = acts[:, i] # [B, act_dim]
            x_i = torch.cat([state, obs_i, acts_i, dll_feat, gnn_summary], dim=-1)
            base_i = self.shared(x_i)
            q_i = torch.stack([h(base_i) for h in self.heads], dim=-1)  # [B, 4]
            agent_q_values.append(q_i)
        
        result = torch.stack(agent_q_values, dim=1)  # [B, n_agents, 4]
        assert result.shape == (state.size(0), len(agent_q_values), 4), \
            f"Expected shape (B, n_agents, 4), got {result.shape}"
        return result

    def mix_joint(self, per_agent_q: torch.Tensor, state, dll_feat, gnn_summary) -> torch.Tensor:
        """
        Monotonically mix per-agent per-head values into joint per-head Q using QMixer.
        Returns [B, 4] (task, edge, latency, logic)
        """
        # Ensure batch dims
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if dll_feat.dim() == 1:
            dll_feat = dll_feat.unsqueeze(0)
        if gnn_summary.dim() == 1:
            gnn_summary = gnn_summary.unsqueeze(0)

        B = per_agent_q.size(0)
        assert per_agent_q.size(1) == self._mix_n, \
            f"per_agent_q has {per_agent_q.size(1)} agents; expected {self._mix_n}"
        if state.size(0) != B:
            state = state.expand(B, -1).contiguous()
        if dll_feat.size(0) != B:
            dll_feat = dll_feat.expand(B, -1).contiguous()
        if gnn_summary.size(0) != B:
            gnn_summary = gnn_summary.expand(B, -1).contiguous()

        ctx = torch.cat([state, dll_feat, gnn_summary], dim=-1)  # [B, ctx_dim]

        q_joint = []
        for h in range(4):
            q_h_agents = per_agent_q[..., h]  # [B, n_agents]
            q_joint_h = self._mixers[h](q_h_agents, ctx)  # [B]
            q_joint.append(q_joint_h)
        result = torch.stack(q_joint, dim=-1)  # [B,4]
        assert result.shape == (B, 4), f"Expected shape (B, 4), got {result.shape}"
        return result
    
    def get_head_weight_norms(self):
        """Get weight norms for all monotone heads for monitoring."""
        return {
            'task': self.heads[0].get_weight_norm().item(),
            'edge': self.heads[1].get_weight_norm().item(),
            'latency': self.heads[2].get_weight_norm().item(),
            'logic': self.heads[3].get_weight_norm().item()
        }

class MonotoneMixer(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.final_layer = nn.Linear(hidden, 1, bias=False)
        with torch.no_grad():
            self.final_layer.weight.data = torch.nn.functional.softplus(
                torch.randn_like(self.final_layer.weight.data)
            )

    def forward(self, x):
        hidden_out = self.hidden(x)
        transformed_weights = torch.nn.functional.softplus(self.final_layer.weight)
        output = torch.nn.functional.linear(hidden_out, transformed_weights, bias=None)
        return output.squeeze(-1)
    
    def get_weight_norm(self):
        eff_w = torch.nn.functional.softplus(self.final_layer.weight.detach())
        return torch.norm(eff_w)

class QMixer(nn.Module):
    """QMixer-style hypernetwork for monotonic mixing."""
    def __init__(self, n_agents: int, ctx_dim: int, hidden: int = 64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.hyper_w = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_agents)
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, q_i: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if q_i.dim() != 2:
            raise ValueError(f"QMixer expects q_i [B, n_agents], got {tuple(q_i.shape)}")
        if q_i.size(1) != self.n_agents:
            raise ValueError(f"QMixer n_agents={self.n_agents}, got {q_i.size(1)}")
        w = F.softplus(self.hyper_w(ctx))
        b = self.hyper_b(ctx)
        return (q_i * w).sum(dim=1, keepdim=False) + b.squeeze(-1)

# ----------------------------------------------------------------------
# Counterfactual advantages (COMA-style) for each head and agent
# ----------------------------------------------------------------------
def counterfactual_advantages(critic, state, obs, dll, gnn, actions, policies, ctx: Optional[Dict] = None, agents: Optional[Dict] = None):
    """Compute COMA-style counterfactual advantages without tracking gradients."""
    device = actions.device
    with torch.no_grad():
        q_all = critic(state, obs, actions, dll, gnn).detach().to(device)  # [B, n_agents, 4]
        B = actions.size(0)

        def _tile_ctx_2d(x: torch.Tensor, B_local: int, K: int) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if x.size(0) == 1:
                x = x.expand(B_local, -1).contiguous()
            elif x.size(0) != B_local:
                raise ValueError(f"Context batch mismatch: expected {B_local}, got {x.size(0)}")
            return x.repeat(K, 1)

        adv = []
        for i, policy in enumerate(policies):
            if actions.dim() == 3 and actions.size(-1) == 3 and i in (0, 1):
                cand_idx = [0, 1, 2]  # small action space for GB/TR
                K = len(cand_idx)

                B_local = obs.size(0) if obs.dim() >= 2 else 1

                # Try policy-derived probabilities; fallback to uniform
                probs = None
                try:
                    obs_i_num = obs[:, i] if obs.dim() == 3 else (obs[i] if obs.dim() == 2 else obs)
                    logps_list = []
                    for idx in cand_idx:
                        lp = policy(obs_i_num).log_prob(torch.tensor(idx, device=device))
                        if isinstance(lp, torch.Tensor):
                            if lp.dim() == 0:
                                lp = lp.expand(B_local)
                            elif lp.dim() == 1 and lp.size(0) == B_local:
                                pass
                            else:
                                raise ValueError(f"log_prob shape {tuple(lp.shape)} incompatible with batch {B_local}")
                            logps_list.append(lp.detach())
                        else:
                            logps_list.append(torch.full((B_local,), float(lp), device=device))
                    probs = torch.softmax(torch.stack(logps_list, dim=1), dim=1)  # [B,K]
                except Exception:
                    probs = None

                if probs is None:
                    probs = torch.full((B, K), 1.0 / K, device=device, dtype=torch.float32)

                # Expected counterfactual value under π_i
                if i == 0:
                    acts_rep = actions.clone().to(device).repeat(K, 1, 1).reshape(K, B, actions.size(1), actions.size(2))
                    idx_tensor = torch.tensor(cand_idx, dtype=torch.float32, device=device).unsqueeze(1).expand(K, B)
                    is_delete = (idx_tensor == 1).float()
                    is_stop = (idx_tensor == 2).float()
                    acts_rep[:, :, i, 0] = idx_tensor
                    acts_rep[:, :, i, 1] = is_delete
                    acts_rep[:, :, i, 2] = is_stop
                    acts_rep = acts_rep.reshape(K * B, actions.size(1), actions.size(2))

                    state_rep = _tile_ctx_2d(state, B, K)
                    dll_rep   = _tile_ctx_2d(dll,   B, K)
                    gnn_rep   = _tile_ctx_2d(gnn,   B, K)
                    obs_rep   = obs.repeat(K, 1, 1)
                    q_cf = critic(state_rep, obs_rep, acts_rep, dll_rep, gnn_rep).detach().to(device)  # [K*B,n_agents,4]
                    q_cf_i = q_cf[:, i].reshape(K, B, 4)  # [K,B,4]
                    weights = probs.transpose(0, 1).unsqueeze(-1)  # [K,B,1]
                    exp_q_cf = (q_cf_i * weights).sum(dim=0)  # [B,4]
                else:
                    # TR: STOP vs CONT baseline (index-based weighting)
                    acts_rep = actions.clone().to(device).repeat(K, 1, 1).reshape(K, B, actions.size(1), actions.size(2))
                    idx_tensor = torch.tensor(cand_idx, dtype=torch.float32, device=device).unsqueeze(1).expand(K, B)
                    stop_flag = (idx_tensor == 0).float()
                    acts_rep[:, :, i, 0] = stop_flag
                    acts_rep = acts_rep.reshape(K * B, actions.size(1), actions.size(2))

                    state_rep = _tile_ctx_2d(state, B, K)
                    obs_rep   = obs.repeat(K, 1, 1)
                    dll_rep   = _tile_ctx_2d(dll,   B, K)
                    gnn_rep   = _tile_ctx_2d(gnn,   B, K)
                    q_cf = critic(state_rep, obs_rep, acts_rep, dll_rep, gnn_rep).detach().to(device)  # [K*B,n_agents,4]
                    q_cf_i = q_cf[:, i].reshape(K, B, 4)
                    weights = probs.transpose(0, 1).unsqueeze(-1)
                    exp_q_cf = (q_cf_i * weights).sum(dim=0)  # [B,4]

                q_all_i = q_all[:, i]
                adv.append(q_all_i - exp_q_cf)
            else:
                # Fallback: single baseline
                replaced = actions.clone().to(device)
                if actions.dim() == 3 and actions.size(-1) == 3:
                    B = actions.size(0)
                    baseline = torch.zeros((B, 3), dtype=actions.dtype, device=device)
                    if i == 0:
                        baseline[:, 0] = 2.0
                    elif i == 1:
                        baseline[:, 0] = 1.0
                    else:
                        baseline[:, 0] = 0.0
                        baseline[:, 1] = 1.0
                        baseline[:, 2] = 1.0
                    replaced[:, i, :] = baseline
                else:
                    if obs.dim() == 1:
                        obs_i = obs[i] if i < obs.size(0) else obs[0]
                    else:
                        obs_i = obs[:, i] if i < obs.size(1) else obs[:, 0]
                    dist = policy(obs_i)
                    baseline_act = dist.sample().detach().to(device)
                    if actions.dim() == 1:
                        replaced[i] = baseline_act if i < actions.size(0) else baseline_act
                    else:
                        replaced[:, i] = baseline_act if i < actions.size(1) else baseline_act

                replaced = replaced.detach()
                q_cf = critic(state, obs, replaced, dll, gnn).detach().to(device)
                q_all_i = q_all[:, i]
                q_cf_i = q_cf[:, i]
                adv.append(q_all_i - q_cf_i)

        cf_adv = torch.stack(adv, dim=1).detach().to(device)  # [B, n_agents, 4]
        assert cf_adv.shape == (actions.size(0), actions.size(1), 4), \
            f"Expected shape (B, n_agents, 4), got {cf_adv.shape}"
    return cf_adv

# ----------------------------------------------------------------------
# PPO update with Lagrangian-shaped advantages and adaptive clipping
# ----------------------------------------------------------------------
def actor_loss(logp_new, logp_old, lag_adv, lambdas, eps0, kappa, eps_min: float = 0.0):
    try:
        pressure = 0.0
        for v in lambdas.values():
            pressure += float(v.detach().item() if isinstance(v, torch.Tensor) else v)
    except Exception:
        pressure = sum(float(v.detach().item() if isinstance(v, torch.Tensor) else v) for v in lambdas.values())
    eps = float(eps0) / (1.0 + float(kappa) * float(pressure))
    if eps_min is not None:
        eps = max(float(eps_min), eps)
    ratio = torch.exp(torch.clamp(logp_new - logp_old, -20, 20))
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    return -torch.mean(torch.min(ratio * lag_adv, clipped * lag_adv))

# ----------------------------------------------------------------------
# Dual ascent on λs using rollout statistics
# ----------------------------------------------------------------------
def update_duals(lambdas, costs, budgets, step_size):
    """In-place dual ascent updates for λ to preserve Tensor object identity."""
    for k, cur in lambdas.items():
        cost_k = costs[k]
        bud_k = budgets[k]
        if isinstance(cur, torch.Tensor):
            with torch.no_grad():
                ss = cur.new_tensor(float(step_size) if not isinstance(step_size, torch.Tensor) else float(step_size.item()))
                ck = cost_k if isinstance(cost_k, torch.Tensor) else cur.new_tensor(float(cost_k))
                bk = bud_k if isinstance(bud_k, torch.Tensor) else cur.new_tensor(float(bud_k))
                updated = torch.clamp(cur + ss * (ck - bk), min=0.0)
                cur.copy_(updated)
        else:
            cs = float(cost_k.detach().item() if isinstance(cost_k, torch.Tensor) else cost_k)
            bs = float(bud_k.detach().item() if isinstance(bud_k, torch.Tensor) else bud_k)
            ss = float(step_size.detach().item() if isinstance(step_size, torch.Tensor) else step_size)
            lambdas[k] = max(0.0, float(cur) + ss * (cs - bs))

# ----------------------------------------------------------------------
# Monotonicity verification utilities
# ----------------------------------------------------------------------
def verify_input_grad_sign_debug(critic, test_inputs, epsilon=1e-6, max_total_elems: int = 200_000, max_batch: int = 64):
    """Debug utility: Check gradient signs of critic outputs w.r.t. raw inputs. Not a mixer proof."""
    try:
        total = 0
        for tup in test_inputs:
            for t in tup:
                if isinstance(t, torch.Tensor):
                    total += t.numel()
                    if t.dim() > 0 and t.size(0) > max_batch:
                        return {'skipped': True, 'reason': 'batch_too_large', 'batch': int(t.size(0)), 'max_batch': int(max_batch)}
        if total > max_total_elems:
            return {'skipped': True, 'reason': 'too_many_elements', 'total_elems': int(total), 'max_total_elems': int(max_total_elems)}
    except Exception:
        pass

    results: Dict[str, object] = {}

    for i, (state, obs, acts, dll, gnn) in enumerate(test_inputs):
        def _prep(t: torch.Tensor):
            if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
                return t.detach().requires_grad_(True)
            return t
        state = _prep(state); obs = _prep(obs); acts = _prep(acts); dll = _prep(dll); gnn = _prep(gnn)
        q_values = critic(state, obs, acts, dll, gnn)
        head_names = ['task', 'edge', 'latency', 'logic']
        for head_idx, head_name in enumerate(head_names):
            q_head = q_values[..., head_idx]
            grad_output = torch.ones_like(q_head)
            for tensor, name in zip([state, obs, acts, dll, gnn], ['state','obs','acts','dll','gnn']):
                if not isinstance(tensor, torch.Tensor) or not tensor.dtype.is_floating_point:
                    results[f'head_{head_name}_{name}_input_{i}'] = True
                    continue
                try:
                    grad_input = torch.autograd.grad(
                        outputs=q_head,
                        inputs=tensor,
                        grad_outputs=grad_output,
                        create_graph=False,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if grad_input is not None:
                        results[f'head_{head_name}_{name}_input_{i}'] = bool((grad_input >= -epsilon).all().item())
                    else:
                        results[f'head_{head_name}_{name}_input_{i}'] = True
                except Exception as e:
                    results[f'head_{head_name}_{name}_input_{i}'] = f'Error: {e}'
    return results

def verify_mixer_monotonicity(critic,
                              q_i_sample: torch.Tensor,
                              state: torch.Tensor,
                              dll: torch.Tensor,
                              gnn: torch.Tensor,
                              eps: float = 1e-6) -> Dict[str, bool]:
    """Verify ∂Q_joint^h/∂q_i^h ≥ 0 for the mixer (true monotonicity check)."""
    try:
        q_i = q_i_sample.detach().clone().requires_grad_(True)
        q_joint = critic.mix_joint(q_i, state, dll, gnn)  # [B, 4]
        head_names = ['task', 'edge', 'latency', 'logic']
        out: Dict[str, bool] = {}
        for h, name in enumerate(head_names):
            ones = torch.ones_like(q_joint[..., h])  # [B]
            grads = torch.autograd.grad(q_joint[..., h], q_i, grad_outputs=ones,
                                        retain_graph=True, create_graph=False)[0]  # [B,n_agents,4]
            cond = (grads[..., h] >= -float(eps)).all().item()
            out[name] = bool(cond)
        return out
    except Exception:
        return {'error': True}

# ----------------------------------------------------------------------
# GAE utilities (per-head, batched)
# ----------------------------------------------------------------------
def compute_gae_batch(rewards: torch.Tensor,
                      values: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float = 0.99,
                      lam: float = 0.95) -> torch.Tensor:
    """
    Batched GAE for per-head returns.

    Args:
        rewards: [T, B]
        values:  [T+1, B, n_agents]
        dones:   [T, B] (1.0 if terminal at t else 0.0)
    """
    T, B = rewards.shape
    n_agents = values.size(-1)
    adv = torch.zeros((T, B, n_agents), dtype=values.dtype, device=values.device)
    lastgaelam = torch.zeros((B, n_agents), dtype=values.dtype, device=values.device)

    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t].to(values.dtype)
        nonterm = nonterm.unsqueeze(-1)
        delta = rewards[t].unsqueeze(-1) + gamma * nonterm * values[t + 1] - values[t]
        lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        adv[t] = lastgaelam

    returns = adv + values[:-1]
    return returns

# ----------------------------------------------------------------------
# Policy Interface for Real Agents
# ----------------------------------------------------------------------
class AgentPolicy:
    """Wrapper that exposes a minimal action head for sampling/log_prob."""
    def __init__(self, agent, agent_type, device):
        self.agent = agent
        self.agent_type = agent_type
        self.device = device

    def __call__(self, obs):
        agent = self.agent
        device = self.device
        if not hasattr(self, 'action_head'):
            obs_dim = obs.shape[-1] if isinstance(obs, torch.Tensor) else 6
            self.action_head = nn.Linear(obs_dim, 2).to(device)  # 2 actions
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        logits = self.action_head(obs)

        class SimpleDistribution:
            def __init__(self, logits):
                self.logits = logits
            def sample(self):
                return torch.multinomial(torch.softmax(self.logits, dim=-1), 1)
            def log_prob(self, action):
                return F.log_softmax(self.logits, dim=-1)[:, action]
        return SimpleDistribution(logits)

def create_enhanced_lc_mappo_policies(agents, device, success_history=0.8):
    """Create LC-MAPPO policies."""
    policies = []
    for i, (agent, agent_type) in enumerate(agents):
        policy = AgentPolicy(agent, agent_type, device)
        policy.success_history = success_history
        policies.append(policy)
    return policies

def update_success_history(policies, em_score):
    """EMA for success history (if you log it elsewhere)."""
    for policy in policies:
        if hasattr(policy, 'success_history'):
            policy.success_history = 0.9 * policy.success_history + 0.1 * em_score

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _broadcast_like(target, reference):
    """Broadcast target tensor to match reference tensor shape."""
    if target.dim() == 1:
        return target.unsqueeze(1).expand_as(reference)
    elif target.dim() == 2 and target.size(1) == 1:
        return target.expand(-1, reference.size(1))
    else:
        return target

def _normalize_feature(feat: Optional[torch.Tensor], target_dim: int = AUX_FEAT_DIM,
                       device: Optional[torch.device] = None,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Clamp or pad a 1D/2D feature tensor to [1, target_dim]."""
    if feat is None:
        return torch.zeros(1, target_dim, device=device, dtype=dtype)
    t = feat.to(device=device, dtype=dtype).reshape(1, -1)
    if t.size(1) >= target_dim:
        return t[:, :target_dim].contiguous()
    pad_len = target_dim - t.size(1)
    return F.pad(t, (0, pad_len), value=0.0)

def _compute_actor_loss_for_agent(i, policy, actions, new_logp, logp_old, lag_adv_detached, 
                                 obs, lambdas, eps0, kappa, eps_min, kl_coef, ent_coef):
    """Compute actor loss for a single agent (PPO clip + optional KL/entropy)."""
    act_i_vec = actions[:, i] if actions.dim() == 3 else actions[i]
    act_idx = act_i_vec[..., 0].long()  # [B]

    if new_logp is not None:
        if new_logp.dim() == 2:
            logp = new_logp[:, i]
        elif new_logp.dim() == 1:
            logp = new_logp[i].unsqueeze(0).expand_as(act_idx)
        else:
            logp = new_logp[:, i] if new_logp.size(1) > i else new_logp[:, 0]
    else:
        if i == 2 and actions.dim() == 3 and actions.size(-1) == 3:
            raise ValueError("DEC requires new_logp (token_norm is not a discrete token id).")
        obs_i = obs[:, i] if obs.dim() == 3 else obs[i]
        dist = policy(obs_i)
        logp = dist.log_prob(act_idx)

    if logp_old.dim() == 2:
        logp_old_i = logp_old[:, i]
    elif logp_old.dim() == 1:
        logp_old_i = logp_old[i].unsqueeze(0).expand_as(act_idx)
    else:
        logp_old_i = logp_old[:, i] if logp_old.size(1) > i else logp_old[:, 0]

    if lag_adv_detached.dim() == 2:
        lag_adv_i = lag_adv_detached[:, i]
    elif lag_adv_detached.dim() == 1:
        lag_adv_i = lag_adv_detached[i].unsqueeze(0).expand_as(act_idx)
    else:
        lag_adv_i = lag_adv_detached[:, i] if lag_adv_detached.size(1) > i else lag_adv_detached[:, 0]

    loss_clip = actor_loss(logp, logp_old_i, lag_adv_i, lambdas, eps0, kappa, eps_min=eps_min)
    approx_kl_i = torch.mean(logp_old_i - logp)
    loss = loss_clip + float(kl_coef) * approx_kl_i

    if ent_coef > 0 and hasattr(policy, 'agent') and hasattr(policy.agent, 'entropy'):
        obs_i = obs[:, i] if obs.dim() == 3 else obs[i]
        ent_i = policy.agent.entropy(obs_i)
        loss = loss - ent_coef * ent_i.mean()
    return loss, approx_kl_i

# ----------------------------------------------------------------------
# Training loop with proper critic loss and optimization
# ----------------------------------------------------------------------
def training_step(batch, critic, policies, lambdas, budgets, eps0, kappa, dual_lr,
                  new_logp=None, eps_min: float = 0.0, kl_coef: float = 0.0, kl_target: Optional[float] = None,
                  ent_coef: float = 0.0, do_dual_update: bool = True, agents: Optional[Dict] = None):
    # batch may include contexts at the end
    if len(batch) == 9:
        state, obs, dll, gnn, actions, logp_old, rewards, costs, _ctx = batch
    else:
        state, obs, dll, gnn, actions, logp_old, rewards, costs = batch

    if actions.dim() == 3 and actions.size(-1) == 3 and new_logp is None:
        raise ValueError(
            "training_step requires new_logp when using feature-vector actions; "
            "ensure lc_mappo_update computes current-policy log-probs via agents."
        )

    # Centralized critic and COMA advantages
    cf_adv = counterfactual_advantages(
        critic, state, obs, dll, gnn, actions, policies,
        _ctx if len(batch) == 9 else None,
        agents=agents,
    )

    # Mix heads with λs to get Lagrangian advantages
    lag_adv = (cf_adv[..., 0]                       # task
               - lambdas['edge']  * cf_adv[..., 1]
               - lambdas['lat']   * cf_adv[..., 2]
               - lambdas['logic'] * cf_adv[..., 3])
    lag_adv_detached = lag_adv.detach()

    # Lagrangian-shaped reward for value targets
    shaped_r = (rewards - lambdas['edge']  * costs['edge']
                          - lambdas['lat']   * costs['lat']
                          - lambdas['logic'] * costs['logic'])

    # PPO actor updates
    actor_losses = []
    kl_terms = []
    for i, policy in enumerate(policies):
        loss, approx_kl_i = _compute_actor_loss_for_agent(
            i, policy, actions, new_logp, logp_old, lag_adv_detached,
            obs, lambdas, eps0, kappa, eps_min, kl_coef, ent_coef
        )
        actor_losses.append(loss)
        kl_terms.append(approx_kl_i)
    total_actor_loss = torch.stack(actor_losses).sum()
    avg_kl = float(torch.stack(kl_terms).mean().item()) if kl_terms else 0.0
    
    # Adaptive KL target control (TRPO-style)
    kl_coef_updated = kl_coef
    if kl_target is not None and kl_target > 0:
        if avg_kl > kl_target * 1.5:
            kl_coef_updated = kl_coef * 1.5
        elif avg_kl < kl_target * 0.5:
            kl_coef_updated = kl_coef * 0.8
        if kl_coef_updated != kl_coef:
            actor_losses = []
            for i, policy in enumerate(policies):
                loss, _ = _compute_actor_loss_for_agent(
                    i, policy, actions, new_logp, logp_old, lag_adv_detached,
                    obs, lambdas, eps0, kappa, eps_min, kl_coef_updated, ent_coef
                )
                actor_losses.append(loss)
            total_actor_loss = torch.stack(actor_losses).sum()

    # Critic loss computation
    critic_losses = []
    traj = _ctx.get('traj', None) if len(batch) == 9 else None

    if traj is not None:
        try:
            task_r = traj['task_r']    # [T,B]
            edge_c = traj['edge_c']    # [T,B]
            lat_c  = traj['lat_c']     # [T,B]
            logic_c= traj['logic_c']   # [T,B]
            dones  = traj.get('dones', torch.zeros_like(task_r))

            states_seq = traj['states']    # [T+1,B,state_dim]
            obs_seq    = traj['obs']       # [T+1,B,n_agents,obs_dim]
            acts_seq   = traj['acts']      # [T+1,B,n_agents,act_dim]
            dll_seq    = traj['dll']       # [T+1,B,dll_dim]
            gnn_seq    = traj['gnn']       # [T+1,B,gnn_dim]

            T = task_r.size(0)
            q_seq = []
            q_joint_seq = []
            for t in range(T + 1):
                q_t = critic(states_seq[t], obs_seq[t], acts_seq[t], dll_seq[t], gnn_seq[t])  # [B,n_agents,4]
                q_seq.append(q_t)
                q_joint_seq.append(critic.mix_joint(q_t, states_seq[t], dll_seq[t], gnn_seq[t]))  # [B,4]
            q_seq = torch.stack(q_seq, dim=0)            # [T+1,B,n_agents,4]
            q_joint_seq = torch.stack(q_joint_seq, dim=0) # [T+1,B,4]

            returns_heads = {}
            returns_heads['task'] = compute_gae_batch(task_r,  q_joint_seq[..., 0].unsqueeze(-1), dones).squeeze(-1)
            returns_heads['edge'] = compute_gae_batch(edge_c,  q_joint_seq[..., 1].unsqueeze(-1), dones).squeeze(-1)
            returns_heads['latency'] = compute_gae_batch(lat_c,   q_joint_seq[..., 2].unsqueeze(-1), dones).squeeze(-1)
            returns_heads['logic'] = compute_gae_batch(logic_c, q_joint_seq[..., 3].unsqueeze(-1), dones).squeeze(-1)

            loss_sum = 0.0
            for head_idx, head_name in enumerate(['task', 'edge', 'latency', 'logic']):
                q_head_seq = q_seq[:-1, ..., head_idx]  # [T,B,n_agents]
                targets = returns_heads[head_name].unsqueeze(-1).expand_as(q_head_seq)
                loss_sum = loss_sum + F.smooth_l1_loss(q_head_seq, targets)
            total_critic_loss = loss_sum
        except Exception:
            current_q_values = critic(state, obs, actions, dll, gnn)
            for head_idx, head_name in enumerate(['task', 'edge', 'latency', 'logic']):
                q_head = current_q_values[..., head_idx]
                if head_name == 'task':
                    targets = rewards
                else:
                    cost_key = head_name if head_name != 'latency' else 'lat'
                    targets = costs[cost_key]
                targets = _broadcast_like(targets, q_head)
                critic_losses.append(F.smooth_l1_loss(q_head, targets))
            total_critic_loss = torch.stack(critic_losses).sum()
    else:
        q_t = critic(state, obs, actions, dll, gnn)           # [B,n_agents,4]
        q_joint = critic.mix_joint(q_t, state, dll, gnn)      # [B,4]
        B = q_t.size(0)
        total_critic_loss = 0.0
        task_r_step = rewards.mean(dim=1) if rewards.dim() == 2 else rewards.reshape(B)
        edge_c_step = costs['edge'].sum(dim=1) if costs['edge'].dim() == 2 else costs['edge'].reshape(B)
        lat_c_step  = costs['lat'].sum(dim=1)  if costs['lat'].dim()  == 2 else costs['lat'].reshape(B)
        logic_c_step= costs['logic'].sum(dim=1)if costs['logic'].dim()== 2 else costs['logic'].reshape(B)
        task_r_1 = task_r_step.unsqueeze(0)
        edge_c_1 = edge_c_step.unsqueeze(0)
        lat_c_1  = lat_c_step.unsqueeze(0)
        logic_c_1= logic_c_step.unsqueeze(0)
        dones = torch.ones((1, B), dtype=torch.float32, device=task_r_1.device)

        for h, (name, r_t) in enumerate([
            ('task', task_r_1), ('edge',  edge_c_1), ('latency',  lat_c_1), ('logic',  logic_c_1)
        ]):
            v_joint_t = q_joint[..., h]
            values_seq = torch.stack([v_joint_t, torch.zeros_like(v_joint_t)], dim=0)  # [2,B]
            returns = compute_gae_batch(r_t, values_seq.unsqueeze(-1), dones).squeeze(-1)  # [1,B]
            target_t = _broadcast_like(returns[0], q_t[..., h])
            total_critic_loss = total_critic_loss + F.smooth_l1_loss(q_t[..., h], target_t)
    
    total_loss = total_actor_loss + 0.5 * total_critic_loss  # 0.5 is critic coefficient
    
    avg_costs = {k: v.mean().detach() for k, v in costs.items()}
    if do_dual_update:
        update_duals(lambdas, avg_costs, budgets, dual_lr)
    
    violation_rates = {}
    for k in ['edge', 'lat', 'logic']:
        violations = (costs[k] > budgets[k]).float()
        violation_rates[f'violation_rate_{k}'] = violations.mean().item()
    
    all_feasible = torch.ones_like(costs['edge'], dtype=torch.bool)
    for k in ['edge', 'lat', 'logic']:
        all_feasible = all_feasible & (costs[k] <= budgets[k])
    feasible_fraction = all_feasible.float().mean().item()
    
    return {
        'total_loss': total_loss,
        'actor_loss': total_actor_loss,
        'critic_loss': total_critic_loss,
        'lag_adv_mean': lag_adv_detached.mean().item(),
        'shaped_reward_mean': shaped_r.mean().item(),
        'approx_kl': avg_kl,
        'kl_coef_updated': kl_coef_updated,
        'lambda_edge': lambdas['edge'].item(),
        'lambda_lat': lambdas['lat'].item(),
        'lambda_logic': lambdas['logic'].item(),
        'avg_cost_edge': avg_costs['edge'].item(),
        'avg_cost_lat': avg_costs['lat'].item(),
        'avg_cost_logic': avg_costs['logic'].item(),
        'violation_rate_edge': violation_rates['violation_rate_edge'],
        'violation_rate_lat': violation_rates['violation_rate_lat'],
        'violation_rate_logic': violation_rates['violation_rate_logic'],
        'feasible_fraction': feasible_fraction,
    }

# ----------------------------------------------------------------------
# Complete LC-MAPPO Training with Optimization
# ----------------------------------------------------------------------
def lc_mappo_update(batch, critic, policies, lambdas, budgets, eps0, kappa, dual_lr, 
                    critic_optimizer, policy_optimizers, max_grad_norm=0.5, agents=None,
                    eps_min: float = 0.0, kl_coef: float = 0.0, kl_target: Optional[float] = None,
                    ent_coef: float = 0.0, actor_update_interval: int = 1, critic_update_interval: int = 1,
                    dual_update_interval: int = 10, global_step: int = 0):
    """
    Complete LC-MAPPO update with both actor and critic optimization.
    """
    critic_optimizer.zero_grad()
    for opt in (policy_optimizers or []):
        opt.zero_grad(set_to_none=True)
    
    # Optionally compute real current-policy log-probs using agents and contexts
    new_logp = None
    if agents is not None and len(batch) == 9:
        state, obs, dll, gnn, actions, logp_old, rewards, costs, ctx = batch
        device = rewards.device
        assert actions.size(0) == 1, "new_logp path currently supports batch size 1"
        gb_action = ctx.get('gb_action') or {"op": "stop"}
        tr_action = ctx.get('tr_action') or {"stop": True}
        dec_action = ctx.get('dec_action') or {"token_id": 0}
        if gb_action is None or 'op' not in gb_action:
            act_idx = actions[0, 0, 0].long()
            gb_action = {"op": "add_edge"} if int(act_idx.item()) == 0 else ({"op": "delete_edge"} if int(act_idx.item()) == 1 else {"op": "stop"})
        if tr_action is None or 'stop' not in tr_action:
            act_idx = actions[0, 1, 0].long()
            tr_action = {"stop": True} if int(act_idx.item()) == 0 else {"stop": False, "candidate_idx": int(act_idx.item())}
        if dec_action is None or 'token_id' not in dec_action:
            token_norm = float(actions[0, 2, 0].item())
            approx_vocab = int(ctx.get('dec_vocab_size', 2048))
            dec_action = {"token_id": int(round(token_norm * (approx_vocab - 1)))}
        lp_gb = agents['gb'].log_prob_action(ctx.get('gb_obs', {}), gb_action)
        lp_tr = agents['tr'].log_prob_action(ctx.get('tr_obs', {}), tr_action)
        dec_key = 'dec' if ('dec' in agents) else ('rerank' if 'rerank' in agents else ('RERANK' if 'RERANK' in agents else None))
        if dec_key is not None:
            lp_dec = agents[dec_key].log_prob_action(ctx.get('dec_obs', {}), dec_action, ctx.get('dec_rule_masks', []), ctx.get('dec_prev_tokens', []))
        else:
            lp_dec = torch.tensor(0.0, device=device)
        new_logp = torch.stack([lp_gb.reshape(()), lp_tr.reshape(()), lp_dec.reshape(())], dim=0).unsqueeze(0).to(device)

    do_actor_step = (actor_update_interval is None) or (int(actor_update_interval) <= 1) or (int(global_step) % int(actor_update_interval) == 0)
    do_critic_step = (critic_update_interval is None) or (int(critic_update_interval) <= 1) or (int(global_step) % int(critic_update_interval) == 0)
    do_dual = do_actor_step and ((dual_update_interval is None) or (int(dual_update_interval) <= 1) or (int(global_step) % int(dual_update_interval) == 0))

    training_stats = training_step(batch, critic, policies, lambdas, budgets, eps0, kappa, dual_lr,
                                   new_logp=new_logp, eps_min=eps_min, kl_coef=kl_coef, kl_target=kl_target,
                                   ent_coef=ent_coef, do_dual_update=do_dual, agents=agents)

    # Backward pass
    training_stats['total_loss'].backward()
    
    # Gradient clipping
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    training_stats['critic_grad_norm'] = critic_grad_norm.item()
    
    # Update critic (fast timescale)
    if do_critic_step:
        critic_optimizer.step()
    
    # Clip and step policy (actor) optimizers
    policy_grad_norms = []
    if do_actor_step:
        for opt in (policy_optimizers or []):
            params = []
            for g in opt.param_groups:
                params.extend(g.get('params', []))
            if params:
                gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                policy_grad_norms.append(float(gn.item() if hasattr(gn, 'item') else gn))
            opt.step()

    training_stats['policy_grad_norms'] = policy_grad_norms
    training_stats['avg_policy_grad_norm'] = (sum(policy_grad_norms) / len(policy_grad_norms)) if policy_grad_norms else 0.0
    training_stats['did_actor_step'] = bool(do_actor_step)
    training_stats['did_critic_step'] = bool(do_critic_step)
    training_stats['did_dual_update'] = bool(do_dual)

    # Keep these keys for logging compatibility (no DLL coverage in cleaned version)
    training_stats['coverage_loss'] = training_stats.get('coverage_loss', 0.0)
    training_stats['coverage_coef'] = 0.0
    return training_stats

# ----------------------------------------------------------------------
# Utility Functions for Training Setup
# ----------------------------------------------------------------------
def create_lc_mappo_policies(agents, device):
    """
    Create AgentPolicy objects for the given agents.
    """
    policies = []
    # Always include GB and TR
    for agent_type, key in [('gb', 'gb'), ('tr', 'tr')]:
        if key in agents:
            policies.append(AgentPolicy(agents[key], agent_type, device))
    # Third slot: DEC or RERANK
    dec_key = 'dec'
    if 'dec' not in agents:
        if 'rerank' in agents:
            dec_key = 'rerank'
        elif 'RERANK' in agents:
            dec_key = 'RERANK'
        else:
            dec_key = None
    if dec_key is not None:
        policies.append(AgentPolicy(agents[dec_key], 'dec', device))
    return policies

def create_lc_mappo_optimizers(critic, policies, lr=3e-4):
    """
    Create optimizers for LC-MAPPO training.
    """
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    seen = set()
    policy_optimizers = []
    for p in policies:
        agent = getattr(p, 'agent', None)
        if agent is None:
            continue
        if id(agent) in seen:
            continue
        seen.add(id(agent))
        params = [t for t in agent.parameters() if t.requires_grad]
        if params:
            policy_optimizers.append(torch.optim.Adam(params, lr=lr))
    return critic_optimizer, policy_optimizers

def prepare_lc_mappo_batch(
    state_feats,
    step_logs,
    recorded_actions,
    gb_summary=None,
    tr_summary=None,
    dec_summary=None,
    tau: float = 1.0,
    device: str = "cpu",
    dec_vocab_size: int = 2048,
    latency_ms: float = None,
    latency_budget_ms: float = 200.0,
    dll_feat: torch.Tensor = None,
    gnn_feat: torch.Tensor = None,
    precomputed_costs: Optional[Dict[str, torch.Tensor]] = None,
):
    """Build a batch for LC-MAPPO from last-step rollout data.

    Produces per-agent observations (6 features each) and action encodings (3 dims each):
      GB obs:  [num_nodes, num_edges, est_tokens, edge_density, tokens_norm, gb_steps]
      TR obs:  [path_len, candidate_count, stop_flag, path_norm, cand_norm, tr_steps]
      DEC obs: [out_len, rule_count, tau, out_len_norm, rule_norm, dec_steps]

      Actions vector encodes the chosen action index in first component.
    """
    # State: average of episode features → [1, D]
    state = torch.stack(state_feats).to(device)
    state = state.mean(dim=0, keepdim=True)

    # Extract last summaries safely
    num_nodes = float((gb_summary or {}).get("num_nodes", 0))
    num_edges = float((gb_summary or {}).get("num_edges", 0))
    est_tokens = float((gb_summary or {}).get("estimated_tokens", 0))
    path_len = float((tr_summary or {}).get("path_len", 0))
    candidate_count = float((tr_summary or {}).get("candidate_count", 0))
    stop_flag = float((tr_summary or {}).get("stop", 0))
    out_len = float((dec_summary or {}).get("out_len", 0))
    rule_count = float((dec_summary or {}).get("rule_count", 0))

    def create_agent_observations(gb_summary, tr_summary, dec_summary, tau, step_logs):
        gb_obs = [
            num_nodes,
            num_edges, 
            est_tokens,
            min(1.0, num_edges / max(1, num_nodes)) if num_nodes > 0 else 0.0,
            min(1.0, est_tokens / 100.0),
            float(len([s for s in step_logs if s.agent == 'gb']))
        ]
        tr_obs = [
            path_len,
            candidate_count,
            stop_flag,
            min(1.0, path_len / 10.0),
            min(1.0, candidate_count / 20.0),
            float(len([s for s in step_logs if s.agent == 'tr']))
        ]
        dec_obs = [
            out_len,
            rule_count,
            float(tau),
            min(1.0, out_len / 50.0),
            min(1.0, rule_count / 10.0),
            float(len([s for s in step_logs if s.agent == 'dec']))
        ]
        return gb_obs, tr_obs, dec_obs
    
    gb_obs, tr_obs, dec_obs = create_agent_observations(gb_summary, tr_summary, dec_summary, tau, step_logs)
    
    def pad_observation(obs, target_length=6):
        while len(obs) < target_length:
            obs.append(0.0)
        return obs[:target_length]
    
    gb_obs = pad_observation(gb_obs)
    tr_obs = pad_observation(tr_obs)
    dec_obs = pad_observation(dec_obs)
    
    obs = torch.tensor([gb_obs, tr_obs, dec_obs], dtype=torch.float32, device=device).unsqueeze(0)

    # Encode actions for each agent
    def encode_gb_action(ga):
        if not isinstance(ga, dict):
            return 0, 0.0, 0.0
        op = ga.get("op") or ga.get("operation") or ga.get("operation_type", "add_edge")
        op_map = {"add_edge": 0, "delete_edge": 1, "stop": 2}
        gb_idx = op_map.get(op, 0)
        is_delete = float(gb_idx == 1)
        is_stop = float(gb_idx == 2)
        return gb_idx, is_delete, is_stop

    def encode_tr_action(ta):
        if not isinstance(ta, dict):
            return 0, 0.0, 0.0
        if ta.get("stop", False):
            tr_idx = 0
        else:
            candidate_idx = ta.get("candidate_idx", 0)
            tr_idx = min(max(1, candidate_idx + 1), 10)
        stop_f = float(ta.get("stop", False))
        path_len_norm = min(1.0, path_len / PATH_LEN_NORM_MAX)
        return tr_idx, stop_f, path_len_norm

    def encode_dec_action(da):
        if not isinstance(da, dict):
            return 0, 0.0, 0.0
        token_id = da.get("token_id", 0)
        token_id = max(0, min(token_id, dec_vocab_size - 1))
        dec_token_norm = float(token_id) / float(max(1, dec_vocab_size - 1))
        is_common = float(token_id < DEC_COMMON_ID_THRESHOLD)
        is_short = float(out_len < DEC_SHORT_OUTPUT_LEN)
        return token_id, dec_token_norm, is_common

    gb_actions = recorded_actions.get("gb_actions", [])
    tr_actions = recorded_actions.get("tr_actions", [])
    dec_actions = recorded_actions.get("dec_actions", [])
    gb_action = gb_actions[-1] if gb_actions else {}
    tr_action = tr_actions[-1] if tr_actions else {}
    dec_action = dec_actions[-1] if dec_actions else {}
    gb_idx, gb_is_delete, gb_is_stop = encode_gb_action(gb_action)
    tr_idx, tr_stop_flag, tr_path_norm = encode_tr_action(tr_action)
    dec_idx, dec_token_norm, dec_is_common = encode_dec_action(dec_action)

    gb_vec = [float(gb_idx), gb_is_delete, gb_is_stop]
    cand_norm = min(1.0, candidate_count / CAND_NORM_MAX)
    tr_vec = [tr_stop_flag, tr_path_norm, cand_norm]
    is_short = float(out_len < DEC_SHORT_OUTPUT_LEN)
    dec_vec = [dec_token_norm, dec_is_common, is_short]
    
    actions = torch.tensor([gb_vec, tr_vec, dec_vec], dtype=torch.float32, device=device).unsqueeze(0)  # [1,3,3]

    # DLL and GNN features (normalized to fixed length)
    dll = _normalize_feature(dll_feat, target_dim=AUX_FEAT_DIM, device=device, dtype=torch.float32)
    gnn = _normalize_feature(gnn_feat, target_dim=AUX_FEAT_DIM, device=device, dtype=torch.float32)

    # Old log-probs per agent from last occurrences
    last = {"gb": None, "tr": None, "dec": None}
    for s in reversed(step_logs):
        if s.agent in last and last[s.agent] is None:
            last[s.agent] = s.get_old_lp().to(device)
        if all(v is not None for v in last.values()):
            break
    logp_old = torch.stack([
        last["gb"] if last["gb"] is not None else torch.tensor(0.0, device=device),
        last["tr"] if last["tr"] is not None else torch.tensor(0.0, device=device),
        last["dec"] if last["dec"] is not None else torch.tensor(0.0, device=device),
    ], dim=0).unsqueeze(0)  # [1, 3]

    # Base rewards: pass-through from step logs (no additional shaping)
    def compute_agent_rewards(step_logs, gb_summary, tr_summary, dec_summary):
        rewards = {"gb": 0.0, "tr": 0.0, "dec": 0.0}
        for s in reversed(step_logs):
            if s.agent in rewards and rewards[s.agent] == 0.0:
                rewards[s.agent] = float(s.reward)
        return rewards
    
    agent_rewards = compute_agent_rewards(step_logs, gb_summary, tr_summary, dec_summary)
    rewards = torch.tensor([agent_rewards["gb"], agent_rewards["tr"], agent_rewards["dec"]], device=device).unsqueeze(0)  # [1, 3]

    # Costs
    violation_stats = {
        'violation_occurred': False,
        'violation_rate': 0.0,
        'total_rules': 0,
        'sampled_token_id': -1
    }
    
    if precomputed_costs is not None:
        edge_cost = precomputed_costs.get('edge', torch.tensor(0.0, device=device))
        lat_cost = precomputed_costs.get('latency', torch.tensor(0.0, device=device))
        logic_cost = precomputed_costs.get('logic', torch.tensor(0.0, device=device))
        if not isinstance(edge_cost, torch.Tensor): edge_cost = torch.tensor(edge_cost, device=device)
        if not isinstance(lat_cost, torch.Tensor):  lat_cost  = torch.tensor(lat_cost, device=device)
        if not isinstance(logic_cost, torch.Tensor):logic_cost= torch.tensor(logic_cost, device=device)
        if edge_cost.dim() == 0:   edge_cost = edge_cost.unsqueeze(0)
        if lat_cost.dim() == 0:    lat_cost = lat_cost.unsqueeze(0)
        if logic_cost.dim() == 0:  logic_cost = logic_cost.unsqueeze(0)
    else:
        # Fallback heuristics for costs (kept to remain functional when no measurements provided)
        edge_cost = min(1.0, num_edges / 50.0) if num_edges > 0 else 0.0
        if latency_ms is not None:
            budget = max(1e-6, float(latency_budget_ms))
            lat_cost = min(1.0, float(latency_ms) / budget)
        else:
            lat_cost = min(1.0, (est_tokens + path_len + out_len) / 64.0)
        # Logic cost from explicit rule violation check (not shaping)
        logic_cost = 0.0
        violation_occurred = False
        violation_rate = 0.0

        latest_dec_action = None
        latest_rule_masks = []
        for s in reversed(step_logs):
            if s.agent == 'dec':
                if latest_dec_action is None:
                    latest_dec_action = s.action if hasattr(s, 'action') else None
                if latest_rule_masks == []:
                    latest_rule_masks = getattr(s, 'rule_masks', []) if hasattr(s, 'rule_masks') else []
                if latest_dec_action is not None and latest_rule_masks:
                    break
        if latest_dec_action is None and recorded_actions.get('dec_actions'):
            latest_dec_action = recorded_actions['dec_actions'][-1]
        if latest_rule_masks == [] and recorded_actions.get('dec_rule_masks'):
            latest_rule_masks = recorded_actions['dec_rule_masks'][-1]
        if latest_dec_action and latest_rule_masks:
            if isinstance(latest_dec_action, dict):
                sampled_token_id = latest_dec_action.get('token_id', -1)
            else:
                sampled_token_id = latest_dec_action
            total_violations = 0
            total_rules = len(latest_rule_masks)
            for rule_mask in latest_rule_masks:
                if isinstance(rule_mask, list) and sampled_token_id in rule_mask:
                    violation_occurred = True
                    total_violations += 1
            violation_rate = (total_violations / total_rules) if total_rules > 0 else 0.0
        logic_cost = 1.0 if violation_occurred else 0.0
        violation_stats = {
            'violation_occurred': violation_occurred,
            'violation_rate': violation_rate,
            'logic_cost_binary': float(logic_cost),
            'total_rules': len(latest_rule_masks) if latest_rule_masks else 0,
            'sampled_token_id': sampled_token_id if 'sampled_token_id' in locals() else -1
        }

    n_agents = int(actions.size(1)) if isinstance(actions, torch.Tensor) and actions.dim() >= 2 else 3
    def compute_agent_specific_costs(edge_cost, lat_cost, logic_cost, gb_summary, tr_summary, dec_summary):
        costs = {
            'edge': torch.zeros(1, n_agents, device=device, dtype=torch.float32),
            'lat': torch.zeros(1, n_agents, device=device, dtype=torch.float32),
            'logic': torch.zeros(1, n_agents, device=device, dtype=torch.float32),
        }
        if gb_summary:
            num_edges = gb_summary.get("num_edges", 0)
            gb_edge_cost = (edge_cost if isinstance(edge_cost, float) else float(edge_cost.item())) * min(1.0, num_edges / 20.0)
            costs['edge'][0, 0] = gb_edge_cost
            costs['lat'][0, 0] = (lat_cost if isinstance(lat_cost, float) else float(lat_cost.item())) * 0.5
            costs['logic'][0, 0] = (logic_cost if isinstance(logic_cost, float) else float(logic_cost.item())) * 0.3
        if tr_summary:
            path_len = tr_summary.get("path_len", 0)
            tr_lat_cost = (lat_cost if isinstance(lat_cost, float) else float(lat_cost.item())) * min(1.0, path_len / 5.0)
            costs['edge'][0, 1] = (edge_cost if isinstance(edge_cost, float) else float(edge_cost.item())) * 0.3
            costs['lat'][0, 1] = tr_lat_cost
            costs['logic'][0, 1] = (logic_cost if isinstance(logic_cost, float) else float(logic_cost.item())) * 0.2
        if dec_summary:
            rule_count = dec_summary.get("rule_count", 0)
            dec_logic_cost = (logic_cost if isinstance(logic_cost, float) else float(logic_cost.item())) * min(1.0, rule_count / 3.0)
            costs['edge'][0, 2] = (edge_cost if isinstance(edge_cost, float) else float(edge_cost.item())) * 0.1
            costs['lat'][0, 2] = (lat_cost  if isinstance(lat_cost,  float) else float(lat_cost.item()))  * 0.3
            costs['logic'][0, 2] = dec_logic_cost
        return costs
    
    costs = compute_agent_specific_costs(edge_cost, lat_cost, logic_cost, gb_summary, tr_summary, dec_summary)

    # Build lightweight context for computing real current-policy log-probs
    cov_vals = []
    try:
        for s in step_logs:
            if getattr(s, 'agent', None) == 'dec':
                cv = getattr(s, 'coverage_loss', None)
                if cv is not None:
                    try:
                        cov_vals.append(float(cv))
                    except Exception:
                        cov_vals.append(float(getattr(cv, 'item', lambda: 0.0)()))
    except Exception:
        pass
    coverage_sum = float(sum(cov_vals)) if cov_vals else 0.0
    coverage_tensor = torch.tensor(coverage_sum, device=device, dtype=torch.float32)
    ctx = {
        'gb_obs': recorded_actions.get('gb_obs', [])[-1] if recorded_actions.get('gb_obs') else {},
        'tr_obs': recorded_actions.get('tr_obs', [])[-1] if recorded_actions.get('tr_obs') else {},
        'dec_obs': recorded_actions.get('dec_obs', [])[-1] if recorded_actions.get('dec_obs') else {},
        'gb_action': recorded_actions.get('gb_actions', [])[-1] if recorded_actions.get('gb_actions') else None,
        'tr_action': recorded_actions.get('tr_actions', [])[-1] if recorded_actions.get('tr_actions') else None,
        'dec_action': recorded_actions.get('dec_actions', [])[-1] if recorded_actions.get('dec_actions') else None,
        'dec_rule_masks': recorded_actions.get('dec_rule_masks', [])[-1] if recorded_actions.get('dec_rule_masks') else [],
        'dec_prev_tokens': recorded_actions.get('dec_prev_tokens', [])[-1] if recorded_actions.get('dec_prev_tokens') else [],
        'dec_vocab_size': dec_vocab_size,
        'violation_stats': violation_stats,
        'coverage_loss': coverage_tensor,
    }

    return (state, obs, dll, gnn, actions, logp_old, rewards, costs, ctx)
