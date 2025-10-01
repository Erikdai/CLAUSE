# mappo.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn


# ----------------------------- Data structures ----------------------------- #

from .common import StepLog


# ----------------------------- Central critic ----------------------------- #

class CentralCritic(nn.Module):
    """
    Enhanced centralized value function V(s) for MAPPO that captures inter-agent dependencies.
    
    Expected state features:
    [step_idx, num_nodes, num_edges, est_tokens, path_len, rule_count, out_len,
     gb_tr_alignment, tr_dec_alignment, pipeline_coherence, bias]
    
    This allows the critic to learn:
    - How well GB subgraphs support TR path finding
    - How well TR paths support DEC answer generation  
    - Overall pipeline coherence and coordination
    """
    def __init__(self, obs_dim: int = 11, hidden: int = 64):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for better generalization
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        # Accepts [T, D] or [D]; returns [T] or scalar accordingly.
        out = self.v(state_feat)
        return out.squeeze(-1)


# ----------------------------- MAPPO trainer ----------------------------- #

class SimpleMAPPO:
    """
    Minimal MAPPO:
      - Three decentralized actors (gb, tr, dec) are optimized jointly
        with a single Adam along with the centralized critic.
      - PPO-style clipped objective for the actors.
      - Optional DLL coverage loss (eta * mean(coverage_loss)).

    IMPORTANT:
    To have a proper PPO ratio, you must provide NEW vs OLD log-probs.
    Do it either by:
      * Setting StepLog.old_log_prob at rollout and StepLog.new_log_prob
        at update time, OR
      * Passing `new_log_probs` (tensor [T]) directly into update().
    """

    def __init__(
        self,
        gb: nn.Module,
        tr: nn.Module,
        dec: nn.Module,
        critic: CentralCritic,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        ent_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        self.gb, self.tr, self.dec, self.critic = gb, tr, dec, critic
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        params = (
            list(gb.parameters())
            + list(tr.parameters())
            + list(dec.parameters())
            + list(critic.parameters())
        )
        self.optim = torch.optim.Adam(params, lr=lr)

    def update(
        self,
        step_logs: List[StepLog],
        state_feats: List[torch.Tensor],
        new_log_probs: Optional[torch.Tensor] = None,  # optional [T]
        eta: float = 0.0,  # weight for DLL coverage term
    ) -> Dict[str, float]:
        """
        Perform one PPO-style update over the given rollout.

        Args
        ----
        step_logs:    sequence of StepLog (length T).
        state_feats:  list of tensors (length T), each [D] for centralized critic.
        new_log_probs:
            Optional [T] tensor of *current* log-probs, if you prefer passing
            them directly rather than setting StepLog.new_log_prob.
        eta:
            Weight for optional DLL coverage loss. Set >0 to activate.

        Returns
        -------
        Dict[str, float]: diagnostics (loss components, entropy, coverage).
        """
        if len(step_logs) == 0:
            raise ValueError("update(): empty step_logs.")
        if len(state_feats) != len(step_logs):
            raise ValueError("update(): state_feats length must match step_logs length.")

        device = next(self.critic.parameters()).device

        # ----- Build tensors -----
        rewards = torch.tensor([float(s.reward) for s in step_logs], dtype=torch.float32, device=device)

        # Monte Carlo returns (simple, stable for short rollouts)
        returns_list: List[float] = []
        G = 0.0
        for r in reversed(rewards.tolist()):
            G = r + self.gamma * G
            returns_list.append(G)
        returns = torch.tensor(list(reversed(returns_list)), dtype=torch.float32, device=device)

        state_tensor = torch.stack(
            [sf if sf.dim() == 1 else sf.squeeze(0) for sf in state_feats]
        ).to(device)  # [T, D]

        with torch.no_grad():
            values_old = self.critic(state_tensor)
        advantages = returns - values_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_lp = torch.stack([s.get_old_lp() for s in step_logs]).to(device)  # [T]

        if new_log_probs is not None:
            new_lp = new_log_probs.reshape(-1).to(device)
        else:
            new_lp = torch.stack([s.get_new_lp() for s in step_logs]).to(device)

        entropies = torch.stack([s.entropy.reshape(()) for s in step_logs]).to(device)
        entropy_mean = entropies.mean()

        # ----- PPO clipped objective -----
        ratio = torch.exp(new_lp - old_lp)                 # ✅ correct ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        actor_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

        # ----- Critic loss -----
        values_pred = self.critic(state_tensor)
        critic_loss = torch.mean((returns - values_pred) ** 2)

        # ----- Optional DLL coverage term -----
        if eta > 0.0:
            cov_terms: List[torch.Tensor] = []
            for s in step_logs:
                if s.coverage_loss is not None:
                    cov_terms.append(s.coverage_loss.reshape(()).to(device))
            coverage = torch.stack(cov_terms).mean() if cov_terms else torch.tensor(0.0, device=device)
        else:
            coverage = torch.tensor(0.0, device=device)

        # ----- Total loss -----
        loss = (
            actor_loss
            - self.ent_coef * entropy_mean
            + self.value_coef * critic_loss
            + eta * coverage
        )

        # ----- Optimize -----
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.gb.parameters())
                + list(self.tr.parameters())
                + list(self.dec.parameters())
                + list(self.critic.parameters()),
                self.max_grad_norm,
            )
        self.optim.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_mean.item()),
            "coverage": float(coverage.item()),
        }
