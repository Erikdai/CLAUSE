# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn


from .common import StepLog


class PerAgentValue(nn.Module):
    def __init__(self, obs_dim: int = 11, hidden: int = 64):
        super().__init__()
        self.gb = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.tr = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.dec = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, agent: str, state_feat: torch.Tensor) -> torch.Tensor:
        if agent == 'gb':
            return self.gb(state_feat).squeeze(-1)
        if agent == 'tr':
            return self.tr(state_feat).squeeze(-1)
        return self.dec(state_feat).squeeze(-1)


class IPPO:
    def __init__(
        self,
        gb: nn.Module,
        tr: nn.Module,
        dec: nn.Module,
        value_net: Optional[PerAgentValue] = None,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        ent_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        self.gb, self.tr, self.dec = gb, tr, dec
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.value = value_net or PerAgentValue()
        self.opt_gb = torch.optim.Adam(self.gb.parameters(), lr=lr)
        self.opt_tr = torch.optim.Adam(self.tr.parameters(), lr=lr)
        self.opt_dec = torch.optim.Adam(self.dec.parameters(), lr=lr)
        self.opt_v = torch.optim.Adam(self.value.parameters(), lr=lr)

    def _ppo_objective(self, old_lp, new_lp, advantages):
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        return -torch.mean(torch.min(ratio * advantages, clipped * advantages))

    def update(
        self,
        step_logs: List[StepLog],
        state_feats: List[torch.Tensor],
        new_log_probs: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> Dict[str, float]:
        if not step_logs:
            raise ValueError("IPPO.update(): empty step_logs")

        device = next(self.gb.parameters()).device
        # Build per-agent tensors (keep order alignment with step_logs)
        states = torch.stack([sf if sf.dim() == 1 else sf.squeeze(0) for sf in state_feats]).to(device)
        rewards = torch.tensor([float(s.reward) for s in step_logs], dtype=torch.float32, device=device)

        # Monte Carlo returns per step (shared for simplicity)
        ret_list, G = [], 0.0
        for r in reversed(rewards.tolist()):
            G = r + self.gamma * G
            ret_list.append(G)
        returns = torch.tensor(list(reversed(ret_list)), dtype=torch.float32, device=device)

        # Compute per-step values from corresponding agent head
        values = torch.stack([self.value(s.agent, st) for s, st in zip(step_logs, states)])
        adv = returns - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_lp = torch.stack([s.get_old_lp() for s in step_logs]).to(device)
        if new_log_probs is not None:
            new_lp = new_log_probs.reshape(-1).to(device)
        else:
            new_lp = torch.stack([s.get_new_lp() for s in step_logs]).to(device)
        ent = torch.stack([s.entropy.reshape(()) for s in step_logs]).to(device).mean()

        # Total actor loss is sum of per-agent actor losses over their own steps
        actor_loss = self._ppo_objective(old_lp, new_lp, adv)
        critic_loss = torch.mean((returns - values) ** 2)
        loss = actor_loss - self.ent_coef * ent + self.value_coef * critic_loss

        # Optimize actors independently (one backward, step per optimizer)
        for opt in (self.opt_gb, self.opt_tr, self.opt_dec, self.opt_v):
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.gb.parameters()), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(list(self.tr.parameters()), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(list(self.dec.parameters()), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(list(self.value.parameters()), self.max_grad_norm)
        self.opt_gb.step(); self.opt_tr.step(); self.opt_dec.step(); self.opt_v.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(ent.item()),
        }


