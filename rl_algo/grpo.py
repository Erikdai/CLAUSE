# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

import torch
from torch import nn

from .common import StepLog


def ppo_actor_loss(new_lp: torch.Tensor, old_lp: torch.Tensor, advantages: torch.Tensor, clip_eps: float) -> torch.Tensor:
    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return -torch.mean(torch.min(ratio * advantages, clipped * advantages))


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


class GRPO:
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
        params = list(self.gb.parameters()) + list(self.tr.parameters()) + list(self.dec.parameters()) + list(self.value.parameters())
        self.optim = torch.optim.Adam(params, lr=lr)

    def update(
        self,
        step_logs: List[StepLog],
        state_feats: List[torch.Tensor],
        new_log_probs: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> Dict[str, float]:
        if not step_logs:
            raise ValueError("GRPO.update(): empty step_logs")

        device = next(self.gb.parameters()).device

        # Stack states and rewards
        states = torch.stack([sf if sf.dim() == 1 else sf.squeeze(0) for sf in state_feats]).to(device)
        rewards = torch.tensor([float(s.reward) for s in step_logs], dtype=torch.float32, device=device)

        # Monte Carlo returns
        ret_list, G = [], 0.0
        for r in reversed(rewards.tolist()):
            G = r + self.gamma * G
            ret_list.append(G)
        returns = torch.tensor(list(reversed(ret_list)), dtype=torch.float32, device=device)

        # Per-agent value estimates for a learned baseline
        values = torch.stack([self.value(s.agent, st) for s, st in zip(step_logs, states)])

        # Group-relative advantages: center by group mean; also include learned baseline
        # Combine both signals to reduce variance: (returns - values) centered
        adv = (returns - values.detach())
        adv = adv - adv.mean()
        adv = adv / (adv.std() + 1e-8)

        old_lp = torch.stack([s.get_old_lp() for s in step_logs]).to(device)
        if new_log_probs is not None:
            new_lp = new_log_probs.reshape(-1).to(device)
        else:
            new_lp = torch.stack([s.get_new_lp() for s in step_logs]).to(device)
        ent = torch.stack([s.entropy.reshape(()) for s in step_logs]).to(device).mean()

        actor_loss = ppo_actor_loss(new_lp, old_lp, adv, self.clip_eps)
        critic_loss = torch.mean((returns - values) ** 2)
        loss = actor_loss - self.ent_coef * ent + self.value_coef * critic_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.gb.parameters()) + list(self.tr.parameters()) + list(self.dec.parameters()) + list(self.value.parameters()),
                self.max_grad_norm,
            )
        self.optim.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(ent.item()),
        }


