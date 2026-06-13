import torch
import torch.nn as nn
from torch import Tensor

from datasets.constants import EOS_token, MASK_token
from schedules.noise_schedule import NoiseSchedule


class NoiseScheduleLoss(nn.Module):
    """
    Continuous-time ELBO loss for discrete diffusion with an arbitrary noise schedule.

    Implements the schedule-agnostic form of the MDLM ELBO:

        L = E_t [ (dp_mask/dt)(t) / p_mask(t) * (-log p_θ(x0 | xt)) ]

    This unifies:
      • CategoricalSchedule  →  dp/p = 1/t  (equivalent to eq8 in loss.py)
      • GaussianSchedule     →  dp/p = pdf/cdf  (equivalent to GaussianLoss)

    Any new NoiseSchedule only needs to supply p_mask and dp_mask; the rest of
    the training objective is handled here.
    """

    def __init__(
        self,
        device: torch.device,
        vocab_size: int,
        schedule: NoiseSchedule,
        T: int,
        sampling_eps: float = 1e-5,
        eos_weight: float = 1.0,
    ):
        super().__init__()
        self.neg_infinity = -1_000_000.0
        self.device = device
        self.schedule = schedule
        self.T = T
        self.sampling_eps = sampling_eps
        self.eos_weight = eos_weight

    def subs_parameterisation(self, logits: Tensor, xt: Tensor) -> Tensor:
        """
        Apply substitution parameterisation so the model only predicts unmasked tokens.

        For masked positions: normalise across non-MASK vocab.
        For unmasked positions: set logits to neg_infinity except the true token (logit=0).
        Returns log-probabilities.
        """
        logits[:, :, MASK_token] = self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = xt != MASK_token
        logits[unmasked] = self.neg_infinity
        logits[unmasked, xt[unmasked]] = 0
        return logits

    def forward(self, xt: Tensor, logits: Tensor, y_true: Tensor, timestep: Tensor) -> Tensor:
        """
        Args:
            xt:        Noisy input tokens,  (B, L).
            logits:    Model output logits, (B, L, vocab_size).
            y_true:    Clean target tokens, (B, L).
            timestep:  Sampled timesteps,   (B, 1).

        Returns:
            Scalar loss averaged over all token positions.
        """
        logits = logits.clone()
        logits = self.subs_parameterisation(logits, xt)

        max_l = logits.shape[1]
        p  = self.schedule.p_mask(timestep, max_l, self.device).clamp(self.sampling_eps, 1 - self.sampling_eps)
        dp = self.schedule.dp_mask(timestep, max_l, self.device)

        log_prob_x = torch.gather(logits, -1, y_true[:, :, None]).squeeze(-1)  # (B, L)
        loss = dp / p * (-log_prob_x)

        token_weight = torch.ones_like(loss)
        token_weight = token_weight.masked_fill(y_true == EOS_token, self.eos_weight)
        loss = (loss * token_weight).sum() / torch.numel(xt)

        return loss
