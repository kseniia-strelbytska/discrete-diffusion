"""Schedule-driven decoding loop for a real masked-diffusion model.

One unified loop drives every decoder (uniform / Gaussian / EB / AR) so the
comparison is fair: same forward pass, same NFE accounting (1 forward = 1 NFE),
same truncation. This mirrors ScheduledUnmasker.forward but over a real model's
canvas, using the ported math in ``schedules.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from realmodel import schedules as S
from realmodel.coda_denoiser import CodaDenoiser


@dataclass
class DecodeConfig:
    decoder: str = "gaussian"          # uniform | gaussian | eb | ar
    sampler: str = "greedy"            # greedy | categorical
    nfe: int = 32                      # forward passes for uniform/gaussian
    sigma: float = 32.0                # gaussian width (canvas-position units)
    gamma: float = 0.5                 # EB entropy budget (nats)
    temperature: float = 1.0           # used by categorical sampler
    max_new_tokens: int = 256
    num_samples: int = 1               # batch of i.i.d. samples per problem

    def tag(self) -> str:
        if self.decoder == "gaussian":
            extra = f"sigma{self.sigma:g}_nfe{self.nfe}"
        elif self.decoder == "uniform":
            extra = f"nfe{self.nfe}"
        elif self.decoder == "eb":
            extra = f"gamma{self.gamma:g}"
        else:  # ar
            extra = "ar"
        return f"{self.decoder}_{self.sampler}_{extra}"


@dataclass
class DecodeResult:
    completions: List[str]
    realised_nfe: int = 0
    meta: dict = field(default_factory=dict)


def _sample_tokens(content_probs: torch.Tensor, sampler: str) -> torch.Tensor:
    if sampler == "greedy":
        return S.choose_greedy(content_probs)
    return S.choose_categorical(content_probs)


@torch.no_grad()
def generate(den: CodaDenoiser, instruction: str, cfg: DecodeConfig) -> DecodeResult:
    device = den.device
    prompt_ids = den.build_prompt_ids(instruction)
    P = prompt_ids.shape[0]
    M = cfg.max_new_tokens
    B = cfg.num_samples
    canvas = den.make_canvas(prompt_ids, M, B)  # (B, P+M)

    temp = cfg.temperature if cfg.sampler == "categorical" else 0.0
    nfe = 0

    def forward_probs():
        """One forward pass; returns content probs for the canvas region (B, M, V)."""
        logits = den.logits(canvas)[:, P:, :].float()
        return torch.stack([
            S.content_probs_from_logits(logits[b], den.mask_id, temp) for b in range(B)
        ])

    def all_done() -> bool:
        if cfg.decoder == "ar":
            # AR: a row is done once it has committed an EOS (rest is irrelevant).
            return bool(((canvas[:, P:] == den.eos_id).any(dim=1)).all().item())
        return bool((canvas[:, P:] != den.mask_id).all().item())

    if cfg.decoder in ("uniform", "gaussian"):
        timesteps = torch.linspace(1.0, 0.0, cfg.nfe + 1, device=device)
        for i in range(cfg.nfe):
            if all_done():
                break
            probs = forward_probs(); nfe += 1
            t, s = timesteps[i].item(), timesteps[i + 1].item()
            if cfg.decoder == "uniform":
                pm_t, pm_s = S.uniform_p_mask(t, M, device), S.uniform_p_mask(s, M, device)
            else:
                pm_t = S.gaussian_p_mask(t, M, cfg.sigma, device)
                pm_s = S.gaussian_p_mask(s, M, cfg.sigma, device)
            unmask_prob = S.schedule_unmask_prob(pm_t, pm_s)
            for b in range(B):
                masked = (canvas[b, P:] == den.mask_id)
                sel = S.select_schedule(unmask_prob, masked)
                if sel.any():
                    chosen = _sample_tokens(probs[b], cfg.sampler)
                    canvas[b, P:][sel] = chosen[sel]

    elif cfg.decoder == "eb":
        while not all_done() and nfe < M + 1:
            probs = forward_probs(); nfe += 1
            for b in range(B):
                masked = (canvas[b, P:] == den.mask_id)
                if not masked.any():
                    continue
                sel = S.select_eb(probs[b], masked, cfg.gamma)
                chosen = _sample_tokens(probs[b], cfg.sampler)
                canvas[b, P:][sel] = chosen[sel]

    elif cfg.decoder == "ar":
        for _ in range(M):
            if all_done():
                break
            probs = forward_probs(); nfe += 1
            for b in range(B):
                masked = (canvas[b, P:] == den.mask_id)
                sel = S.select_autoregressive(masked)
                if sel.any():
                    chosen = _sample_tokens(probs[b], cfg.sampler)
                    canvas[b, P:][sel] = chosen[sel]
    else:
        raise ValueError(f"unknown decoder {cfg.decoder!r}")

    # Mop-up: fill any slots still masked after the budget (uniform/gaussian).
    if not all_done() and cfg.decoder in ("uniform", "gaussian"):
        probs = forward_probs(); nfe += 1
        for b in range(B):
            masked = (canvas[b, P:] == den.mask_id)
            if masked.any():
                chosen = _sample_tokens(probs[b], cfg.sampler)
                canvas[b, P:][masked] = chosen[masked]

    completions = [den.decode_completion(canvas[b, P:]) for b in range(B)]
    return DecodeResult(completions=completions, realised_nfe=nfe,
                        meta={"prompt_len": P, "max_new_tokens": M})
