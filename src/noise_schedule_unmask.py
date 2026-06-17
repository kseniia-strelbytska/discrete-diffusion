import torch
import torch.nn as nn
from datasets.constants import EOS_token, SOS_token, PAD_token, MASK_token
from schedules.decoding_strategy import ScheduleDrivenDecoding, EBSamplerDecoding, AutoregressiveDecoding
from schedules.sampling_strategy import GreedySampling, CategoricalSampling


_DECODING_ALIASES = {
    'schedule': ScheduleDrivenDecoding,
    'schedule_driven': ScheduleDrivenDecoding,
    'eb': EBSamplerDecoding,
    'ebsampler': EBSamplerDecoding,
    'autoregressive': AutoregressiveDecoding,
    'ar': AutoregressiveDecoding,
}

_SAMPLING_ALIASES = {
    'greedy': GreedySampling,
    'categorical': CategoricalSampling,
}


class ScheduledUnmasker(nn.Module):
    def __init__(self, model, device, T=100, denoise="0", oracle=False, oracle_model=None,
                 schedule=None, gaussian_noise=False, sigma=1.0,
                 decoding_strategy=None, sampling_strategy=None, eb_gamma=0.1):
        """
        Args:
            schedule:          A NoiseSchedule instance.  When provided it is used for
                               all alpha_t / alpha_s computations; the legacy
                               gaussian_noise / sigma flags are ignored.
            gaussian_noise:    Legacy flag — kept for backward compatibility.
            sigma:             Legacy Gaussian sigma — kept for backward compatibility.
            decoding_strategy: DecodingStrategy instance, string alias, or None.
                               None → ScheduleDrivenDecoding (schedule-based Bernoulli draws).
                               Aliases: 'schedule', 'schedule_driven', 'eb', 'ebsampler'.
            sampling_strategy: SamplingStrategy instance, string alias, or None.
                               None → resolved lazily in forward() from the temperature arg
                               (temperature <= 0 → GreedySampling, else CategoricalSampling).
                               Aliases: 'greedy', 'categorical'.
            eb_gamma:          Gamma hyperparameter for EBSamplerDecoding when
                               decoding_strategy is 'eb' / 'ebsampler'.
        """
        super().__init__()
        self.model = model
        self.device = device
        self.T = T
        self.denoise = denoise
        self.oracle = oracle
        self.oracle_model = oracle_model

        # Build schedule from legacy flags when no explicit schedule is provided.
        if schedule is not None:
            self._schedule = schedule
        elif gaussian_noise:
            from schedules.gaussian_schedule import GaussianSchedule
            self._schedule = GaussianSchedule(sigma)
        else:
            from schedules.categorical_schedule import CategoricalSchedule
            self._schedule = CategoricalSchedule()

        self.gaussian_noise = gaussian_noise
        self.sigma = sigma

        # Build decoding strategy.  Schedule-driven variants receive self._schedule
        # so they can own the alpha/weight/mask_prob calculations internally.
        if decoding_strategy is None:
            self.decoding_strategy = ScheduleDrivenDecoding(self._schedule)
        elif isinstance(decoding_strategy, str):
            key = decoding_strategy.lower()
            if key not in _DECODING_ALIASES:
                raise ValueError(f"Unknown decoding_strategy: {decoding_strategy!r}")
            if key in ('eb', 'ebsampler'):
                self.decoding_strategy = EBSamplerDecoding(gamma=eb_gamma)
            elif key in ('autoregressive', 'ar'):
                self.decoding_strategy = AutoregressiveDecoding()
            else:
                self.decoding_strategy = ScheduleDrivenDecoding(self._schedule)
        else:
            self.decoding_strategy = decoding_strategy

        # Sampling strategy: None is resolved lazily in forward() for backward compat.
        if sampling_strategy is None:
            self.sampling_strategy = None
        elif isinstance(sampling_strategy, str):
            cls = _SAMPLING_ALIASES.get(sampling_strategy.lower())
            if cls is None:
                raise ValueError(f"Unknown sampling_strategy: {sampling_strategy!r}")
            self.sampling_strategy = cls()
        else:
            self.sampling_strategy = sampling_strategy

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_logits(self, X, timestep):
        if self.oracle:
            return self.model(X)
        return self.model(X.unsqueeze(0), timestep.unsqueeze(0))[0]

    def _content_probs(self, logits, temperature):
        V = logits.shape[-1]
        content_idx = torch.tensor(
            [j for j in range(V) if j != MASK_token],
            device=logits.device, dtype=torch.long,
        )
        raw = logits[:, content_idx]
        scaled = raw / temperature if temperature > 0 else raw
        probs = scaled if self.oracle else torch.softmax(scaled, dim=-1)
        return content_idx, probs

    def _error_fmt(self, step, X, error_changed_mask, error_probs, error_logits, exc):
        changed = (
            error_changed_mask.nonzero(as_tuple=True)[0].tolist()
            if error_changed_mask is not None else []
        )
        msg = f'Oracle failed at step {step} with input {X}.\nMessage:{exc}\nInvestigating probs and logits:\n'
        for tok in changed:
            msg += (f'\nToken index {tok}'
                    f'\nprob={error_probs[tok].cpu().numpy()}'
                    f'\nlogit={error_logits[tok].cpu().numpy()}'
                    f'\nChoice: {X[tok].item()}\n')
        return msg

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, init_X, timestep, strategy='categorical', temperature=1.0, return_steps=False, eps=1e-5):
        X = init_X.clone().long().to(self.device)
        timestep = timestep.clone().to(self.device)

        # Resolve sampling strategy. Explicit constructor arg wins; otherwise
        # derive from the temperature arg for backward compatibility.
        sampling_strategy = (
            self.sampling_strategy if self.sampling_strategy is not None
            else (GreedySampling() if temperature <= 0 else CategoricalSampling())
        )

        if not self.oracle:
            self.model.eval()

        error_message = 'No errors occured.'
        error_probs, error_logits, error_changed_mask = None, None, None

        with torch.no_grad():
            if self.decoding_strategy.needs_schedule:
                # ── Schedule-driven loop ───────────────────────────────────────
                # Pre-compute a timestep tensor; ScheduleDrivenDecoding owns the
                # per-step alpha / weight / mask_prob calculations.
                if self.denoise == "eps":
                    num_steps = int(self.T * timestep)
                    timesteps = torch.linspace(timestep, eps, num_steps + 1, device=self.device)
                    dt = (timestep - eps) / num_steps
                elif self.denoise == "0":
                    num_steps = int(torch.ceil(timestep * self.T).item())
                    timestep = num_steps / self.T
                    timesteps = torch.linspace(timestep, 0, num_steps + 1, device=self.device)
                    dt = 1 / self.T
                else:
                    raise ValueError(f"{self.denoise} is not defined")

                steps = [X.clone()]
                timesteps_log = [timestep]
                mopup_timestep = timesteps[-1]  # fallback for mop-up

                for i in range(num_steps):
                    if timesteps[i] <= 0 or not (X == MASK_token).any():
                        break

                    mopup_timestep = timesteps[i]

                    try:
                        logits = self._get_logits(X, timesteps[i])
                    except ValueError as e:
                        error_message = self._error_fmt(i, X, error_changed_mask, error_probs, error_logits, e)
                        break

                    if not self.oracle and self.oracle_model is not None:
                        try:
                            self.oracle_model.forward(X)
                        except ValueError as e:
                            error_message = self._error_fmt(i, X, error_changed_mask, error_probs, error_logits, e)
                            break

                    content_idx, content_probs = self._content_probs(logits, temperature)
                    masked_mask = (X == MASK_token)

                    positions_mask = self.decoding_strategy.select_positions(
                        X=X, content_probs=content_probs, masked_mask=masked_mask,
                        timestep=timesteps[i], dt=dt, device=self.device,
                    )

                    # Diagnostic probability tensor built from the schedule math that
                    # ScheduleDrivenDecoding just computed and stored as attributes.
                    probs = torch.zeros_like(logits)
                    probs[:, content_idx] = content_probs * self.decoding_strategy.last_weight
                    probs[:, MASK_token] = self.decoding_strategy.last_mask_prob
                    probs = probs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    probs[probs.sum(dim=-1) == 0, MASK_token] = 1.0

                    chosen = sampling_strategy.choose_tokens(
                        content_probs=content_probs, content_idx=content_idx,
                        positions_mask=positions_mask, device=self.device,
                    )
                    apply = positions_mask & masked_mask

                    error_probs, error_logits, error_changed_mask = probs, logits, apply
                    X[apply] = chosen[apply]
                    steps.append(X.clone())
                    timesteps_log.append(timesteps[i] - dt)

            else:
                # ── Adaptive loop (EBSamplerDecoding) ─────────────────────────
                # No pre-computed timestep tensor.  Run until fully unmasked or
                # MAX_STEPS is exhausted.  All schedule math is omitted.
                steps = [X.clone()]
                initial_masked = max((X == MASK_token).sum().item(), 1)
                timesteps_log = [float(timestep)]
                mopup_timestep = torch.tensor(0.0, device=self.device)

                # AutoregressiveDecoding unmasks one token per step, so the
                # number of steps equals the sequence length — no T needed.
                if isinstance(self.decoding_strategy, AutoregressiveDecoding):
                    max_steps = X.shape[0]
                else:
                    max_steps = getattr(self.decoding_strategy, 'MAX_STEPS', 10**5)

                for i in range(max_steps):
                    if not (X == MASK_token).any():
                        break

                    try:
                        logits = self._get_logits(X, mopup_timestep)
                    except ValueError as e:
                        error_message = self._error_fmt(i, X, error_changed_mask, error_probs, error_logits, e)
                        break

                    if not self.oracle and self.oracle_model is not None:
                        try:
                            self.oracle_model.forward(X)
                        except ValueError as e:
                            error_message = self._error_fmt(i, X, error_changed_mask, error_probs, error_logits, e)
                            break

                    content_idx, content_probs = self._content_probs(logits, temperature)
                    masked_mask = (X == MASK_token)

                    positions_mask = self.decoding_strategy.select_positions(
                        X=X, content_probs=content_probs, masked_mask=masked_mask,
                        device=self.device,
                    )

                    # Diagnostic: no weight/mask_prob available; use content_probs directly.
                    diag_probs = torch.zeros_like(logits)
                    diag_probs[:, content_idx] = content_probs

                    chosen = sampling_strategy.choose_tokens(
                        content_probs=content_probs, content_idx=content_idx,
                        positions_mask=positions_mask, device=self.device,
                    )
                    apply = positions_mask & masked_mask

                    error_probs, error_logits, error_changed_mask = diag_probs, logits, apply
                    X[apply] = chosen[apply]
                    steps.append(X.clone())
                    n_remaining = (X == MASK_token).sum().item()
                    timesteps_log.append(n_remaining / initial_masked)

            # ── Mop-up: ensure no MASK tokens remain ──────────────────────────
            if (X == MASK_token).any():
                if self.oracle:
                    try:
                        logits = self.model(X)
                        do_mopup = True
                    except ValueError:
                        do_mopup = False
                else:
                    logits = self.model(X.unsqueeze(0), mopup_timestep.unsqueeze(0))[0]
                    do_mopup = True

                if do_mopup:
                    content_idx, content_probs = self._content_probs(logits, temperature)
                    
                    remaining_masked = (X == MASK_token)
                    chosen_mu = sampling_strategy.choose_tokens(
                        content_probs=content_probs, content_idx=content_idx,
                        positions_mask=remaining_masked, device=self.device,
                    )
                    X[remaining_masked] = chosen_mu[remaining_masked]
                    steps.append(X.clone())
                    timesteps_log.append(0.0)

        if return_steps:
            return X, steps, timesteps_log, error_message
        return X
