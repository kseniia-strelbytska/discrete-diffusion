import torch
import torch.nn as nn
from datasets.constants import EOS_token, SOS_token, PAD_token, MASK_token
from schedules.decoding_strategy import ScheduleDrivenDecoding, EBSamplerDecoding
from schedules.sampling_strategy import GreedySampling, CategoricalSampling


_DECODING_ALIASES = {
    'schedule': ScheduleDrivenDecoding,
    'schedule_driven': ScheduleDrivenDecoding,
    'eb': EBSamplerDecoding,
    'ebsampler': EBSamplerDecoding,
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
                               all alpha_t / alpha_s computations and the legacy
                               gaussian_noise / sigma flags are ignored.
            gaussian_noise:    Legacy flag — kept for backward compatibility.
            sigma:             Legacy Gaussian sigma — kept for backward compatibility.
            decoding_strategy: DecodingStrategy instance or string alias, or None.
                               None → ScheduleDrivenDecoding (schedule-based Bernoulli draws).
                               Supported aliases: 'schedule', 'schedule_driven', 'eb', 'ebsampler'.
            sampling_strategy: SamplingStrategy instance or string alias, or None.
                               None → resolved lazily in forward() from the temperature arg
                               (temperature <= 0 → GreedySampling, else CategoricalSampling).
                               Supported aliases: 'greedy', 'categorical'.
            eb_gamma:          Gamma hyperparameter for EBSamplerDecoding when decoding_strategy
                               is specified as a string alias ('eb' / 'ebsampler').
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

        # Keep legacy attributes so existing code that reads them still works.
        self.gaussian_noise = gaussian_noise
        self.sigma = sigma

        # Build decoding strategy.
        if decoding_strategy is None:
            self.decoding_strategy = ScheduleDrivenDecoding()
        elif isinstance(decoding_strategy, str):
            key = decoding_strategy.lower()
            cls = _DECODING_ALIASES.get(key)
            if cls is None:
                raise ValueError(f"Unknown decoding_strategy: {decoding_strategy!r}")
            if key in ('eb', 'ebsampler'):
                self.decoding_strategy = cls(gamma=eb_gamma)
            else:
                self.decoding_strategy = cls()
        else:
            self.decoding_strategy = decoding_strategy

        # Sampling strategy may be None; resolved lazily in forward() for backward compat.
        if sampling_strategy is None:
            self.sampling_strategy = None
        elif isinstance(sampling_strategy, str):
            cls = _SAMPLING_ALIASES.get(sampling_strategy.lower())
            if cls is None:
                raise ValueError(f"Unknown sampling_strategy: {sampling_strategy!r}")
            self.sampling_strategy = cls()
        else:
            self.sampling_strategy = sampling_strategy

    def forward(self, init_X, timestep, strategy='categorical', temperature=1.0, return_steps=False, eps=1e-5):
        X = init_X.clone().long().to(self.device)
        timestep = timestep.clone().to(self.device)
        L = X.shape[0]

        # Scale down the number of denoising steps according to noise level.
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

        steps, timesteps_log = [X.clone()], [timestep]

        if not self.oracle:
            self.model.eval()

        # Resolve sampling strategy. Explicit constructor arg wins; otherwise derive
        # from the temperature forward arg for backward compatibility with call sites
        # that pass temperature=0 (greedy) or temperature>0 (categorical) to forward().
        if self.sampling_strategy is not None:
            sampling_strategy = self.sampling_strategy
        elif temperature <= 0:
            sampling_strategy = GreedySampling()
        else:
            sampling_strategy = CategoricalSampling()

        error_message = 'No errors occured.'
        error_probs, error_logits, error_changed_mask = None, None, None
        
        if isinstance(self.decoding_strategy, EBSamplerDecoding):
            num_steps = 10**4
            # the EB sampler is adaptive; the number of timesteps is not fixed in advance. 
            # for simplicity, we use a large number of timesteps and break early when all tokens are unmasked.
        
        with torch.no_grad():
            for i in range(num_steps):
                if timesteps[i] <= 0:
                    break
                
                if (X == MASK_token).sum() == 0:
                    # Early stop if all tokens are unmasked (no MASK tokens remain).
                    break

                # Linear schedule: α_t = 1 - t, where α_t is the proportion of original
                # content retained at step t.  t=0 (clean) → α_t=1; t=1 (masked) → α_t=0.
                # s < t → α_s > α_t → more content retained at step s than t.
                alpha_t = 1 - self._schedule.p_mask(timesteps[i], max_l=L, device=self.device)
                alpha_s = 1 - self._schedule.p_mask(timesteps[i] - dt, max_l=L, device=self.device)

                if not self.oracle:
                    logits = self.model(X.unsqueeze(0), timesteps[i].unsqueeze(0))[0]  # (L, V)

                    if self.oracle_model is not None:
                        try:
                            self.oracle_model.forward(X)
                        except ValueError as e:
                            changed_tokens = error_changed_mask.nonzero(as_tuple=True)[0].tolist() if error_changed_mask is not None else []
                            error_message = f'''Oracle failed at step {i} with input {X}.
                            Message:{e}
                            Investigating probs and logits:
                            '''
                            for token in changed_tokens:
                                error_message += f'\nToken index {token}\nprob={error_probs[token].cpu().numpy()}\nlogit={error_logits[token].cpu().numpy()}\nChoice: {X[token].item()}\n'
                            break
                else:
                    try:
                        logits = self.model(X)
                    except ValueError as e:
                        changed_tokens = error_changed_mask.nonzero(as_tuple=True)[0].tolist() if error_changed_mask is not None else []
                        error_message = f'''Oracle failed at step {i} with input {X}.
                        Message:{e}
                        Investigating probs and logits:
                        '''
                        for token in changed_tokens:
                            error_message += f'\nToken index {token}\nprob={error_probs[token].cpu().numpy()}\nlogit={error_logits[token].cpu().numpy()}\nChoice: {X[token].item()}\n'
                        break

                # Convert to probabilities (x_θ in the paper).
                # MASK_token is not the last token for grammars with vocab_size > 6,
                # so index by position rather than assuming MASK is at :-1.
                V = logits.shape[-1]
                content_idx = torch.tensor([j for j in range(V) if j != MASK_token],
                                           device=logits.device, dtype=torch.long)
                raw_content = logits[:, content_idx]  # (L, V-1)
                scaled_logits = raw_content / temperature if temperature > 0 else raw_content
                if self.oracle:
                    content_probs = scaled_logits  # already probabilities
                else:
                    content_probs = torch.softmax(scaled_logits, dim=-1)

                # The following probs are not used for EB Sampler. 
                weight = ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(min=0.0)
                mask_prob = ((1 - alpha_s) / (1 - alpha_t)).clamp(min=0.0, max=1.0)
                weight = weight.squeeze(0).unsqueeze(-1)  # (L, 1)
                mask_prob = mask_prob.squeeze(0)          # (L,)

                # Diagnostic probability tensor for oracle error tracking.
                probs = torch.zeros_like(logits)
                probs[:, content_idx] = content_probs * weight
                probs[:, MASK_token] = mask_prob
                probs = probs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                zero_rows = probs.sum(dim=-1) == 0
                probs[zero_rows, MASK_token] = 1.0

                masked_mask = (X == MASK_token)

                positions_mask = self.decoding_strategy.select_positions(
                    X=X, content_probs=content_probs, mask_prob=mask_prob,
                    masked_mask=masked_mask, step=i, num_steps=num_steps, device=self.device,
                )
                chosen = sampling_strategy.choose_tokens(
                    content_probs=content_probs, content_idx=content_idx,
                    positions_mask=positions_mask, device=self.device,
                )

                # Apply only at selected, currently-masked positions.
                # For schedule-driven decoding + categorical sampling this is equivalent
                # to the old joint multinomial over [content_probs * weight, mask_prob]:
                # the joint factorises as Bernoulli(1 - mask_prob) × Categorical(content_probs)
                # because weight = 1 - mask_prob, and the choice of content token is
                # independent of the unmask/stay-masked decision given we unmask.
                apply = positions_mask & masked_mask

                error_probs = probs
                error_logits = logits
                error_changed_mask = apply

                X[apply] = chosen[apply]
                steps.append(X.clone())
                timesteps_log.append(timesteps[i] - dt)

            # Ensure no MASK tokens remain (can happen due to numerical issues with the schedule).
            if (X == MASK_token).any():
                if self.oracle:
                    try:
                        logits = self.model(X)
                        do_mopup = True
                    except ValueError:
                        do_mopup = False
                else:
                    do_mopup = True
                    logits = self.model(X.unsqueeze(0), timesteps[i].unsqueeze(0))[0]

                if do_mopup:
                    V_mu = logits.shape[-1]
                    content_idx_mu = torch.tensor([j for j in range(V_mu) if j != MASK_token],
                                                  device=logits.device, dtype=torch.long)
                    probs_content = torch.softmax(logits[:, content_idx_mu], dim=-1)
                    remaining_masked = (X == MASK_token)
                    chosen_mu = sampling_strategy.choose_tokens(
                        content_probs=probs_content, content_idx=content_idx_mu,
                        positions_mask=remaining_masked, device=self.device,
                    )
                    X[remaining_masked] = chosen_mu[remaining_masked]
                    steps.append(X.clone())
                    timesteps_log.append(timesteps[-1] - dt)

        if return_steps == True:
            return X, steps, timesteps_log, error_message
        return X
