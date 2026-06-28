"""Thin wrapper around Salesforce/CoDA-v0-Instruct.

Exposes exactly what the decoding loop needs:
  - the tokenizer, mask-token id, eos id, vocab size
  - a prompt builder (chat template)
  - ``logits(canvas_ids)`` : a single diffusion forward pass over a batch of
    partially-masked canvases, returning per-position logits.

This is the drop-in replacement for ``oracleModel.forward`` — it returns LOGITS
over the full BPE vocab (the ``oracle=False`` path), not exact probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_MODEL = "Salesforce/CoDA-v0-Instruct"


@dataclass
class CodaDenoiser:
    model: object
    tokenizer: object
    mask_id: int
    eos_id: int
    pad_id: int
    vocab_size: int
    device: str

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL, device: str = "cuda",
             dtype: str = "bfloat16") -> "CodaDenoiser":
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # CoDA auto_map maps AutoModel -> CoDALanguageModel (not AutoModelForCausalLM).
        # attn_implementation="eager" required: inner CoDAModel doesn't support SDPA.
        # rope_scaling={'rope_type':'default',...} uses a newer HF format not accepted by
        # CoDA's own RopeScaling dataclass; rope_type='default' means no special scaling,
        # and rope_theta is already in config.rope_theta, so nullifying is correct.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        cfg.rope_scaling = None
        model = AutoModel.from_pretrained(
            model_name, config=cfg, trust_remote_code=True, torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        model = model.to(device).eval()

        gen_cfg = getattr(model, "generation_config", None)
        mask_id = _resolve_mask_id(model, tok, gen_cfg)
        eos_id = _first_not_none(
            getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None,
            tok.eos_token_id,
        )
        eos_id = eos_id[0] if isinstance(eos_id, (list, tuple)) else eos_id
        pad_id = _first_not_none(tok.pad_token_id, eos_id)
        vocab = model.config.vocab_size

        obj = cls(model, tok, int(mask_id), int(eos_id), int(pad_id), int(vocab), device)
        print(f"[CodaDenoiser] mask_id={obj.mask_id} eos_id={obj.eos_id} "
              f"pad_id={obj.pad_id} vocab={obj.vocab_size} dtype={dtype}")
        return obj

    # ── prompts ───────────────────────────────────────────────────────────────
    def build_prompt_ids(self, instruction: str) -> torch.Tensor:
        """Chat-format an instruction and return a 1-D LongTensor of prompt ids."""
        messages = [{"role": "user", "content": instruction}]
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )[0]
        return ids.to(self.device)

    def build_prompt_ids_instruct(self, user_msg: str, gen_prefix: str) -> torch.Tensor:
        """Official CoDA eval format: chat template + generation prefix as one string.

        Matches eval_mbpp_humaneval.sh: the gen_prefix is fixed context, not masked.
        Tokenize full string at once to avoid boundary tokenization artefacts.
        """
        chat_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer(chat_str + gen_prefix, return_tensors="pt").input_ids[0]
        return ids.to(self.device)

    def make_canvas(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                    batch: int) -> torch.Tensor:
        """[B, P + max_new_tokens] with the tail filled by mask tokens."""
        P = prompt_ids.shape[0]
        canvas = torch.full((batch, P + max_new_tokens), self.mask_id,
                            dtype=torch.long, device=self.device)
        canvas[:, :P] = prompt_ids.unsqueeze(0)
        return canvas

    # ── forward ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def logits(self, canvas_ids: torch.Tensor) -> torch.Tensor:
        """One denoising forward pass. canvas_ids: (B, L) -> logits (B, L, V).

        CoDA.forward() at inference returns (logits, None). The logits are then
        right-shifted by 1 (repeating position 0) to align causal predictions
        with their target positions — matching CoDA's _sample() convention.
        """
        raw_logits, _ = self.model(input_ids=canvas_ids)
        # right-shift: shifted[:, i] = raw[:, i-1] for i >= 1
        return torch.cat([raw_logits[:, :1], raw_logits[:, :-1]], dim=1)

    def decode_completion(self, generated_ids: torch.Tensor) -> str:
        """Decode the generated region, truncating at the first EOS."""
        ids = generated_ids.tolist()
        if self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def _resolve_mask_id(model, tok, gen_cfg) -> int:
    # Preferred: generation_config.mask_token_id (CoDA sets this).
    if gen_cfg is not None and getattr(gen_cfg, "mask_token_id", None) is not None:
        return gen_cfg.mask_token_id
    if getattr(model.config, "mask_token_id", None) is not None:
        return model.config.mask_token_id
    if getattr(tok, "mask_token_id", None) is not None:
        return tok.mask_token_id
    raise ValueError(
        "Could not resolve mask_token_id from generation_config, model.config, "
        "or tokenizer. Inspect the model's generation_utils.py and pass it "
        "explicitly."
    )


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None
