"""First verification step (run on the GPU box).

1. Load CoDA, print resolved mask/eos/vocab so the wrapper's token conventions
   are confirmed against the model.
2. Run CoDA's NATIVE model.generate on one prompt — confirms our understanding
   of the interface and gives a reference completion.
3. Run OUR unified loop (uniform/greedy and gaussian/greedy) on the same prompt
   — confirms the harness produces sensible code and matches the native path.

    python -m realmodel.sanity_check
"""

from __future__ import annotations

import torch

from realmodel.coda_denoiser import CodaDenoiser
from realmodel.decode import DecodeConfig, generate

PROMPT = (
    "Complete the following Python function. Return only the complete function "
    "in a markdown code block.\n\n"
    "def has_close_elements(numbers, threshold):\n"
    "    \"\"\" Check if in given list of numbers, are any two numbers closer to\n"
    "    each other than given threshold. \"\"\"\n"
)


def main():
    den = CodaDenoiser.load()

    print("\n=== native model.generate ===")
    try:
        ids = den.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(den.device)
        out = den.model.diffusion_generate(ids, max_new_tokens=256, steps=64, temperature=0.0)
        print(den.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    except Exception as e:  # interface may differ; surfaces the real signature
        print(f"[native generate failed — inspect signature] {e}")

    for decoder, kw in [("uniform", {}), ("gaussian", {"sigma": 32.0})]:
        cfg = DecodeConfig(decoder=decoder, sampler="greedy", nfe=32,
                           max_new_tokens=256, num_samples=1, **kw)
        print(f"\n=== our loop: {cfg.tag()} ===")
        res = generate(den, PROMPT, cfg)
        print(f"realised_nfe={res.realised_nfe}")
        print(res.completions[0])


if __name__ == "__main__":
    main()
