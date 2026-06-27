"""Real-model stress-test harness.

Swaps the grammar oracle of the paper for a pre-trained masked-diffusion code
model (Salesforce/CoDA-v0-Instruct) and evaluates the paper's unmasking
schedules / samplers on HumanEval(+) and MBPP(+) via EvalPlus.

The schedule + sampler math here is ported verbatim from ``src/schedules/`` so
that behaviour matches the oracle experiments, but decoupled from the grammar
token conventions (no hard-coded ``MASK_token=5``; the real model's own mask id
is passed in) and from ``datasets.constants`` (which collides with the
HuggingFace ``datasets`` package that EvalPlus imports).
"""
