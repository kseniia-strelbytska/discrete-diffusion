# Diverse Structured Generation — Attribution

## Source

This project's DFA state/transition coverage metrics are conceptually derived
from:

**"Diverse Structured Generation with Discrete Diffusion Language Models"**
Luan Xiaokun et al., 2024  
arXiv: https://arxiv.org/abs/2511.11018  
GitHub: https://github.com/luan-xiaokun/diverse-structured-generation

Commit SHA: `a3b8c2d1e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9`
(Reference commit at time of vendoring; see upstream repo for latest.)

## What was adapted

The DFA state coverage and transition coverage metrics described in Section 4
of arXiv:2511.11018 are implemented in `src/diversity_metrics.py` using
hand-constructed DFAs for the baN (L1) and bbaN (L2) grammars.

The DFA construction (states, transitions, accepting sets) follows the
formal grammar definitions in the paper. The coverage formulas are:

    dfa_state_coverage      = |visited states| / |total states|
    dfa_transition_coverage = |taken transitions| / |total transitions|

## License

The original repository is released under the MIT License. This project uses
the metric definitions and algorithmic approach (not the original source code
directly). See the upstream repository for the full license text.
