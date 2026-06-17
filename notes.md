Taining samples:
Total number of masks 799865/3695120
Average number of masks per training sample 4.32930459633246/20
Number of masks unmasked as 1s 400484 (0.5006895065307617)

1000 epochs 
model = TransformerClassifier(max_len=16, vocab_size=4, n_head=4, n_layers=4, embed_dim=64, dim_feedforward=128, dropout=0.1)

Accuracy on fully masked sequences: 
Evaluation from generation satisfies rule #1: 859/1000 (0.859)
Evaluation from generation satisfies rule #2: 863/1000 (0.863)
Evaluation from generation satisfies btoh rules: 747/1000 (0.747)

I fixed the anbn diffusion, and I ran an experiment after 15000 epochs of training. 
I test accuracy on the following data: 100 random samples of strings seen in training data 
(even length from 2 to 256 with equal number of 0s (rule 1) and ones and with 0s preceeding 1s (rule 2); strings have format  SOS0000...00111...11EOSPADPADPAD... and padded to each have length 258 (256 + 2 SOS/EOS)). 
Evaluation from generation satisfies rule #1: 32/100 (0.32)
Evaluation from generation satisfies rule #2: 95/100 (0.95)
Evaluation from generation satisfies both rules: 32/100 (0.32)
This is quite bad! But I looked into the generated samples: almost all meet the said format (diffusion learned that SOS goes before 0s before 1s before EOS before PAD). This is amazing since SOS/EOS are quite rare in data (2/258 tokens), and loss doesn't weigh them differently, and results are still high! However, the model still struggles with counting: most generated examples generate 1-2 more tokens '1' or '0' than needed. 


discrete-diffusion ❯ python tests/test_efficiency.py --seq-length 512

Oracle grammar benchmark  |  seq_length=512  n_samples=100
──────────────────────────────────────────────────────────
  grammar                                        avg µs    min µs    max µs
  ──────────────────────────────────────────── ────────  ────────  ────────
  aNbN                                            296.8     276.0     694.1
  baN                                            2378.2    1303.1    3588.5
  bbaN                                          52887.5    6649.9   98636.0
  aNbNcN                                         1336.5    1308.4    1606.8
  parentheses_and_brackets                       2208.3    1371.0    5204.7
  not_nested_parentheses_and_brackets            2942.9    1421.5    4232.9

  (keep in mind when running large oracle evaluations that bbaN grammar's oracle is much less efficient)

UPD: fixed/optimised!

~/Desktop/programming/Projects/AI/discrete-diffusion integrate-formal-grammar-dataset*
discrete-diffusion ❯ python tests/test_efficiency.py --seq-length 512

Oracle grammar benchmark  |  seq_length=512  n_samples=100
──────────────────────────────────────────────────────────
  grammar                                        avg µs    min µs    max µs
  ──────────────────────────────────────────── ────────  ────────  ────────
  aNbN                                            307.1     285.3     671.7
  baN                                            2461.4    1329.9    3449.0
  bbaN                                           1320.9    1030.1   25691.5
  aNbNcN                                         1340.1    1304.9    1599.7
  parentheses_and_brackets                       2353.0    1375.3    4974.7
  not_nested_parentheses_and_brackets            2862.7    1407.2   25806.5

All oracle brute-force L=10 tests pass.

~/Desktop/programming/Projects/AI/discrete-diffusion integrate-formal-grammar-dataset* 25s
discrete-diffusion ❯ pytest tests/test_grammar_oracles.py --seq-length 10 -v

============================================================================= test session starts =============================================================================
platform darwin -- Python 3.12.4, pytest-9.0.3, pluggy-1.6.0 -- /Users/kseniia/Desktop/programming/Projects/AI/discrete-diffusion/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/kseniia
plugins: anyio-4.13.0
collected 38 items                                                                                                                                                            

tests/test_grammar_oracles.py::test_manual_small_sequences PASSED                                                                                                       [  2%]
tests/test_grammar_oracles.py::TestANBN::test_count PASSED                                                                                                              [  5%]
tests/test_grammar_oracles.py::TestANBN::test_oracle_vs_brute_force PASSED                                                                                              [  7%]
tests/test_grammar_oracles.py::TestANBN::test_fully_unmasked_is_deterministic PASSED                                                                                    [ 10%]
tests/test_grammar_oracles.py::TestANBN::test_invalid_returns_none PASSED                                                                                               [ 13%]
tests/test_grammar_oracles.py::TestBaN::test_count PASSED                                                                                                               [ 15%]
tests/test_grammar_oracles.py::TestBaN::test_oracle_vs_brute_force PASSED                                                                                               [ 18%]
tests/test_grammar_oracles.py::TestBaN::test_fully_unmasked_is_deterministic PASSED                                                                                     [ 21%]
tests/test_grammar_oracles.py::TestBaN::test_position1_must_be_B PASSED                                                                                                 [ 23%]
tests/test_grammar_oracles.py::TestBaN::test_position1_masked_returns_B PASSED                                                                                          [ 26%]
tests/test_grammar_oracles.py::TestBaN::test_odd_a_count_no_valid_completion PASSED                                                                                     [ 28%]
tests/test_grammar_oracles.py::TestBaN::test_different_lengths[6] PASSED                                                                                                [ 31%]
tests/test_grammar_oracles.py::TestBaN::test_different_lengths[8] PASSED                                                                                                [ 34%]
tests/test_grammar_oracles.py::TestBBaN::test_count PASSED                                                                                                              [ 36%]
tests/test_grammar_oracles.py::TestBBaN::test_oracle_vs_brute_force PASSED                                                                                              [ 39%]
tests/test_grammar_oracles.py::TestBBaN::test_fully_unmasked_is_deterministic PASSED                                                                                    [ 42%]
tests/test_grammar_oracles.py::TestBBaN::test_no_Bs_returns_none PASSED                                                                                                 [ 44%]
tests/test_grammar_oracles.py::TestBBaN::test_A_before_B_returns_none PASSED                                                                                            [ 47%]
tests/test_grammar_oracles.py::TestBBaN::test_different_lengths[6] PASSED                                                                                               [ 50%]
tests/test_grammar_oracles.py::TestBBaN::test_different_lengths[8] PASSED                                                                                               [ 52%]
tests/test_grammar_oracles.py::TestBBaN::test_different_lengths[10] PASSED                                                                                              [ 55%]
tests/test_grammar_oracles.py::TestANBNCN::test_count PASSED                                                                                                            [ 57%]
tests/test_grammar_oracles.py::TestANBNCN::test_oracle_vs_brute_force PASSED                                                                                            [ 60%]
tests/test_grammar_oracles.py::TestANBNCN::test_fully_unmasked_is_deterministic PASSED                                                                                  [ 63%]
tests/test_grammar_oracles.py::TestANBNCN::test_wrong_order_returns_none PASSED                                                                                         [ 65%]
tests/test_grammar_oracles.py::TestANBNCN::test_different_lengths[7] PASSED                                                                                             [ 68%]
tests/test_grammar_oracles.py::TestANBNCN::test_different_lengths[10] PASSED                                                                                            [ 71%]
tests/test_grammar_oracles.py::TestNotNestedParenthesesAndBrackets::test_count PASSED                                                                                   [ 73%]
tests/test_grammar_oracles.py::TestNotNestedParenthesesAndBrackets::test_oracle_vs_brute_force PASSED                                                                   [ 76%]
tests/test_grammar_oracles.py::TestNotNestedParenthesesAndBrackets::test_fully_unmasked_is_deterministic PASSED                                                         [ 78%]
tests/test_grammar_oracles.py::TestNotNestedParenthesesAndBrackets::test_mismatched_returns_none PASSED                                                                 [ 81%]
tests/test_grammar_oracles.py::TestParenthesesAndBrackets::test_count PASSED                                                                                            [ 84%]
tests/test_grammar_oracles.py::TestParenthesesAndBrackets::test_oracle_vs_brute_force PASSED                                                                            [ 86%]
tests/test_grammar_oracles.py::TestParenthesesAndBrackets::test_fully_unmasked_is_deterministic PASSED                                                                  [ 89%]
tests/test_grammar_oracles.py::TestParenthesesAndBrackets::test_wrong_close_returns_none PASSED                                                                         [ 92%]
tests/test_grammar_oracles.py::TestParenthesesAndBrackets::test_independent_vs_nested_differ PASSED                                                                     [ 94%]
tests/test_grammar_oracles.py::TestCrossGrammar::test_baN_seqs_invalid_for_bbaN PASSED                                                                                  [ 97%]
tests/test_grammar_oracles.py::TestCrossGrammar::test_marginals_sum_to_one_at_each_position PASSED                                                                      [100%]

======================================================================= 38 passed in 1412.84s (0:23:32) =======================================================================
