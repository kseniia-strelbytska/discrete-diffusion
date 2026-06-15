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
