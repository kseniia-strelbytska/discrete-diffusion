Models trained with the following parameters:
anbn grammar
learneable embedding positional encoding
weight of 10 is applied to EOS token
l=256
model = Model(max_len=l+2, vocab_size=6, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1) # +2 SOS/EOS token
Accuracy measure: all prefixes of length 1-127 "000...0" and 1-128 "000...01"
Accuracy plataus at ~251/254 (0.9881889763779528) and the only three cases that disobey 
rule #1 are "000...01" with 74, 81, 105 zeros. 
They generate exactly 1 less '1' than needed, e.g. for input_l=106 ("000...01"), the model 
generates a string with 106 zeros followed by 104 ones
