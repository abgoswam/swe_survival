You are given a list of n untrained neural networks: [nn_1, nn_2, ..., nn_n].

Each network takes an input tensor x of size (B, E_1) and produces an output tensor of the shape (B, E_2).

Create nn.Module that takes a list of these `n` modules and performs a weighted sum of their outputs. 

The weights should be trainable parameters of your module.
