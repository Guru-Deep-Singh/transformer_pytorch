import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Converts token indices to dense embedding vectors.

    Maps discrete token indices to continuous vector representations that can be used
    by the transformer model. The embeddings are scaled by the square root of the model dimension.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Args:
            d_model (int): The dimensionality of the embedding vectors. Paper Attention Is all you need mentions 512
            vocab_size (int): The size of the vocabulary (number of unique tokens).
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of token indices with shape (batch_size, sequence_length).
        Returns:
            Tensor: Embedded representations with shape (batch_size, sequence_length, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the "Attention Is All You Need" paper.
    Adds information about the position of each token in the sequence by using sine and cosine
    functions of different frequencies. The positional encodings are added to the token embeddings
    so that the model can utilize the order of the sequence.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Args:
            d_model (int): The dimensionality of the embeddings.
            seq_len (int): The maximum length of the input sequences.
            dropout (float): Dropout rate to apply after adding positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Done for numerical stability
        # Apply the sin to even pos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) # Parameter saved in the module

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            Tensor: The input tensor with positional encodings added and dropout applied, shape (batch_size, sequence_length, d_model).
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) # we do not need to learn as positional embedding is fixed
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    This module normalizes the input tensor across the features dimension for each position in each batch,
    ensuring zero mean and unit variance, and then applies learnable scaling (alpha) and shifting (bias) parameters.
    """

    def __init__(self, eps: float = 10**-6) -> None:
        """
        Args:
            eps (float, optional): A small value added to the denominator for numerical stability (default: 1e-6).
        """
        super().__init__()
        self.eps = eps # For preventing division by zero
        self.alpha =nn.Parameter(torch.ones(1)) # Multiplied for scaling
        self.bias = nn.Parameter(torch.zeros(1)) # Added for scaling 

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Layer-normalized tensor of the same shape as input, where normalization is applied over the last dimension
                    (features), followed by learnable scaling (alpha) and bias parameters.
        """
        mean = x.mean(dim = -1, keepdim=True) # Apply mean to all the tensors in a batch individually per position
        # What happens:
        #Input: (2, 100, 512) (batch, seq_len, d_model)
        #Batch 1:                    Batch 2:
        #Position 1: [512 features] → mean₁₁  Position 1: [512 features] → mean₂₁
        #Position 2: [512 features] → mean₁₂  Position 2: [512 features] → mean₂₂
        #Position 3: [512 features] → mean₁₃  Position 3: [512 features] → mean₂₃
        #...
        #Position 100: [512 features] → mean₁₁₀₀  Position 100: [512 features] → mean₂₁₀₀
        #
        #Output: (2, 100, 1) (keepdim=True foreced the shape as (2,100,1) instead of (2,100) for broadcasting)

        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



