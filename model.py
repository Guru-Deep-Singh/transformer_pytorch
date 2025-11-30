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
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # we do not need to learn as positional embedding is fixed
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs with per-feature parameters.

    This module normalizes the input tensor across the features dimension for each position in each batch,
    ensuring zero mean and unit variance, and then applies learnable scaling (alpha) and shifting (bias) parameters
    sized to the `features` dimension passed at construction time.
    """

    def __init__(self, features:int, eps: float = 10**-6) -> None:
        """
        Args:
            features (int): Number of features per token; controls the size of the learnable scale/bias vectors.
            eps (float, optional): A small value added to the denominator for numerical stability (default: 1e-6).
        """
        super().__init__()
        self.eps = eps # For preventing division by zero
        self.alpha =nn.Parameter(torch.ones(features)) # Multiplied for scaling
        self.bias = nn.Parameter(torch.zeros(features)) # Added for scaling 

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

        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Implements the Position-wise Feed Forward block used in Transformer architectures.

    This block consists of two linear transformations with a ReLU activation and a dropout layer in between.
    It is applied independently to each position (token) in the sequence.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model (int): The dimension of the embedding/hidden state.
            d_ff (int): The dimension of the feed-forward network's inner layer.
            dropout (float): Dropout probability applied after the activation.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        """
        Applies position-wise feed-forward transformation to the input.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        
        The transformation flow:
            (batch, seq_len, d_model) --(linear 1)--> (batch, seq_len, d_ff) 
            --(linear 2)--> (batch, seq_len, d_model)
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

    
class MultiHeadAttentionBlock(nn.Module):
    """
    Implements Multi-Head Attention mechanism as described in "Attention Is All You Need".

    This module computes scaled dot-product attention across multiple heads in parallel,
    enabling the model to jointly attend to information from different representation subspaces at different positions.

    Args:
        d_model (int): The dimensionality of the input and output features.
        h (int): Number of attention heads.
        dropout (float): Dropout probability applied to the attention scores.

    Attributes:
        w_q (nn.Linear): Linear layer to project input queries to the attention space.
        w_k (nn.Linear): Linear layer to project input keys to the attention space.
        w_v (nn.Linear): Linear layer to project input values to the attention space.
        w_o (nn.Linear): Linear layer to project concatenated output of all heads back to d_model.
        dropout (nn.Dropout): Dropout layer for regularizing attention weights.
        d_k (int): Dimensionality of each attention head.
        h (int): Number of attention heads.
        d_model (int): Input/output feature dimension.
    """

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute the scaled dot-product attention.

        Args:
            query (Tensor): Query tensor of shape (batch_size, h, seq_len, d_k).
            key (Tensor): Key tensor of shape (batch_size, h, seq_len, d_k).
            value (Tensor): Value tensor of shape (batch_size, h, seq_len, d_k).
            mask (Tensor or None): Mask tensor that prevents attention to certain positions, shape broadcastable to (batch_size, 1, seq_len, seq_len).
            dropout (nn.Dropout): Dropout layer applied to the attention scores.

        Returns:
            Tuple[Tensor, Tensor]:
                - Output: Attention-weighted values, shape (batch_size, h, seq_len, d_k).
                - Attention Scores: The attention weights after softmax, same shape as the attention matrix.
        """
        d_k = query.shape[-1]

        #(batch, h, seq_len, d_k) X (batch, h, d_k, seq_len) = (batch, h, seq_len, seq_len)
        # Matrix (seq_len, seq_len) represents the attention between the tokens
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        
        # We need to apply mask before softmax so that wherever the mask value is 0 we put -1e9 in attention
        # thus the softmax will result in 0 i.e. no attention between the tokens
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e4) # -1e4 needed for the fp16 conversion with TensorRT

        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        #(batch, h, seq_len, seq_len) X (batch, h, seq_len, d_k) = (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores # we are returning the attention_scores just for visu

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        Initializes the MultiHeadAttentionBlock module.

        Args:
            d_model (int): Total dimension of the model (input/output features).
            h (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h # Number of head
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of features per head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq (Query)
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk (Key)
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv (Value)

        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo (Output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        Applies multi-head attention mechanism to the input.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor or None): Mask tensor to prevent attention to certain positions, 
                                   shape broadcastable to (batch_size, 1, seq_len, seq_len).

        Returns:
            Tensor: The result of multi-head attention, shape (batch_size, seq_len, d_model).
        """
        query = self.w_q(q) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # The transpose is done to get mini batches i.e. heads i.e. group of features per token or sequence
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #(batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch,  seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization and dropout.

    The residual connection is applied as: 
    y = x + Dropout(Sublayer(LayerNorm(x)))
    This helps stabilize training of deep transformer models by allowing gradients 
    to flow directly through the network. The normalization also helps to 
    stabilize the activations within the network.

    Args:
        features (int): Feature dimension passed to the internal LayerNormalization.
        dropout (float): Dropout probability to use after the sublayer.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied after the sublayer.
        norm (LayerNormalization): Layer normalization applied before the sublayer.
    """

    def __init__(self, features:int, dropout: float) -> None:
        """
        Initialize the ResidualConnection module.

        Args:
            features (int): Size of the token features used by the associated layer norm.
            dropout (float): The dropout probability to apply after the sublayer transformation.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features, eps=1e-6)

    def forward(self, x, sublayer):
        """
        Applies residual connection, layer normalization, the given sublayer, and dropout.

        Args:
            x (Tensor): The input tensor.
            sublayer (Callable[[Tensor], Tensor]): The sublayer (e.g. attention or feed-forward network) 
                                                    to apply to the normalized input.

        Returns:
            Tensor: The output tensor after applying residual connection, normalization, sublayer, and dropout.
        """
        # Note: we are first applying norm then feed it to sublayer
        # Some implementation do norm on the output of sublayer i.e. self.norm(sublayer(x))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    A single encoder block in the Transformer architecture.

    Implements one layer of the encoder stack, consisting of:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Position-wise feed-forward network with residual connection and layer normalization

    The encoder block processes input sequences by allowing each position to attend
    to all positions in the previous layer, enabling the model to capture dependencies
    and relationships within the input sequence.

    Args:
        features (int): Feature dimension used by the contained residual connections.
        self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention module.
        feed_forward_block (FeedForwardBlock): The position-wise feed-forward network module.
        dropout (float): Dropout probability applied in residual connections.
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # List of 2 ResidualConnections

    def forward(self, x, src_mask):
        """
        Applies the encoder block transformations to the input.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (Tensor or None): Source mask tensor to prevent attention to certain positions,
                                      shape broadcastable to (batch_size, 1, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model) after applying
                   self-attention and feed-forward transformations with residual connections.
        """
        # Here the lambda creates a wrapper function which allows self_attention_block to be called
        # as sublayer(x) as needed inside the ResidualConnection.forward function.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    """
    The complete encoder stack of the Transformer architecture.

    Comprises multiple encoder blocks stacked together, where each block processes
    the input sequence through self-attention and feed-forward networks with residual
    connections. The encoder transforms input sequences into contextualized representations
    that capture dependencies within the source sequence.

    Args:
        features (int): Feature dimension consumed by the final layer normalization.
        layers (nn.ModuleList): A list of EncoderBlock modules. Typically contains N blocks
                                (N=6 in the original paper).

    Attributes:
        layers (nn.ModuleList): Stack of encoder blocks.
        norm (LayerNormalization): Final layer normalization applied after all blocks.
    """

    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        """
        Initialize the Encoder module.

        Args:
            features (int): Model dimension used to size the terminal LayerNormalization.
            layers (nn.ModuleList): A ModuleList containing EncoderBlock instances.
                                   Typically N blocks where N=6 in the original paper.
        """
        super().__init__()
        self.layers = layers # Here the ModuleList will have EncoderBlocks x N where N = 6 in paper
        self.norm = nn.LayerNorm(features, eps=1e-6)

    def forward(self, x, mask):
        """
        Processes the input through all encoder blocks.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor or None): Source mask tensor to prevent attention to certain positions,
                                   shape broadcastable to (batch_size, 1, seq_len, seq_len).

        Returns:
            Tensor: Encoded output tensor of shape (batch_size, seq_len, d_model) after
                   passing through all encoder blocks and final layer normalization.
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    A single decoder block in the Transformer architecture.

    Implements one layer of the decoder stack, consisting of three sub-layers:
    1. Multi-head self-attention with residual connection and layer normalization (masked)
    2. Multi-head cross-attention with residual connection and layer normalization
    3. Position-wise feed-forward network with residual connection and layer normalization

    The decoder block processes target sequences while attending to both the target
    sequence (masked self-attention) and the encoder output (cross-attention), enabling
    the model to generate output sequences based on the encoded source.

    Args:
        features (int): Feature dimension provided to the residual connections.
        self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention module
                                                        (masked to prevent future information leakage).
        cross_attention_block (MultiHeadAttentionBlock): The multi-head cross-attention module
                                                         that attends to encoder output.
        feed_forward_block (FeedForwardBlock): The position-wise feed-forward network module.
        dropout (float): Dropout probability applied in residual connections.
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Initialize the DecoderBlock module.

        Args:
            features (int): Model dimension used for all internal layer norms.
            self_attention_block (MultiHeadAttentionBlock): Masked self-attention module.
            cross_attention_block (MultiHeadAttentionBlock): Cross-attention module.
            feed_forward_block (FeedForwardBlock): Feed-forward network module.
            dropout (float): Dropout probability for residual connections.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Applies the decoder block transformations to the input.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model).
            encoder_output (Tensor): Output from the encoder stack, shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor or None): Source mask tensor to prevent attention to certain source positions,
                                       shape broadcastable to (batch_size, 1, 1, src_seq_len).
            tgt_mask (Tensor or None): Target mask tensor to prevent attention to future positions (causal mask),
                                       shape broadcastable to (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model) after applying
                   self-attention, cross-attention, and feed-forward transformations with residual connections.
        """
        x  = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """
    The complete decoder stack of the Transformer architecture.

    Comprises multiple decoder blocks stacked together, where each block processes
    the target sequence through masked self-attention, cross-attention (attending to
    encoder output), and feed-forward networks with residual connections. The decoder
    generates output sequences by attending to both the target sequence (with masking)
    and the encoded source sequence.

    Args:
        features (int): Feature dimension provided to the final layer normalization.
        layers (nn.ModuleList): A list of DecoderBlock modules. Typically contains N blocks
                                (N=6 in the original paper).

    Attributes:
        layers (nn.ModuleList): Stack of decoder blocks.
        norm (LayerNormalization): Final layer normalization applied after all blocks.
    """

    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        """
        Initialize the Decoder module.

        Args:
            features (int): Model dimension used to size the terminal LayerNormalization.
            layers (nn.ModuleList): A ModuleList containing DecoderBlock instances.
                                   Typically N blocks where N=6 in the original paper.
        """
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features, eps=1e-6)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Processes the input through all decoder blocks.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model).
            encoder_output (Tensor): Output from the encoder stack, shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor or None): Source mask tensor to prevent attention to certain source positions,
                                       shape broadcastable to (batch_size, 1, 1, src_seq_len).
            tgt_mask (Tensor or None): Target mask tensor to prevent attention to future positions (causal mask),
                                       shape broadcastable to (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            Tensor: Decoded output tensor of shape (batch_size, tgt_seq_len, d_model) after
                   passing through all decoder blocks and final layer normalization.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Final projection layer that maps decoder output to vocabulary logits.

    Projects the decoder's output from the model dimension (d_model) to the vocabulary size,
    and returns raw logits over the vocabulary for each position. Softmax can be applied externally
    depending on the training or inference workflow.

    Args:
        d_model (int): The dimensionality of the model's hidden states.
        vocab_size (int): The size of the vocabulary (number of tokens).

    Attributes:
        proj (nn.Linear): Linear layer that projects from d_model to vocab_size.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initialize the ProjectionLayer module.

        Args:
            d_model (int): The dimensionality of the input features (hidden states).
            vocab_size (int): The size of the output vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # Mapping the feature dim back to number from vocab list

    def forward(self, x):
        """
        Projects the input to vocabulary logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Logits over the vocabulary for each position,
                   shape (batch_size, seq_len, vocab_size).
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x) 


class Transformer(nn.Module):
    """
    Complete Transformer model as described in "Attention Is All You Need".

    Combines the encoder, decoder, input embeddings, positional encodings, and projection layer
    to form the complete sequence-to-sequence transformer architecture. The model processes
    source sequences through the encoder and generates target sequences using the decoder,
    with both components using multi-head attention mechanisms.

    Args:
        encoder (Encoder): The encoder stack that processes source sequences.
        decoder (Decoder): The decoder stack that generates target sequences.
        src_embed (InputEmbeddings): Source sequence embedding layer.
        tgt_embed (InputEmbeddings): Target sequence embedding layer.
        src_pos (PositionalEncoding): Source positional encoding layer.
        tgt_pos (PositionalEncoding): Target positional encoding layer.
        projection_layer (ProjectionLayer): Final projection layer mapping to vocabulary.

    Attributes:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbeddings): Source embedding layer.
        tgt_embed (InputEmbeddings): Target embedding layer.
        src_pos (PositionalEncoding): Source positional encoding layer.
        tgt_pos (PositionalEncoding): Target positional encoding layer.
        projection_layer (ProjectionLayer): Vocabulary projection layer.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """
        Initialize the Transformer model.

        Args:
            encoder (Encoder): The encoder stack module.
            decoder (Decoder): The decoder stack module.
            src_embed (InputEmbeddings): Source sequence embedding layer.
            tgt_embed (InputEmbeddings): Target sequence embedding layer.
            src_pos (PositionalEncoding): Source positional encoding layer.
            tgt_pos (PositionalEncoding): Target positional encoding layer.
            projection_layer (ProjectionLayer): Final projection layer to vocabulary.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.

        Args:
            src (Tensor): Source token indices, shape (batch_size, src_seq_len).
            src_mask (Tensor or None): Source mask tensor to prevent attention to certain positions,
                                     shape broadcastable to (batch_size, 1, src_seq_len, src_seq_len).

        Returns:
            Tensor: Encoded source sequence, shape (batch_size, src_seq_len, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence using encoder output.

        Args:
            encoder_output (Tensor): Output from the encoder, shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor or None): Source mask tensor, shape broadcastable to (batch_size, 1, 1, src_seq_len).
            tgt (Tensor): Target token indices, shape (batch_size, tgt_seq_len).
            tgt_mask (Tensor or None): Target mask tensor (causal mask), 
                                      shape broadcastable to (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            Tensor: Decoded target sequence, shape (batch_size, tgt_seq_len, d_model).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Projects decoder output to vocabulary log probabilities.

        Args:
            x (Tensor): Decoder output tensor, shape (batch_size, tgt_seq_len, d_model).

        Returns:
            Tensor: Log probabilities over vocabulary, shape (batch_size, tgt_seq_len, vocab_size).
        """
        return self.projection_layer(x)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Full forward pass: encode src, decode tgt, then project to logits.

        Args:
            src: (batch, src_seq_len) token ids
            tgt: (batch, tgt_seq_len) token ids
            src_mask: (batch, 1, src_seq_len, src_seq_len)
            tgt_mask: (batch, 1, tgt_seq_len, tgt_seq_len)

        Returns:
            logits: (batch, tgt_seq_len, tgt_vocab_size)
        """
        enc = self.encode(src, src_mask)
        dec = self.decode(enc, src_mask, tgt, tgt_mask)
        logits = self.project(dec)
        return logits


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Builds and initializes a complete Transformer model.

    Constructs a transformer model with the specified architecture parameters,
    creates all necessary components (encoder, decoder, embeddings, positional encodings,
    and projection layer), and initializes the parameters using Xavier uniform initialization.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum length of source sequences.
        tgt_seq_len (int): Maximum length of target sequences.
        d_model (int, optional): Dimension of the model (default: 512).
        N (int, optional): Number of encoder and decoder blocks (default: 6).
        h (int, optional): Number of attention heads (default: 8).
        dropout (float, optional): Dropout probability (default: 0.1).
        d_ff (int, optional): Dimension of the feed-forward network's inner layer (default: 2048).

    Returns:
        Transformer: A fully initialized Transformer model ready for training or inference.

    Note:
        All parameters with dimensions greater than 1 are initialized using Xavier uniform
        initialization as recommended in the original paper.
    """
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # If the src and tgt seq len is same then we can use only one pos layer for both

    # Create Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model,encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model,decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the param with Xavier Uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


