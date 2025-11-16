import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):
    """
    A PyTorch Dataset for bilingual translation tasks.

    This dataset processes parallel source and target language text pairs,
    tokenizes them, and prepares them for transformer model training with
    proper padding, masking, and causal masking for the decoder.

    Args:
        ds: The dataset containing translation pairs (typically from HuggingFace datasets).
        tokenizer_src: Tokenizer for the source language.
        tokenizer_tgt: Tokenizer for the target language.
        src_lang (str): Source language code (e.g., "en").
        tgt_lang (str): Target language code (e.g., "it").
        seq_len (int): Maximum sequence length for padding and truncation.

    Attributes:
        ds: The underlying dataset.
        tokenizer_src: Source language tokenizer.
        tokenizer_tgt: Target language tokenizer.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        seq_len (int): Maximum sequence length.
        sos_token (Tensor): Start-of-sequence token tensor.
        eos_token (Tensor): End-of-sequence token tensor.
        pad_token (Tensor): Padding token tensor.
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        """
        Initialize the BilingualDataset.

        Args:
            ds: The dataset containing translation pairs (typically from HuggingFace datasets).
            tokenizer_src: Tokenizer for the source language.
            tokenizer_tgt: Tokenizer for the target language.
            src_lang (str): Source language code (e.g., "en").
            tgt_lang (str): Target language code (e.g., "it").
            seq_len (int): Maximum sequence length for padding and truncation.
        """
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of translation pairs in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        """
        Retrieves a single sample from the dataset and prepares it for training.

        Processes a translation pair by tokenizing both source and target texts,
        adding special tokens ([SOS], [EOS], [PAD]), padding to seq_len, and
        creating appropriate masks for encoder and decoder.

        Args:
            index (Any): Index of the sample to retrieve from the dataset.

        Returns:
            dict: A dictionary containing:
                - encoder_input (Tensor): Tokenized and padded source sequence, shape (seq_len).
                - decoder_input (Tensor): Tokenized and padded target sequence with [SOS], shape (seq_len).
                - encoder_mask (Tensor): Mask for encoder attention, shape (1, 1, seq_len).
                - decoder_mask (Tensor): Combined padding and causal mask for decoder, shape (1, seq_len, seq_len).
                - label (Tensor): Target sequence for loss computation (decoder input shifted by one), shape (seq_len).
                - src_text (str): Original source text (for visualization/debugging).
                - tgt_text (str): Original target text (for visualization/debugging).

        Raises:
            ValueError: If the source or target sentence exceeds the maximum sequence length.
        """
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # The minus 2 is for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # The minus 1 is for [SOS], we do not give [EOS]

        # Check to see that no sentence in the ds should be longer than seq_len 
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Only adding [SOS]
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Label offset by one token i.e. last token of the input of decoder  -> [EOS] token as result 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Simple sanity check
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len) which in attention corresponds to (batch, seq_len, seq_len) 
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text, # Only for visu
            "tgt_text": tgt_text # Only for visu
        }

def causal_mask(size):
    """
    Creates a causal (lower triangular) attention mask for the decoder.

    Generates a mask that prevents tokens from attending to future positions,
    ensuring that during training, each token can only attend to previous tokens
    and itself. This is essential for autoregressive generation in the decoder.

    Args:
        size (int): The sequence length (size of the square mask matrix).

    Returns:
        Tensor: A boolean mask of shape (1, size, size) where:
            - True (1) indicates positions that can be attended to (lower triangle + diagonal).
            - False (0) indicates positions that should be masked (upper triangle).

    Note:
        The mask is inverted (mask == 0) so that True values allow attention
        and False values prevent attention to future positions.
    """
    # This will give a matrix with upper diagonal as 1 with diagonal as 1
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)

    # We return the ,atrix with lower diagonal as 1 i.e. the causal attention 
    # Every token can have attention to the token before it not after it
    return mask == 0
