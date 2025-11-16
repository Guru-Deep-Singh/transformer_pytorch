from typing import Any
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path

import warnings
import torchmetrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Performs greedy decoding to generate target sequence from source sequence.
    
    Args:
        model: The transformer model to use for encoding and decoding.
        source: Source input tensor of shape (batch_size, seq_len).
        source_mask: Mask tensor for source sequence.
        tokenizer_src: Tokenizer for source language.
        tokenizer_tgt: Tokenizer for target language.
        max_len: Maximum length of the generated sequence.
        device: Device to run the computation on (e.g., 'cuda' or 'cpu').
    
    Returns:
        torch.Tensor: Decoded sequence tensor of shape (seq_len,).
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder Input with [SOS]
    # We have two dim (batch, length of the decoder input token)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token (we only take the ouput of last token)
        prob = model.project(out[:, -1])
        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Runs validation on the model and computes evaluation metrics.
    
    Args:
        model: The transformer model to validate.
        validation_ds: DataLoader containing validation dataset.
        tokenizer_src: Tokenizer for source language.
        tokenizer_tgt: Tokenizer for target language.
        max_len: Maximum length for generated sequences.
        device: Device to run the computation on (e.g., 'cuda' or 'cpu').
        print_msg: Function to print messages (typically from tqdm).
        global_step: Current global training step for logging.
        writer: TensorBoard SummaryWriter for logging metrics.
        num_examples: Number of validation examples to process (default: 2).
    
    Note:
        Batch size must be 1 for validation. The function computes CER, WER, and BLEU scores.
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())


            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print on console (function given by tqdm to avoid any conflicts)
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

        if writer:
            # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()


def get_all_sentences(ds, lang):
    """
    Generator function that yields all sentences in a specific language from a dataset.
    
    Args:
        ds: Dataset containing translation pairs.
        lang: Language code to extract sentences from (e.g., 'en', 'fr').
    
    Yields:
        str: Sentences in the specified language from the dataset.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Gets an existing tokenizer from file or builds a new one if it doesn't exist.
    
    Args:
        config: Configuration dictionary containing tokenizer file path.
        ds: Dataset to train the tokenizer on if building a new one.
        lang: Language code for the tokenizer.
    
    Returns:
        Tokenizer: A WordLevel tokenizer for the specified language.
    
    Note:
        If the tokenizer file doesn't exist, it will be trained on the dataset
        and saved. The tokenizer uses special tokens: [UNK], [PAD], [SOS], [EOS].
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer= trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    Loads and prepares the dataset for training and validation.
    
    Args:
        config: Configuration dictionary containing dataset parameters such as
                lang_src, lang_tgt, seq_len, and batch_size.
    
    Returns:
        tuple: A tuple containing:
            - train_dataloader: DataLoader for training data
            - val_dataloader: DataLoader for validation data
            - tokenizer_src: Tokenizer for source language
            - tokenizer_tgt: Tokenizer for target language
    
    Note:
        The dataset is split into 90% training and 10% validation.
        Also prints the maximum length of source and target sentences.
    """
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentences: {max_len_src}')
    print(f'Max length of target sentences: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Builds and returns a transformer model with the specified configuration.
    
    Args:
        config: Configuration dictionary containing model parameters such as
                seq_len and d_model.
        vocab_src_len: Vocabulary size of the source language.
        vocab_tgt_len: Vocabulary size of the target language.
    
    Returns:
        nn.Module: A transformer model built according to the configuration.
    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    """
    Main training function that trains the transformer model.
    
    Args:
        config: Configuration dictionary containing all training parameters such as:
                - model_folder: Directory to save model checkpoints
                - experiment_name: Name for TensorBoard logging
                - lr: Learning rate for optimizer
                - preload: Optional model checkpoint to resume from
                - num_epochs: Number of training epochs
                - seq_len: Sequence length
                - batch_size: Batch size for training
    
    Note:
        The function handles:
        - Device selection (CUDA if available, else CPU)
        - Model checkpointing (saves after each epoch)
        - Resuming from checkpoint if preload is specified
        - TensorBoard logging for loss and validation metrics
        - Validation runs during training
    """
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard for loss visu
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr= config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # Forcing the loss function to ignore the loss from Padding tokens
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device) #(batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(batch, 1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensor through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(batch, seq_len, d_model)
            proj_output = model.project(decoder_output) #(batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) #(batch, seq_len)
            
            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss 
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step +=1

        # Run validation after every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model for every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
            },model_filename
        )

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)










    
