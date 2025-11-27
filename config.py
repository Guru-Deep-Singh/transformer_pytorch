from pathlib import Path

def get_config():
    """
    Returns a dictionary containing the default configuration parameters for the transformer model.

    The configuration includes hyperparameters for training, model architecture, file paths,
    and language settings for the translation task.

    Returns:
        dict: Configuration dictionary with the following keys:
            - batch_size (int): Number of samples per training batch (default: 8).
            - num_epochs (int): Number of training epochs (default: 20).
            - lr (float): Learning rate for the optimizer (default: 1e-4).
            - seq_len (int): Maximum sequence length for source and target sequences (default: 350).
            - d_model (int): Dimension of the model embeddings and hidden states (default: 512).
            - lang_src (str): Source language code (default: "en").
            - lang_tgt (str): Target language code (default: "it").
            - model_folder (str): Directory name where model weights will be saved (default: "weights").
            - model_basename (str): Base filename prefix for saved model weights (default: \"tmodel\_\).
            - preload (str or None): Path to a pre-trained model to load, or None if starting from scratch.
            - tokenizer_file (str): Filename pattern for the tokenizer JSON file.
            - experiment_name (str): Directory path for experiment tracking/logging (default: "runs/tmodel").
    """
    return {
        "batch_size": 4,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 512,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "de",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "29",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    """
    Constructs the file path for saving or loading model weights for a specific epoch.

    Combines the model folder, model basename, and epoch number to create a complete
    file path for the model checkpoint file.

    Args:
        config (dict): Configuration dictionary containing 'model_folder' and 'model_basename' keys.
        epoch (str): Epoch number or identifier to include in the filename.

    Returns:
        str: Complete file path to the model weights file, e.g., "weights/tmodel_10.pt".

    Example:
        >>> config = {"model_folder": "weights", "model_basename": "tmodel_"}
        >>> get_weights_file_path(config, "10")
        'weights/tmodel_10.pt'
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

def latest_weights_file_path(config):
    """
    Finds the most recent model weights file in the specified model folder.

    Looks for all model checkpoint files in the model directory matching the model basename 
    pattern, sorts them, and returns the path to the latest (most recent) checkpoint file. 
    Returns None if no checkpoint files are found.

    Args:
        config (dict): Configuration dictionary containing 'datasource', 'model_folder', and 'model_basename' keys.

    Returns:
        str or None: Path to the latest model weights file if available, otherwise None.

    Example:
        >>> config = {"datasource": "opus_books", "model_folder": "weights", "model_basename": "tmodel_"}
        >>> latest_weights_file_path(config)
        'opus_books_weights/tmodel_19.pt'
    """
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
