# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Transformer PyTorch'
copyright = '2025, Guru Deep Singh'
author = 'Guru Deep Singh'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # supports Google/NumPy docstrings
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchaudio",
    "torchtext",
    "datasets",
    "tokenizers",
    "torchmetrics",
    "tensorboard",
    "altair",
    "wandb",
    "tqdm",
    "tensorrt",
    "pycuda",
    "numpy",
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
