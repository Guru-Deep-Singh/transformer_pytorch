#!/bin/bash
set -e

echo "=== TorchText 0.18 Installer for Jetson (Orin) ==="

### CONFIGURATION
VENV_NAME="myenv_jetson"
TORCHTEXT_VERSION="v0.18.0"
REPO_URL="https://github.com/pytorch/text.git"

echo "[1/8] Installing system dependencies..."
sudo apt update
sudo apt install -y \
    ninja-build \
    build-essential \
    cmake \
    libopenblas-dev \
    libsentencepiece-dev \
    python3-dev \
    gcc g++ git

echo "[2/8] Creating clean Python virtual environment..."
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

echo "[3/8] Upgrading pip & setuptools..."
pip install --upgrade pip setuptools wheel

echo "[4/8] Cloning torchtext repository..."
rm -rf text
git clone --recursive $REPO_URL
cd text
git checkout $TORCHTEXT_VERSION
git submodule update --init --recursive

echo "[5/8] Cleaning previous build artifacts (if any)..."
git clean -xfd

echo "[6/8] Exporting ABI flag for PyTorch compatibility..."
# Jetson PyTorch is built with CXX11 ABI = 1
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"

echo "[7/8] Building & installing torchtext $TORCHTEXT_VERSION..."
python setup.py install

echo "[8/8] Validating installation..."
python - << 'EOF'
import torch
import torchtext
try:
    import torchtext._torchtext
    print("✔ Torch:", torch.__version__)
    print("✔ TorchText:", torchtext.__version__)
    print("✔ C++ extension loaded successfully. Install OK!")
except Exception as e:
    print("❌ TorchText extension load failed:", e)
EOF

echo "=== TorchText 0.18 Installation Completed ==="
