#!/bin/bash
# TEXT2GRAD Environment Setup Script
# Usage: bash env.sh

echo "Setting up TEXT2GRAD environment..."

# Create and activate conda environment
if command -v conda &> /dev/null; then
    echo "Setting up conda environment 'text2grad'..."
    conda create -n text2grad python=3.10 -y || { echo "Failed to create conda environment"; exit 1; }
    
    # Handle conda activation in script
    eval "$(conda shell.bash hook)"
    conda activate text2grad || { echo "Failed to activate conda environment"; exit 1; }
else
    echo "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.4..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install required packages
echo "Installing required packages..."
pip install trl==0.11.3
pip install scikit-learn pandas
pip install peft --no-dependencies
pip install transformers
pip install numpy==1.26.4
pip install accelerate
pip install -U bitsandbytes
pip install rouge rouge_score
pip install bert_score deepspeed azure-cli
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"

# Configure git credentials (optional)
echo "Configuring git credentials..."
git config --global credential.helper store

# Install OpenAI and Azure packages
echo "Installing OpenAI and Azure packages..."
pip install openai azure-identity-broker --upgrade

# Setup Weights & Biases (optional - will prompt for login)
echo "Setting up Weights & Biases..."
pip install wandb
echo "Please login to Weights & Biases when prompted (or press Ctrl+C to skip):"
wandb login

# Setup Hugging Face (optional)
echo "Setting up Hugging Face access..."
pip install huggingface_hub
echo "To login to Hugging Face, run the following command:"
echo "python -c \"from huggingface_hub import login; login('YOUR_HF_TOKEN')\""

echo "Environment setup complete! Activate with: conda activate ppo"
