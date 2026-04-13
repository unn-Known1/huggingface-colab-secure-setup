#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 HuggingFace Colab CLI - One-Line Model Setup
# ═══════════════════════════════════════════════════════════════════════════════
#
# Quick setup for running Hugging Face models in Google Colab
#
# Usage:
#   bash colab-hf.sh [model_id] [quantization]
#
# Examples:
#   bash colab-hf.sh gpt2
#   bash colab-hf.sh mistralai/Mistral-7B-Instruct-v0.2 4
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL_ID="${1:-gpt2}"
QUANT_BITS="${2:-4}"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗"
echo "║         🤖 HuggingFace Colab Setup - CLI Tool              ║"
echo "╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓${NC} Model: ${YELLOW}$MODEL_ID${NC}"
echo -e "${GREEN}✓${NC} Quantization: ${YELLOW}${QUANT_BITS}-bit${NC}"
echo ""

# Create notebook
NOTEBOOK_NAME="colab_setup_${MODEL_ID//\//_}.ipynb"

cat > "$NOTEBOOK_NAME" << 'NOTEBOOK_EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# HuggingFace Model Setup\n",
    "\n",
    "Model: MODEL_PLACEHOLDER\n",
    "\n",
    "![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup\n",
    "import os\n",
    "os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"\n",
    "os.environ["HF_HOME"] = "/content/.hf_cache"\n",
    "!pip install -q transformers accelerate bitsandbytes\n",
    "print("Setup complete!")
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Model\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
    "import torch\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained("MODEL_PLACEHOLDER")\n",
    "model = AutoModelForCausalLM.from_pretrained("MODEL_PLACEHOLDER")\n",
    "print("Model loaded!")
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate\n",
    "def generate(prompt, max_tokens=100):\n",
    "    inputs = tokenizer(prompt, return_tensors="pt")\n",
    "    outputs = model.generate(**inputs, max_new_tokens=max_tokens)\n",
    "    return tokenizer.decode(outputs[0])\n",
    "\n",
    "print("Ready! Use generate('prompt')")
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
NOTEBOOK_EOF

sed -i "s/MODEL_PLACEHOLDER/$MODEL_ID/g" "$NOTEBOOK_NAME"

echo -e "${GREEN}✓${NC} Created: ${YELLOW}$NOTEBOOK_NAME${NC}"
echo ""
echo "Open in Colab and run all cells!"
echo ""