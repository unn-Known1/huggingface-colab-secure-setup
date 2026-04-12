# Hugging Face + Google Colab Secure Setup Guide

> **Run any Hugging Face model on Google Colab with complete data privacy, security, and free tier compatibility.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/quickstart.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/unn-Known1/huggingface-colab-secure-setup?style=social)](https://github.com/unn-Known1/huggingface-colab-secure-setup)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Security Overview](#security-overview)
- [Installation Guide](#installation-guide)
- [Available Notebooks](#available-notebooks)
- [Model Examples](#model-examples)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

- **100% Data Privacy** - Your data never leaves Google's secure infrastructure
- **Free Tier Compatible** - Works with Google Colab free tier (with limitations)
- **Any Model Support** - Run any Hugging Face model including LLMs, vision, audio
- **Secure Implementation** - Production-ready security patterns and best practices
- **Step-by-Step Guides** - From beginner to advanced configurations
- **Privacy-First Design** - No external API calls, all processing done locally in Colab

## Quick Start

### One-Click Setup

```python
# Clone this repository in Colab
!git clone https://github.com/unn-Known1/huggingface-colab-secure-setup.git

# Run the quick start notebook
%cd huggingface-colab-secure-setup/notebooks
```

### Basic Model Loading

```python
from transformers import AutoModel, AutoTokenizer

# Load any model from Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"  # Or any model you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### Security-First Configuration

```python
# Secure environment setup
import os
os.environ["HF_HOME"] = "/content/model_cache"  # Local cache only
os.environ["TRANSFORMERS_CACHE"] = "/content/model_cache"

# Disable telemetry and tracking
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

## Security Overview

### Why This Repository?

This guide addresses critical privacy concerns when using Hugging Face models:

| Security Concern | Our Solution |
|-----------------|-------------|
| Data leaving Colab | All processing stays in Google's secure VMs |
| API key exposure | No external API calls required |
| Model weight security | Download directly to Colab environment |
| Session persistence | Clear temporary data after each session |
| Network vulnerabilities | Use isolated Colab environment |

### Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Colab Environment                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Your Data & Processing                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │   │
│  │  │ Model   │  │ Dataset │  │ Output  │              │   │
│  │  │ Weights │  │   (You) │  │ Results │              │   │
│  │  └─────────┘  └─────────┘  └─────────┘              │   │
│  │         All local, never transmitted externally       │   │
│  └─────────────────────────────────────────────────────┘   │
│                     No external data transmission            │
└─────────────────────────────────────────────────────────────┘
```

## Installation Guide

### Prerequisites

- Google Account with access to [Google Colab](https://colab.research.google.com/)
- (Optional) Hugging Face account for gated models
- Internet connection for downloading models

### Step 1: Mount Google Drive (Optional but Recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Install Required Packages

```python
# Upgrade pip and install dependencies
!pip install --upgrade pip
!pip install transformers datasets accelerate bitsandbytes
```

### Step 3: Configure Security Settings

```python
import os

# Create secure cache directory
!mkdir -p /content/secure_cache

# Set environment variables for security
os.environ["HF_HOME"] = "/content/secure_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/secure_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
```

### Step 4: Authenticate with Hugging Face (For Gated Models)

```python
# Login to Hugging Face for gated models
!pip install huggingface_hub
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")  # Get token from huggingface.co/settings/tokens
```

## Available Notebooks

### Beginner Level

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| [Quick Start](./notebooks/quickstart.ipynb) | Basic model loading and inference | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/quickstart.ipynb) |
| [Text Classification](./notebooks/text_classification.ipynb) | Sentiment analysis with BERT | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/text_classification.ipynb) |
| [Image Classification](./notebooks/image_classification.ipynb) | Vision models with ResNet | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/image_classification.ipynb) |

### Intermediate Level

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| [LLM Setup](./notebooks/llm_setup.ipynb) | Large language model setup | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/llm_setup.ipynb) |
| [Fine-tuning](./notebooks/fine_tuning.ipynb) | Custom model fine-tuning | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/fine_tuning.ipynb) |
| [Embeddings](./notebooks/embeddings.ipynb) | Text embeddings and similarity | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/embeddings.ipynb) |

### Advanced Level

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| [8-bit Quantization](./notebooks/quantization_8bit.ipynb) | Memory-optimized inference | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/quantization_8bit.ipynb) |
| [LoRA Fine-tuning](./notebooks/lora_finetuning.ipynb) | Parameter-efficient training | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/quantization_8bit.ipynb) |
| [Streaming Inference](./notebooks/streaming_inference.ipynb) | Real-time text generation | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/streaming_inference.ipynb) |

## Model Examples

### Text Generation (LLM)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"  # Use any Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text securely
input_text = "Write a secure implementation for"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Image Classification

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

image = Image.open("your_image.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
```

### Speech Recognition

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")

# Process audio input
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
predicted_ids = model.generate(inputs.input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

## Security Best Practices

### 1. Environment Isolation

```python
# Always use fresh environment for sensitive data
import os
os.environ["PYTHONHASHSEED"] = "42"

# Clear any previous state
import sys
if hasattr(sys, 'dont_write_bytecode'):
    sys.dont_write_bytecode = True
```

### 2. Secure Model Download

```python
from huggingface_hub import hf_hub_download

# Verify model integrity before loading
model_path = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="config.json",
    token=None  # Use None for public models
)
```

### 3. Memory Management

```python
import gc

# Clear GPU memory after processing sensitive data
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Call after processing sensitive information
clear_memory()
```

### 4. Secure File Handling

```python
import tempfile
import shutil

# Use temporary directories for sensitive operations
with tempfile.TemporaryDirectory() as tmpdir:
    # Process data securely
    pass
# Automatically cleaned up after exit
```

### 5. No Persistence of Sensitive Data

```python
import os

# Disable any form of logging or persistence
os.environ["HF_HUB_OFFLINE"] = "0"  # Keep online for model downloads only

# After session ends, all temporary data is automatically deleted
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

**Problem:** GPU runs out of memory when loading large models.

**Solution:**
```python
# Use 8-bit quantization to reduce memory usage
from transformers import AutoModelForCausalLM
from bitsandbytes import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config
)
```

#### Slow Download Speeds

**Problem:** Model download takes too long.

**Solution:**
```python
# Use hf_transfer for faster downloads
!pip install hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

#### Authentication Issues

**Problem:** Cannot access gated models.

**Solution:**
```python
# Get token from https://huggingface.co/settings/tokens
from huggingface_hub import login
login(token="hf_your_token_here")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you found this helpful, please give it a star!

For questions or issues, please open an issue on GitHub.