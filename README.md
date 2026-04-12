# Hugging Face + Google Colab Secure Setup Guide

> **Run any Hugging Face model on Google Colab with complete data privacy, security, and free tier compatibility.** Updated for Gemma 4, Mistral, and all trending models.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/gemma4_setup.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/unn-Known1/huggingface-colab-secure-setup?style=social)](https://github.com/unn-Known1/huggingface-colab-secure-setup)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

## Table of Contents

- [✨ New: Gemma 4 Setup](#-new-gemma-4-setup)
- [🚀 Quick Start](#quick-start)
- [📚 Available Notebooks](#available-notebooks)
- [🔧 Templates & Scripts](#-templates--scripts)
- [🎯 Trending Models Guide](#trending-models-guide)
- [⚙️ Configuration Reference](#configuration-reference)
- [🔒 Security & Privacy](#security--privacy)
- [💡 Usage Examples](#usage-examples)

---

## ✨ New: Gemma 4 Setup

**Now supporting Google Gemma 4 with Colab optimization!**

[![Open Gemma 4 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/gemma4_setup.ipynb)

### Gemma 4 Variants

| Model | Parameters | Memory (4-bit) | Best For |
|-------|------------|----------------|----------|
| `google/gemma-2-2b` | 2B | ~1.2GB | Free tier, fast inference |
| `google/gemma-2-2b-it` | 2B | ~1.2GB | Instruction following |
| `google/gemma-2-9b` | 9B | ~5GB | Higher quality (16GB GPU) |
| `google/gemma-2-9b-it` | 9B | ~5GB | Best chat quality |

### Quick Gemma 4 Code

```python
# One-click Gemma 4 setup
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure environment
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = "/content/.hf_cache"

# Quantization config for free tier
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# Load Gemma 4
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## 🚀 Quick Start

### One-Click Setup (Recommended)

```python
# Clone repository and run
!git clone https://github.com/unn-Known1/huggingface-colab-secure-setup.git
%cd huggingface-colab-secure-setup

# Or import setup script directly
!pip install hf_transfer
import sys
sys.path.append('/content/huggingface-colab-secure-setup/scripts')
from hf_colab_setup import quick_start, generate, chat, secure_cleanup

# Load any model with one function
model, tokenizer, info = quick_start("gpt2", quantization_bits=4)
```

### Universal Model Loader

Load **any** Hugging Face model with optimal Colab settings:

```python
# Just change the model ID - everything else is automatic!
MODEL_ID = "google/gemma-2-2b-it"  # Change to any model
# or: "mistralai/Mistral-7B-Instruct-v0.2"
# or: "microsoft/Phi-3-mini-128k-instruct"
# or: "Qwen/Qwen2-7B-Instruct"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto"
)
```

---

## 📚 Available Notebooks

### 🔥 Trending Models (New!)

| Notebook | Model | Colab Link |
|----------|-------|------------|
| **Gemma 4 Setup** | Google Gemma 2/9B | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/gemma4_setup.ipynb) |
| **Mistral/Mixtral** | Mistral 7B, Mixtral MoE | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/templates/mistral_setup.ipynb) |
| **Universal Loader** | Any model | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/templates/universal_model_loader.ipynb) |

### 📖 Beginner Notebooks

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| [Quick Start](./notebooks/quickstart.ipynb) | Basic model loading | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/quickstart.ipynb) |
| [Text Generation](./notebooks/llm_setup.ipynb) | LLMs on Colab | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/llm_setup.ipynb) |
| [Image Classification](./notebooks/image_classification.ipynb) | Vision models | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unn-Known1/huggingface-colab-secure-setup/blob/main/notebooks/image_classification.ipynb) |

### 🔧 Templates

| Template | Purpose | Direct Load |
|----------|---------|-------------|
| [Universal Model Loader](./templates/universal_model_loader.ipynb) | Load any HF model | `?url=raw.githubusercontent.com/...` |
| [Mistral Setup](./templates/mistral_setup.ipynb) | Mistral/Mixtral models | Direct Colab link |
| [Vision Model](./templates/vision_model.ipynb) | Image processing | Direct Colab link |

---

## 🔧 Templates & Scripts

### Python Setup Script

Import reusable functions for any project:

```python
# In Colab, run:
!pip install hf_transfer

# Download and import
import subprocess
subprocess.run(['curl', '-O', 'https://raw.githubusercontent.com/unn-Known1/huggingface-colab-secure-setup/main/scripts/hf_colab_setup.py'])

# Use the functions
from hf_colab_setup import setup_environment, load_model, generate, secure_cleanup

# One-line setup
setup_environment()
model, tokenizer = load_model("gpt2")
result = generate(model, tokenizer, "Hello world")
secure_cleanup()
```

### Available Functions

| Function | Description |
|----------|-------------|
| `setup_environment()` | Configure security & performance |
| `install_dependencies()` | Install required packages |
| `check_gpu()` | Check GPU and get recommendations |
| `load_model(id, bits, device, dtype)` | Load any model with optimal settings |
| `generate(model, tokenizer, prompt, ...)` | Generate text securely |
| `chat(model, tokenizer, message, ...)` | Chat with model |
| `secure_cleanup()` | Clear sensitive data from memory |
| `quick_start(model_id, bits)` | One-click complete setup |

---

## 🎯 Trending Models Guide

### Gemma 4 (NEW!)

```python
MODEL_ID = "google/gemma-2-2b-it"  # or gemma-2-9b-it for larger
```

**Features**: 2B/9B parameters, 32K context, 32 languages, multimodal

### Mistral & Mixtral

```python
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# or Mixtral
MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
```

**Features**: Excellent performance, mixture of experts architecture

### LLaMA 2/3

```python
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # Requires access
# or
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # LLaMA 3
```

**Features**: Top-tier quality, widely supported

### Phi-3

```python
MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
```

**Features**: Microsoft optimized, long context

### Qwen 2

```python
MODEL_ID = "Qwen/Qwen2-7B-Instruct"
```

**Features**: Alibaba's model, excellent Chinese support

### Memory Requirements

| Model Size | Parameters | FP16 | 8-bit | 4-bit | Free Tier? |
|------------|------------|------|-------|-------|------------|
| Tiny | < 1B | 2GB | 1GB | 0.5GB | ✅ |
| Small | 1-3B | 6GB | 3GB | 1.5GB | ✅ |
| Medium | 3-7B | 14GB | 7GB | 3.5GB | ⚠️ |
| Large | 7-13B | 26GB | 13GB | 6.5GB | ❌ |
| XL | 13B+ | 32GB+ | 16GB+ | 8GB+ | ❌ |

---

## ⚙️ Configuration Reference

### Environment Variables

```python
# Security
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cache
os.environ["HF_HOME"] = "/content/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/.hf_cache"

# Performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

### Quantization Options

```python
# 4-bit (Recommended for free tier)
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# 8-bit (Balance of speed and memory)
BitsAndBytesConfig(load_in_8bit=True)

# No quantization (Full precision, more memory needed)
torch_dtype=torch.float16  # or torch.float32
```

### Model Loading Options

```python
# For causal LMs (LLMs)
AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)

# For masked LMs (BERT)
AutoModelForMaskedLM.from_pretrained(MODEL_ID, ...)

# For sequence classification
AutoModelForSequenceClassification.from_pretrained(MODEL_ID, ...)

# For image classification
AutoModelForImageClassification.from_pretrained(MODEL_ID, ...)
```

Full configuration guide: [docs/CONFIGURATION.md](./docs/CONFIGURATION.md)

---

## 🔒 Security & Privacy

### 100% Privacy Features

| Feature | Protection |
|---------|------------|
| **Telemetry Disabled** | No data sent to Hugging Face |
| **Local Processing** | All data stays in Colab VM |
| **Memory Cleanup** | Sensitive data cleared after use |
| **No External APIs** | No third-party service calls |
| **Secure Cache** | Models downloaded only to Colab |

### Security Checklist

```python
# ✓ Always run this at the start
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✓ Clean up after processing
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ✓ Don't log sensitive data
# print(sensitive_data)  # NEVER
```

Full security guide: [docs/SECURITY.md](./docs/SECURITY.md)

Privacy guide: [docs/PRIVACY.md](./docs/PRIVACY.md)

---

## 💡 Usage Examples

### Basic Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("The future is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Chat with Model

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][inputs.shape[1]:])
print(response)
```

### Vision Model

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1)
```

### Zero-Shot Classification

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is about AI and machine learning",
    candidate_labels=["technology", "sports", "politics"]
)
```

---

## 📦 Documentation

| Document | Description |
|----------|-------------|
| [README.md](./README.md) | This file |
| [docs/SECURITY.md](./docs/SECURITY.md) | Security best practices |
| [docs/PRIVACY.md](./docs/PRIVACY.md) | Privacy configuration |
| [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) | All settings reference |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | How to contribute |
| [LICENSE](./LICENSE) | MIT License |

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure notebooks run end-to-end
5. Update documentation

---

## ⭐ Support

If this helped you, please give it a star!

For issues or questions: [Open an issue](https://github.com/unn-Known1/huggingface-colab-secure-setup/issues)

---

**Last Updated**: April 2024 - Now supporting Gemma 4, Mistral, LLaMA 3, and all trending models!