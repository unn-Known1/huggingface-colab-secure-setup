# HuggingFace Colab Setup - Comprehensive Configuration Guide

> Complete reference for all available settings, optimizations, and configurations for Hugging Face models on Google Colab.

## Table of Contents

- [Quick Setup Scripts](#quick-setup-scripts)
- [Environment Variables](#environment-variables)
- [Quantization Options](#quantization-options)
- [Model Loading Configs](#model-loading-configs)
- [GPU Optimization](#gpu-optimization)
- [Security Settings](#security-settings)
- [Troubleshooting Configs](#troubleshooting-configs)

---

## Quick Setup Scripts

### Basic Setup (All Models)

```python
# ============================================
# BASIC SETUP - Works with 99% of models
# ============================================

import os
import sys

# Environment configuration
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = "/content/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/.hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable bytecode caching
sys.dont_write_bytecode = True

# Install dependencies
!pip install -q transformers accelerate

print("✓ Basic setup complete!")
```

### Full Setup (LLMs with Quantization)

```python
# ============================================
# FULL SETUP - For large language models
# ============================================

import os
import sys

# Security
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "42"

# Cache
os.environ["HF_HOME"] = "/content/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/.hf_cache"

# Performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Install
!pip install -q transformers accelerate bitsandbytes hf_transfer

print("✓ Full setup complete!")
```

---

## Environment Variables

### Core Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `/content/.hf_cache` | Model cache location |
| `TRANSFORMERS_CACHE` | `/content/.hf_cache` | Alternative cache path |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | Disable usage tracking |
| `TOKENIZERS_PARALLELISM` | `false` | Prevent race conditions |

### Performance Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Memory fragmentation prevention |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | 3-5x faster downloads |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `600` | Extended download timeout |
| `HF_HUB_HTTP_RETRY_DELAY` | `3` | Retry delay for failed downloads |

### Security Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONHASHSEED` | `42` | Hash randomization for security |
| `HF_HUB_DISABLE_SYMLINKS` | `1` | Prevent symlink attacks |
| `HF_HUB_OFFLINE` | `0` | Keep online for downloads only |

---

## Quantization Options

### 4-bit Quantization (Recommended for Free Tier)

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 8-bit Quantization (Balance of Speed and Memory)

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### No Quantization (Full Precision)

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16  # or torch.float32 for full
)
```

---

## Model Loading Configs

### Causal LM (LLMs, Chat Models)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)
```

### Masked LM (BERT-like models)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    device_map="auto"
)
```

### Sequence Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_LABELS
)
```

### Image Classification

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
```

### Text-to-Image Generation

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None  # Disable for faster inference
)
```

### Automatic Speech Recognition

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
```

---

## GPU Optimization

### Memory Optimization

```python
import torch

# Optimize CUDA settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Clear cache before loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Mixed Precision

```python
from torch.cuda.amp import autocast

with autocast():
    outputs = model.generate(**inputs)
```

### Gradient Checkpointing (Memory saving)

```python
model.gradient_checkpointing_enable()
```

---

## Security Settings

### Complete Security Configuration

```python
import os
import sys

def configure_security():
    """Configure complete security settings."""
    
    # Core security
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    
    # Disable parallelism for security
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Cache configuration
    os.environ["HF_HOME"] = "/content/.secure_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/content/.secure_cache"
    
    # Python hardening
    sys.dont_write_bytecode = True
    
    return True

configure_security()
```

### Secure Memory Cleanup

```python
import gc
import torch

def secure_cleanup():
    """Securely clear all sensitive data."""
    
    # Force garbage collection
    for _ in range(3):
        gc.collect()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    print("✓ Memory securely cleared")
```

---

## Troubleshooting Configs

### Out of Memory (OOM)

```python
# Solution 1: Aggressive quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Solution 2: CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Solution 3: Smaller model
MODEL_ID = "distilbert/distilgpt2"  # 82M parameters
```

### Slow Downloads

```python
# Install hf_transfer
!pip install hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Increase timeout
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1200"
```

### Authentication Issues

```python
# Get token from https://huggingface.co/settings/tokens
from huggingface_hub import login
login(token="hf_your_token_here")

# Verify access
from huggingface_hub import whoami
user = whoami()
print(f"Logged in as: {user['name']}")
```

### Model Not Found

```python
# Check model exists
from huggingface_hub import model_info
try:
    info = model_info(MODEL_ID)
    print(f"Model found: {info.id}")
except Exception as e:
    print(f"Model not found: {e}")
```

---

## Model Memory Requirements

| Model Size | Parameters | FP16 Memory | 8-bit Memory | 4-bit Memory |
|------------|------------|-------------|--------------|--------------|
| Tiny | < 1B | 2GB | 1GB | 0.5GB |
| Small | 1-3B | 6GB | 3GB | 1.5GB |
| Medium | 3-7B | 14GB | 7GB | 3.5GB |
| Large | 7-13B | 26GB | 13GB | 6.5GB |
| XL | 13B+ | 32GB+ | 16GB+ | 8GB+ |

---

## Free Tier Recommendations

### For 12GB GPU Memory

- **Best**: 2-3B parameter models with 4-bit quantization
- **Examples**: Gemma 2B, Mistral 7B (quantized), Phi-2

### For 16GB GPU Memory

- **Best**: 7B parameter models with 4-bit quantization
- **Examples**: Mistral 7B, Llama 2 7B, Qwen 7B

### CPU Only (No GPU)

- **Best**: 1B parameter models with quantization
- **Examples**: DistilGPT2, TinyLlama, Phi-2 (4-bit)

---

## Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Quantization Guide](https://huggingface.co/blog/quantization)
- [Gemma on HuggingFace](https://huggingface.co/google/gemma)
- [Memory Optimization](https://huggingface.co/docs/accelerate/usage_guides/gradient_checkpointing)

---

**License**: MIT | **Author**: MiniMax Agent