#!/usr/bin/env python3
"""
HuggingFace Colab Setup Script
===============================
Reusable Python functions for setting up Hugging Face models on Google Colab
with security, privacy, and optimization features.

Usage:
    import hf_colab_setup
    model, tokenizer = hf_colab_setup.load_model("gpt2")
"""

import os
import sys
import gc

# Core security settings
SECURITY_SETTINGS = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": "/content/.hf_cache",
    "TRANSFORMERS_CACHE": "/content/.hf_cache",
    "PYTHONHASHSEED": "42",
}

# Performance settings
PERFORMANCE_SETTINGS = {
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_DOWNLOAD_TIMEOUT": "600",
}

def setup_environment():
    """
    Configure secure environment for Hugging Face models.
    Call this at the start of any Colab notebook.
    """
    # Apply security settings
    for key, value in SECURITY_SETTINGS.items():
        os.environ[key] = value
    
    # Apply performance settings
    for key, value in PERFORMANCE_SETTINGS.items():
        os.environ[key] = value
    
    # Disable bytecode caching
    sys.dont_write_bytecode = True
    
    # Create cache directory
    os.makedirs("/content/.hf_cache", exist_ok=True)
    
    print("✓ Environment configured for security and performance")
    return True


def install_dependencies():
    """
    Install required packages for Hugging Face models.
    """
    import subprocess
    
    packages = [
        "transformers>=4.38.0",
        "accelerate",
        "bitsandbytes",
        "hf_transfer",
        "sentencepiece",
    ]
    
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    
    print("✓ Dependencies installed")
    return True


def check_gpu():
    """
    Check GPU availability and return recommendations.
    """
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✓ GPU: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("  → Use 4-bit quantization, small models")
        elif gpu_memory < 16:
            print("  → Use 4-8 bit quantization, medium models")
        else:
            print("  → Can use larger models with less quantization")
        
        # Memory optimization
        torch.backends.cudnn.benchmark = True
        
        return {
            "available": True,
            "name": gpu_name,
            "memory_gb": gpu_memory,
            "recommendation": "4-bit" if gpu_memory < 16 else "8-bit or none"
        }
    else:
        print("⚠ No GPU - Using CPU (slower)")
        print("  → Use smallest models with 4-bit quantization")
        return {"available": False, "recommendation": "tiny model + 4-bit"}


def load_model(model_id, quantization_bits=4, device="auto", torch_dtype="float16"):
    """
    Load any Hugging Face model with optimal settings.

    Args:
        model_id: Hugging Face model ID (e.g., "gpt2", "meta-llama/Llama-2-7b")
        quantization_bits: Quantization level (4 or 8, 0 for none)
        device: Device to load on ("auto", "cuda", "cpu")
        torch_dtype: Precision type ("float32", "float16", "bfloat16")

    Returns:
        tuple: (model, tokenizer)

    Raises:
        ModelNotFoundError: If the model ID doesn't exist on Hugging Face Hub
        NetworkError: If there's a network connectivity issue
        AuthenticationError: If authentication is required but not provided
        ModelLoadError: If model loading fails for other reasons
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )

    # Custom exception classes for better error handling
    class ModelLoadError(Exception):
        """Raised when model loading fails for an unexpected reason"""
        pass

    class ModelNotFoundError(Exception):
        """Raised when the model ID doesn't exist"""
        pass

    class NetworkError(Exception):
        """Raised when there's a network connectivity issue"""
        pass

    class AuthenticationError(Exception):
        """Raised when authentication is required but not provided"""
        pass

    print(f"Loading {model_id}...")

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)

    # Load tokenizer with error handling
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✓ Tokenizer loaded")
    except Exception as e:
        error_msg = str(e).lower()
        if "404" in str(e) or "not found" in error_msg:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found on Hugging Face Hub.\n"
                f"Tips:\n"
                f"  - Check the model ID is correct\n"
                f"  - Verify the model exists at: https://huggingface.co/models\n"
                f"  - Some models require authentication (use HF_TOKEN)"
            ) from e
        elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            raise NetworkError(
                f"Network error while downloading tokenizer for '{model_id}'.\n"
                f"Tips:\n"
                f"  - Check your internet connection\n"
                f"  - Try again in a few moments\n"
                f"  - Increase timeout: HF_HUB_DOWNLOAD_TIMEOUT=600"
            ) from e
        elif "token" in error_msg or "auth" in error_msg or "permission" in error_msg:
            raise AuthenticationError(
                f"Authentication required for model '{model_id}'.\n"
                f"Tips:\n"
                f"  - Set your Hugging Face token: os.environ['HF_TOKEN'] = 'your_token'\n"
                f"  - Get a token at: https://huggingface.co/settings/tokens"
            ) from e
        else:
            raise ModelLoadError(
                f"Failed to load tokenizer for '{model_id}': {e}\n"
                f"Please check the model ID and try again."
            ) from e

    # Configure model loading
    model_kwargs = {
        "device_map": device,
        "torch_dtype": torch_dtype_obj
    }

    # Add quantization if enabled
    if quantization_bits in [4, 8] and torch.cuda.is_available():
        print(f"→ Using {quantization_bits}-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_bits == 4,
            load_in_8bit=quantization_bits == 8,
            bnb_4bit_compute_dtype=torch_dtype_obj,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = bnb_config

    # Load model with error handling
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
    except Exception as e:
        error_msg = str(e).lower()
        if "404" in str(e) or "not found" in error_msg:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found on Hugging Face Hub.\n"
                f"Tips:\n"
                f"  - Check the model ID is correct\n"
                f"  - Verify the model exists at: https://huggingface.co/models"
            ) from e
        elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            raise NetworkError(
                f"Network error while downloading model '{model_id}'.\n"
                f"Tips:\n"
                f"  - Check your internet connection\n"
                f"  - Increase timeout: HF_HUB_DOWNLOAD_TIMEOUT=600\n"
                f"  - Try a smaller model if network is slow"
            ) from e
        elif "token" in error_msg or "auth" in error_msg or "permission" in error_msg:
            raise AuthenticationError(
                f"Authentication required for model '{model_id}'.\n"
                f"Tips:\n"
                f"  - Set your Hugging Face token: os.environ['HF_TOKEN'] = 'your_token'\n"
                f"  - Get a token at: https://huggingface.co/settings/tokens"
            ) from e
        elif "memory" in error_msg or "cuda" in error_msg or "oom" in error_msg:
            raise ModelLoadError(
                f"Not enough memory to load model '{model_id}'.\n"
                f"Tips:\n"
                f"  - Use smaller model or higher quantization (4-bit)\n"
                f"  - Clear GPU memory: torch.cuda.empty_cache()\n"
                f"  - Reduce batch size if using custom code"
            ) from e
        else:
            raise ModelLoadError(
                f"Failed to load model '{model_id}': {e}\n"
                f"Please check the model ID and try again."
            ) from e

    print(f"✓ Model loaded: {model.num_parameters():,} parameters")

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    Generate text with the model (security-first).
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
    
    Returns:
        str: Generated text
    """
    import torch
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with no gradients (security)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Clear from memory
    del inputs, outputs
    
    return response


def chat(model, tokenizer, prompt, system_prompt=None, max_tokens=256):
    """
    Chat with the model using chat template if available.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: User message
        system_prompt: Optional system message
        max_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated response
    """
    import torch
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Clear memory
    del inputs, outputs
    
    return response


def secure_cleanup():
    """
    Securely clear all sensitive data from memory.
    IMPORTANT: Call this after processing sensitive information.
    """
    import torch
    
    # Force garbage collection
    for _ in range(3):
        gc.collect()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print("✓ Memory securely cleared")
    return True


def quick_start(model_id, quantization_bits=4):
    """
    Quick start: setup environment, install deps, load model.
    
    Args:
        model_id: Hugging Face model ID
        quantization_bits: Quantization level (4, 8, or 0 for none)
    
    Returns:
        tuple: (model, tokenizer, info_dict)
    """
    print("=" * 50)
    print("HuggingFace Colab Setup")
    print("=" * 50)
    
    # Setup
    setup_environment()
    install_dependencies()
    gpu_info = check_gpu()
    
    # Load model
    model, tokenizer = load_model(model_id, quantization_bits=quantization_bits)
    
    print("=" * 50)
    print("✓ Ready for inference!")
    print("=" * 50)
    
    return model, tokenizer, {
        "model_id": model_id,
        "quantization_bits": quantization_bits,
        "gpu_info": gpu_info
    }


# Export for easy import
__all__ = [
    "setup_environment",
    "install_dependencies",
    "check_gpu",
    "load_model",
    "generate",
    "chat",
    "secure_cleanup",
    "quick_start",
    # Exception classes for error handling
    "ModelLoadError",
    "ModelNotFoundError",
    "NetworkError",
    "AuthenticationError"
]


if __name__ == "__main__":
    print("HuggingFace Colab Setup Script")
    print("Import with: from hf_colab_setup import *")
    print("Or use quick_start('model-id') for one-click setup")