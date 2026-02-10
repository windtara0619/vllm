"""
Demo: Register KernelSageModel with vLLM's ModelRegistry and run inference.

This script demonstrates how to:
1. Register a custom model (KernelSageModel) under the "GptOssForCausalLM"
   architecture name so vLLM uses it instead of the built-in implementation.
2. Instantiate vLLM's LLM engine with the registered model and generate text.

Requirements:
    - vLLM installed with CUDA support (GPU machine)
    - pip install vllm

Usage:
    python kernelsage/demo.py
"""

import sys
from pathlib import Path

# Add the repo root to sys.path so "kernelsage" is importable regardless of
# where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernelsage import KernelSageModel

# Step 1: Register KernelSageModel under the "GptOssForCausalLM" architecture.
# When vLLM loads a model whose HuggingFace config reports
# architectures=["GptOssForCausalLM"], it will now use KernelSageModel.
from vllm.model_executor.models.registry import ModelRegistry

ModelRegistry.register_model("GptOssForCausalLM", KernelSageModel)

# Verify the registration worked.
print("Registration successful:",
      "GptOssForCausalLM" in ModelRegistry.get_supported_archs())

# Step 2: Create the vLLM engine pointing at the model on HuggingFace Hub.
from vllm import LLM

llm = LLM("openai/gpt-oss-20b")

# Step 3: Run a sample generation.
outputs = llm.generate(["Hello, KernelSage!"])
for output in outputs:
    print(output.outputs[0].text)
