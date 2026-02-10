"""
Demo: Register KernelSageModel with vLLM's ModelRegistry and run inference.

This script demonstrates how to:
1. Register a custom model (KernelSageModel) under the "GptOssForCausalLM"
   architecture name so vLLM uses it instead of the built-in implementation.
2. Instantiate vLLM's LLM engine with the registered model and generate text.

Usage:
    python -m kernelsage.demo
"""

from vllm import LLM
from vllm.model_executor.models import ModelRegistry

from kernelsage import KernelSageModel

# Step 1: Register KernelSageModel under the "GptOssForCausalLM" architecture.
# When vLLM loads a model whose HuggingFace config reports
# architectures=["GptOssForCausalLM"], it will now use KernelSageModel.
ModelRegistry.register_model("GptOssForCausalLM", KernelSageModel)

# Step 2: Create the vLLM engine pointing at the model on HuggingFace Hub.
llm = LLM("openai/gpt-oss-20b")

# Step 3: Run a sample generation.
outputs = llm.generate(["Hello, KernelSage!"])
for output in outputs:
    print(output.outputs[0].text)
