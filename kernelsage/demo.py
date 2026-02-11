"""
Demo: Register KernelSageModel with vLLM's ModelRegistry and run inference.

This script demonstrates how to:
1. Run baseline inference with the stock model.
2. Register a custom model (KernelSageModel) under the "GptOssForCausalLM"
   architecture name so vLLM uses it instead of the built-in implementation.
3. Register KernelSageAttentionBackend as the CUSTOM attention backend.
4. Re-run inference with the custom model + backend and compare outputs.

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
from kernelsage.triton_attn import KernelSageAttentionBackend


def main():
    from vllm import LLM
    from vllm.config import AttentionConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    prompt = "Hello, KernelSage!"

    # ------------------------------------------------------------------
    # Step 1: Baseline — run with the stock model (no custom registrations).
    # ------------------------------------------------------------------
    print("=== Baseline (stock model) ===")
    baseline_llm = LLM("openai/gpt-oss-20b")
    baseline_outputs = baseline_llm.generate([prompt])
    baseline_text = baseline_outputs[0].outputs[0].text
    print(baseline_text)
    del baseline_llm  # free GPU memory before loading the next engine

    # ------------------------------------------------------------------
    # Step 2: Register KernelSageModel under the "GptOssForCausalLM"
    # architecture so vLLM uses it instead of the built-in implementation.
    # ------------------------------------------------------------------
    ModelRegistry.register_model("GptOssForCausalLM", KernelSageModel)
    print("\nModel registration successful:",
          "GptOssForCausalLM" in ModelRegistry.get_supported_archs())

    # ------------------------------------------------------------------
    # Step 3: Register KernelSageAttentionBackend as the CUSTOM backend.
    # ------------------------------------------------------------------
    register_backend(AttentionBackendEnum.CUSTOM,
                     "kernelsage.triton_attn.KernelSageAttentionBackend")
    print("Attention backend registered:",
          AttentionBackendEnum.CUSTOM.is_overridden())

    # ------------------------------------------------------------------
    # Step 4: Create a new LLM engine with the custom attention backend.
    # ------------------------------------------------------------------
    print("\n=== KernelSage (custom model + attention backend) ===")
    kernelsage_llm = LLM(
        "openai/gpt-oss-20b",
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )
    kernelsage_outputs = kernelsage_llm.generate([prompt])
    kernelsage_text = kernelsage_outputs[0].outputs[0].text
    print(kernelsage_text)

    # ------------------------------------------------------------------
    # Step 5: Compare the two outputs.
    # ------------------------------------------------------------------
    print("\n=== Comparison ===")
    print(f"Baseline:    {baseline_text!r}")
    print(f"KernelSage:  {kernelsage_text!r}")
    print(f"Match: {baseline_text == kernelsage_text}")


if __name__ == "__main__":
    # Optional but harmless on Linux; required on Windows/frozen apps.
    import multiprocessing as mp
    mp.freeze_support()
    main()
