"""Model Inference Utilities using vLLM.

Provides generic text generation interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class ModelInference:
    """vLLM-based model inference wrapper."""
    
    def __init__(
        self,
        model_path: str | Path,
        system_prompt: str = "You are a helpful assistant that translates English text to Chinese.",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.5,
        dtype: str = "auto",
    ) -> None:
        """Initialize model inference.
        
        Args:
            model_path: Path to model directory
            system_prompt: System prompt for the model (default: translation assistant)
            max_new_tokens: Maximum new tokens to generate (default: 4096)
            temperature: Sampling temperature (default: 0.0 for deterministic)
            top_p: Top-p sampling parameter (default: 1.0)
            gpu_memory_utilization: GPU memory utilization ratio (default: 0.9)
            dtype: Data type for model weights (default: "auto")
        """
        self.model_path = str(model_path)
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"max_new_tokens={max_new_tokens}, temperature={temperature}, dtype={dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=self.model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            skip_special_tokens=True,
        )
        
        logger.info("Model loaded successfully")
        logger.info(f"System prompt: {self.system_prompt}")
    
    def generate(self, user_input: str) -> str:
        """Generate text from user input.
        
        Args:
            user_input: Raw user input from blockchain (e.g., "hello")
            
        Returns:
            Generated text (e.g., "你好")
        """
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Apply chat template if available
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            formatted_prompt = f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:"
        
        # Generate
        outputs = self.llm.generate([formatted_prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text


__all__ = [
    "ModelInference",
]
