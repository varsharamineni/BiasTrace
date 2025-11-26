"""
LM Configuration for DSPy Judge Script

This file contains the configuration for the language model API.
Modify these settings to match your vLLM server setup.
"""

import dspy

# ================================================================
# Default LM Configuration
# ================================================================

# Model configuration
DEFAULT_MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

# API configuration
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_CUSTOM_LLM_PROVIDER = "openai"

# Timeout and retry settings
DEFAULT_TIMEOUT = 600.0  # 10 minutes per request
DEFAULT_MAX_RETRIES = 3  # Retry failed requests up to 3 times

# Sampling parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048


# ================================================================
# LM Factory Function
# ================================================================

def create_lm(
    model: str = None,
    api_base: str = None,
    api_key: str = None,
    custom_llm_provider: str = None,
    timeout: float = None,
    max_retries: int = None,
    temperature: float = None,
    max_tokens: int = None,
):
    """
    Create a DSPy LM instance with the given configuration.
    
    Args:
        model: Model name/path (defaults to DEFAULT_MODEL)
        api_base: API base URL (defaults to DEFAULT_API_BASE)
        api_key: API key (defaults to DEFAULT_API_KEY)
        custom_llm_provider: LLM provider type (defaults to DEFAULT_CUSTOM_LLM_PROVIDER)
        timeout: Request timeout in seconds (defaults to DEFAULT_TIMEOUT)
        max_retries: Max retry attempts (defaults to DEFAULT_MAX_RETRIES)
        temperature: Sampling temperature (defaults to DEFAULT_TEMPERATURE)
        max_tokens: Max tokens to generate (defaults to DEFAULT_MAX_TOKENS)
    
    Returns:
        dspy.LM: Configured language model instance
    """
    return dspy.LM(
        model=model or DEFAULT_MODEL,
        api_base=api_base or DEFAULT_API_BASE,
        api_key=api_key or DEFAULT_API_KEY,
        custom_llm_provider=custom_llm_provider or DEFAULT_CUSTOM_LLM_PROVIDER,
        timeout=timeout or DEFAULT_TIMEOUT,
        max_retries=max_retries or DEFAULT_MAX_RETRIES,
        temperature=temperature or DEFAULT_TEMPERATURE,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
    )


# ================================================================
# Pre-configured LM instances (optional)
# ================================================================

# Example: Local vLLM server with Nemotron model
local_nemotron = dspy.LM(
    model=DEFAULT_MODEL,
    api_base=DEFAULT_API_BASE,
    api_key=DEFAULT_API_KEY,
    custom_llm_provider=DEFAULT_CUSTOM_LLM_PROVIDER,
    timeout=DEFAULT_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
)

# You can add more pre-configured instances here
# Example:
# local_qwen = dspy.LM(
#     model="Qwen/Qwen3-4B",
#     api_base="http://localhost:8001/v1",
#     api_key="dummy",
#     custom_llm_provider="openai",
#     timeout=600.0,
#     max_retries=3,
#     temperature=0.7,
#     max_tokens=2048,
# )

