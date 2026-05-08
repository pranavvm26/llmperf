"""Tokenizer factory.

Loads a HuggingFace tokenizer by explicit name. The tokenizer must always
be specified by the caller — there is no auto-resolution from model names.
"""

import logging
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_cache: dict[str, PreTrainedTokenizerBase] = {}


def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer by name.

    Uses a cache to avoid re-downloading tokenizers.

    Args:
        tokenizer_name: A HuggingFace tokenizer identifier,
            e.g. "Qwen/Qwen3-4B", "meta-llama/Llama-3.1-8B".

    Returns:
        A HuggingFace tokenizer instance.

    Raises:
        ValueError: If tokenizer_name is not provided.
        Exception: If the tokenizer cannot be loaded from HuggingFace.
    """
    if not tokenizer_name:
        raise ValueError(
            "--tokenizer is required. Specify a HuggingFace tokenizer, "
            "e.g. --tokenizer Qwen/Qwen3-4B"
        )

    if tokenizer_name in _cache:
        return _cache[tokenizer_name]

    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    _cache[tokenizer_name] = tokenizer
    return tokenizer
