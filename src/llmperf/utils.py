import json
import math
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase


RESULTS_VERSION = "2023-08-31"


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


def randomly_sample_sonnet_lines_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    tokenizer: PreTrainedTokenizerBase = None,
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from the Shakespeare sonnet at sonnet.txt.

    Args:
        prompt_tokens_mean: The mean length of the prompt to generate.
        prompt_tokens_stddev: The standard deviation of the length of the prompt to generate.
        expect_output_tokens: The number of tokens to expect in the output.
        tokenizer: A HuggingFace tokenizer instance. Must be provided by the caller.

    Returns:
        A tuple of the prompt and the length of the prompt.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    get_token_length = lambda text: len(tokenizer.encode(text))

    prompt = (
        "Randomly stream lines from the following text "
        f"with {expect_output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    # get a prompt length that is at least as long as the base
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_length(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)
    sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
    with open(sonnet_path, "r") as f:
        sonnet_lines = f.readlines()
    random.shuffle(sonnet_lines)
    sampling_lines = True
    while sampling_lines:
        for line in sonnet_lines:
            line_to_add = line
            if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                # This will cut off a line in the middle of a word, but that's ok since an
                # llm should be able to handle that.
                line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                sampling_lines = False
                prompt += line_to_add
                break
            prompt += line_to_add
            remaining_prompt_tokens -= get_token_length(line_to_add)
    return (prompt, num_prompt_tokens)


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def generate_vision_prompt(
    expect_output_tokens: int = 150,
    tokenizer: PreTrainedTokenizerBase = None,
) -> Tuple[str, int]:
    """Generate a multimodal prompt with the bundled lena.png image.

    The prompt is a JSON-serialized list of content parts in OpenAI's
    multimodal format. The LiteLLM client detects this and builds the
    message accordingly.

    Args:
        expect_output_tokens: The number of tokens to request in the output.
        tokenizer: A HuggingFace tokenizer instance for counting text tokens.

    Returns:
        A tuple of (json_content_string, estimated_prompt_tokens).
    """
    import base64

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    image_path = pathlib.Path(__file__).parent.resolve() / "lena.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Vision test image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    text_prompt = (
        f"Describe this image in detail with {expect_output_tokens} output tokens. "
        "Include colors, objects, composition, and any text visible."
    )

    # OpenAI-compatible multimodal content format
    content_parts = [
        {"type": "text", "text": text_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
            },
        },
    ]

    # Estimate token count: text tokens + ~85 tokens for a standard image
    # (OpenAI uses 85 tokens for low-detail images, 170+ for high-detail)
    text_tokens = len(tokenizer.encode(text_prompt))
    estimated_image_tokens = 85
    estimated_total = text_tokens + estimated_image_tokens

    content_json = json.dumps(content_parts)
    return (content_json, estimated_total)
