"""Direct SageMaker client using boto3.

Bypasses LiteLLM entirely. Sends OpenAI-compatible chat payloads
directly to SageMaker endpoints via invoke_endpoint_with_response_stream.
Supports both text and vision (multimodal) prompts.
"""

import io
import json
import os
import time
from typing import Any, Dict

import boto3
import ray

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from llmperf.tokenizer_factory import get_tokenizer


@ray.remote
class SageMakerDirectClient(LLMClient):
    """Direct boto3 client for SageMaker endpoints."""

    def __init__(self):
        self._tokenizer = None
        self._sm_runtime = None

    def _get_runtime(self):
        if self._sm_runtime is None:
            region = (
                os.environ.get("AWS_REGION_NAME")
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
            )
            if not region:
                raise ValueError("AWS_REGION_NAME or AWS_REGION must be set.")
            self._sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
        return self._sm_runtime

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        import json as _json

        prompt = request_config.prompt
        prompt_text, prompt_len = prompt

        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(request_config.tokenizer_name)

        # Extract endpoint name from model string
        # Supports: "sagemaker/endpoint-name", "sagemaker_direct/endpoint-name", or just "endpoint-name"
        model = request_config.model
        for prefix in ("sagemaker_direct/", "sagemaker/", "sagemaker_chat/"):
            if model.startswith(prefix):
                model = model[len(prefix):]
                break

        # Build OpenAI-compatible message content
        # Detect multimodal (JSON-serialized content parts)
        try:
            content_parts = _json.loads(prompt_text)
            if isinstance(content_parts, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content_parts
            ):
                user_content = content_parts
            else:
                user_content = prompt_text
        except (ValueError, TypeError, _json.JSONDecodeError):
            user_content = prompt_text

        messages = [{"role": "user", "content": user_content}]

        # Build request payload
        sampling_params = dict(request_config.sampling_params or {})
        payload = {
            "messages": messages,
            "stream": True,
            **sampling_params,
        }

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            sm_runtime = self._get_runtime()
            invoke_kwargs = {
                "EndpointName": model,
                "ContentType": "application/json",
                "Body": _json.dumps(payload),
                "CustomAttributes": "accept_eula=true",
            }
            if request_config.inference_component_name:
                invoke_kwargs["InferenceComponentName"] = request_config.inference_component_name
            response = sm_runtime.invoke_endpoint_with_response_stream(**invoke_kwargs)
            event_stream = response["Body"]
            json_byte_list = []
            ttft_captured = False

            for line, line_ttft, _ in _LineIterator(event_stream):
                if not line.strip():
                    continue
                if line == "[DONE]":
                    break
                try:
                    json_chunk = _json.loads(line)
                    json_byte_list.append(json_chunk)

                    if not ttft_captured:
                        ttft = line_ttft - start_time
                        ttft_captured = True

                    time_to_next_token.append(
                        time.monotonic() - most_recent_received_token_time
                    )
                    most_recent_received_token_time = time.monotonic()

                except _json.JSONDecodeError:
                    continue

            total_request_time = time.monotonic() - start_time
            generated_text = "".join(
                chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                for chunk in json_byte_list
            )
            tokens_received = len(self._tokenizer.encode(generated_text))
            if total_request_time > 0:
                output_throughput = tokens_received / total_request_time

        except Exception as e:
            error_msg = str(e)
            error_response_code = 500

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config


class _LineIterator:
    """Parse SSE byte stream from SageMaker streaming response."""

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0
        self.ttft = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                if self.ttft == 0:
                    self.ttft = time.monotonic()
                self.read_pos += len(line)
                decoded_line = line.decode("utf-8").strip()
                if decoded_line.startswith("data: "):
                    decoded_line = decoded_line[6:]
                return decoded_line, self.ttft, time.monotonic()
            if line and self.read_pos == self.buffer.getbuffer().nbytes - 1:
                self.read_pos += 1
                decoded_line = line.decode("utf-8").strip()
                if decoded_line.startswith("data: "):
                    decoded_line = decoded_line[6:]
                return decoded_line, self.ttft, time.monotonic()
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])
