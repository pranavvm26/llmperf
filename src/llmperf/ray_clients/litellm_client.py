import time
import logging
from typing import Any, Dict
import ray

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class LiteLLMClient(LLMClient):
    """Client for LiteLLM Completions API."""

    def __init__(self):
        # Suppress noisy LiteLLM loggers inside Ray workers
        for name in ("LiteLLM", "litellm", "httpx"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        # litellm package isn't serializable, so we import it within the function
        # to maintain compatibility with ray.
        import json as _json
        from litellm import completion, validate_environment

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        # Detect multimodal content (JSON-serialized content parts from vision mode)
        is_multimodal = False
        try:
            content_parts = _json.loads(prompt)
            if isinstance(content_parts, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content_parts
            ):
                # Multimodal: use content parts directly
                user_content = content_parts
                is_multimodal = True
            else:
                user_content = prompt
        except (ValueError, TypeError, AttributeError, _json.JSONDecodeError):
            user_content = prompt

        message = [
            {"role": "user", "content": user_content},
        ]
        if not is_multimodal:
            message.insert(0, {"role": "system", "content": ""})

        assert (
            request_config.llm_api is not None
        ), "the request config's llm_api must be set."
        if request_config.llm_api == "litellm":
            model = request_config.model
        else:
            model = request_config.llm_api + "/" + request_config.model

        sampling_params = request_config.sampling_params

        # For multimodal on SageMaker, use the chat completion path
        # which properly handles OpenAI-format content parts.
        # We also need to set hf_model_id so LiteLLM sends the correct
        # model name in the request body (TGI rejects endpoint names).
        if is_multimodal and "sagemaker/" in model:
            model = model.replace("sagemaker/", "sagemaker_chat/", 1)
        # Also handle case where user passed sagemaker_chat/ directly
        if is_multimodal and "sagemaker_chat/" in model:
            if request_config.tokenizer_name:
                if sampling_params is None:
                    sampling_params = {}
                sampling_params["hf_model_id"] = request_config.tokenizer_name

        validation_result = validate_environment(model)
        if validation_result["missing_keys"]:
            raise ValueError(
                f"The following environment vars weren't found but were necessary for "
                f"the model {request_config.model}: {validation_result['missing_keys']}"
            )
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        body.update(sampling_params or {})

        # Pass inference component name if provided
        if request_config.inference_component_name:
            body["model_id"] = request_config.inference_component_name

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        try:
            start_time = time.monotonic()
            most_recent_received_token_time = time.monotonic()

            response = completion(**body)
            ttft = 0
            for tok in response:
                if tok.choices[0].delta:
                    delta = tok.choices[0].delta
                    if delta.get("content", None):
                        if ttft == 0:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        generated_text += delta["content"]
                        most_recent_received_token_time = time.monotonic()
                        tokens_received += 1

            total_request_time = time.monotonic() - start_time

            output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            metrics[common_metrics.ERROR_CODE] = error_response_code

            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len
        return metrics, generated_text, request_config
