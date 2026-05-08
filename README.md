# AWS LLM-Perf

A CLI tool for benchmarking LLM endpoints on AWS (SageMaker, Bedrock) and other providers.

> This is a fork of [ray-project/llmperf](https://github.com/ray-project/llmperf) with the following changes:
> - Direct SageMaker (`sagemaker_direct`) and LiteLLM (`litellm`) backends
> - Installable CLI tool (`sm-benchmarker`)
> - Rich terminal output with formatted tables and PDF report generation
> - Power-of-2 concurrency sweep (1 → 2 → 4 → 8 → ... → max)
> - Multimodal (vision) benchmarking support
> - Inference Component support for multi-model endpoints
> - Model-specific tokenizer via mandatory `--tokenizer` flag
> - Endpoint warmup and `--debug` mode

## Setup

```bash
pip install uv
uv venv py312 --python 3.12
source py312/bin/activate

git clone https://github.com/pranavvm26/aws-llmperf.git
cd aws-llmperf
uv pip install -e .
```

## AWS Credentials

```bash
export AWS_REGION=us-east-1
export AWS_REGION_NAME=us-east-1
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export AWS_SESSION_TOKEN=<your-session-token>
```

## Usage

### Text — LiteLLM backend

Uses LiteLLM as the gateway to SageMaker. Works for standard JumpStart endpoints.

```bash
sm-benchmarker \
  --model "sagemaker/jumpstart-dft-hf-reasoning-qwen3-4b-20260323-223535" \
  --tokenizer "Qwen/Qwen3-4B" \
  --mean-input-tokens 1024 --stddev-input-tokens 10 \
  --mean-output-tokens 256 --stddev-output-tokens 10 \
  --max-num-completed-requests 32 --timeout 1800 \
  --num-concurrent-requests 16 \
  --results-dir "qwen-4B" --llm-api litellm \
  --warmup-requests 2
```

### Text — Inference Components

For endpoints with Inference Components (multi-model endpoints), use `sagemaker_direct` with `--inference-component-name`.

```bash
sm-benchmarker \
  --model "sagemaker/jumpstart-dft-openai-reasoning-gpt-20260401-183612" \
  --tokenizer "openai/gpt-oss-20b" \
  --inference-component-name "openai-reasoning-gpt-oss-20b-20260401-183612" \
  --mean-input-tokens 1024 --stddev-input-tokens 10 \
  --mean-output-tokens 256 --stddev-output-tokens 10 \
  --max-num-completed-requests 32 --timeout 1800 \
  --num-concurrent-requests 16 \
  --results-dir "gpt-oss-20b" --llm-api sagemaker_direct \
  --warmup-requests 2
```

To find the inference component name for an endpoint:
```bash
aws sagemaker list-inference-components \
  --endpoint-name "<your-endpoint-name>" \
  --query "InferenceComponents[].InferenceComponentName" \
  --output text
```

### Vision Example

Multimodal benchmarking sends a bundled test image (`lena.png`) with each request. Use `sagemaker_direct` for vision to bypass LiteLLM's SageMaker serialization limitations.

```bash
sm-benchmarker \
  --model "sagemaker/jumpstart-dft-hf-vlm-qwen2-vl-7b-in-20260401-164721" \
  --tokenizer "Qwen/Qwen2-VL-7B-Instruct" \
  --modality vision \
  --mean-output-tokens 256 --stddev-output-tokens 10 \
  --max-num-completed-requests 32 --timeout 1800 \
  --num-concurrent-requests 16 \
  --results-dir "qwen2-vl" --llm-api sagemaker_direct \
  --warmup-requests 2
```

### Vision

```bash
sm-benchmarker \
  --model "sagemaker/jumpstart-dft-hf-vlm-gemma-3-4b-ins-20260401-165826" \
  --tokenizer "google/gemma-3-4b-it" \
  --modality vision \
  --mean-output-tokens 256 --stddev-output-tokens 10 \
  --max-num-completed-requests 32 --timeout 1800 \
  --num-concurrent-requests 16 \
  --results-dir "gemma-3-4b" --llm-api sagemaker_direct \
  --warmup-requests 2
```

## API Backends

| Backend | Flag | Use when |
|---|---|---|
| `litellm` | `--llm-api litellm` | Standard SageMaker/Bedrock text endpoints. Uses LiteLLM as gateway. |
| `sagemaker_direct` | `--llm-api sagemaker_direct` | Inference Components, vision/multimodal, or when LiteLLM has issues. Uses boto3 directly. |
| `openai` | `--llm-api openai` | OpenAI-compatible API endpoints. |
| `sagemaker` | `--llm-api sagemaker` | Legacy direct SageMaker client (original llmperf). |

For SageMaker endpoints, prefer `sagemaker_direct` — it sends clean OpenAI-format payloads via boto3 with no intermediary.

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | Model identifier (e.g. `sagemaker/<endpoint-name>`) |
| `--tokenizer` | (required) | HuggingFace tokenizer for token counting (e.g. `Qwen/Qwen3-4B`) |
| `--llm-api` | openai | API backend: `litellm`, `sagemaker_direct`, `openai`, `sagemaker`, `vertexai` |
| `--modality` | text | Prompt modality: `text` or `vision` |
| `--inference-component-name` | (none) | SageMaker Inference Component name for multi-model endpoints |
| `--mean-input-tokens` | 550 | Mean number of input tokens per request |
| `--stddev-input-tokens` | 150 | Std dev of input tokens |
| `--mean-output-tokens` | 150 | Mean number of output tokens per request |
| `--stddev-output-tokens` | 80 | Std dev of output tokens |
| `--num-concurrent-requests` | 10 | Max concurrency. Runs power-of-2 sweep: 1, 2, 4, ..., max |
| `--max-num-completed-requests` | 10 | Number of requests to complete per concurrency step |
| `--timeout` | 90 | Timeout in seconds per concurrency step |
| `--results-dir` | (none) | Directory to save JSON results and PDF report |
| `--warmup-requests` | 0 | Warmup requests before benchmarking (first step only) |
| `--additional-sampling-params` | `{}` | Extra sampling params as JSON string |
| `--metadata` | (none) | Comma-separated key=value pairs for result metadata |
| `--debug` | off | Enable verbose output including Ray worker logs and full error messages |

## Metrics

Each concurrency step reports two tables (Latency Metrics and Throughput & Token Metrics) with percentiles (p25, p50, p75, p90, p95, p99), mean, min, max, and stddev. A summary panel follows with aggregate stats.

### Latency Metrics

| Metric | Unit | Description |
|---|---|---|
| Inter-Token Latency (ITL) | seconds | Average time between consecutive generated tokens within a single request. Lower is better. |
| Time to First Token (TTFT) | seconds | Time from sending the request to receiving the first generated token. Critical for interactive use cases. |
| End-to-End Latency (E2E) | seconds | Total wall-clock time from request sent to last token received. |

### Throughput & Token Metrics

| Metric | Unit | Description |
|---|---|---|
| Output Throughput / Request | tok/s | Output tokens generated per second for a single request. |
| Input Tokens | count | Actual number of input tokens sent per request. |
| Output Tokens | count | Actual number of tokens generated per request. |

### Summary Panel

| Metric | Description |
|---|---|
| Overall Output Throughput (tok/s) | Total output tokens / total test wall-clock time at the given concurrency. |
| Completed Requests | Requests that completed successfully. |
| Requests Per Minute | Completed requests normalized to per-minute rate. |
| Errored Requests | Count and percentage of failed requests. |
| Sample Error Messages | First few unique error messages (shown when errors > 0). |

### Concurrency Sweep Summary

When `--num-concurrent-requests` > 1, a final comparison table is printed:

| Column | Description |
|---|---|
| Concurrency | Number of concurrent requests for this step |
| Throughput (tok/s) | Overall output throughput |
| Requests/min | Completed requests per minute |
| E2E Latency p50/p99 (s) | Median and tail end-to-end latency |
| TTFT p50 (s) | Median time to first token |
| ITL p50 (s) | Median inter-token latency |
| Errors | Number of errored requests |

## Results

Results are saved to `--results-dir`:
- `*_c{N}_summary.json` — aggregate metrics for concurrency level N
- `*_c{N}_individual_responses.json` — per-request metrics
- `benchmark_report.pdf` — PDF report with charts and tables

## How It Works

1. Generates prompts: text mode uses Shakespeare sonnets, vision mode uses a bundled test image (`lena.png`) — tokenized with `--tokenizer`
2. Optionally sends warmup requests to prime cold endpoints
3. Runs a concurrency sweep at power-of-2 steps (1, 2, 4, 8, ..., max)
4. At each step, spawns concurrent Ray workers hitting the endpoint
5. Collects per-request metrics: ITL, TTFT, E2E latency, throughput
6. Prints rich formatted tables per step and a sweep summary at the end
7. Saves JSON results and a PDF report to `--results-dir`

## Caveats

- Results depend on the endpoint's backend hardware and current load.
- Results may vary with time of day and concurrent usage from other users.
- Warmup helps with cold-start endpoints but does not eliminate all variance.
- Token counts use the tokenizer specified via `--tokenizer`. Always match it to your deployed model.
- For vision mode, `--mean-input-tokens` is ignored (prompt is fixed: image + instruction).
- Inference Components must be in `InService` state to be invocable.

## License

Apache-2.0. See [LICENSE.txt](LICENSE.txt).
