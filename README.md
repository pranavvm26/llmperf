# AWS LLM-Perf

A CLI tool for benchmarking LLM endpoints on AWS (SageMaker, Bedrock) and other providers. Uses [LiteLLM](https://docs.litellm.ai/docs/providers) as the gateway.

> This is a fork of [ray-project/llmperf](https://github.com/ray-project/llmperf) with the following changes:
> - LiteLLM integration for SageMaker and Bedrock endpoints
> - Installable CLI tool (`sm-benchmarker`)
> - Rich terminal output with formatted tables
> - Power-of-2 concurrency sweep (1 → 2 → 4 → 8 → ... → max)
> - Endpoint warmup support
> - Suppressed Ray/LiteLLM log noise

## Setup

```bash
git clone https://github.com/pranavvm26/llmperf.git
cd aws-llmperf

pip install uv
uv venv py312 --python 3.12
source py312/bin/activate
uv pip install -e .
```

## AWS Credentials

```bash
export AWS_REGION=<your-region>
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export AWS_SESSION_TOKEN=<your-session-token>
```

## Usage

### SageMaker

```bash
sm-benchmarker \
  --model "sagemaker/jumpstart-dft-hf-reasoning-qwen3-4b-20260323-223535" \
  --mean-input-tokens 1024 --stddev-input-tokens 10 \
  --mean-output-tokens 1024 --stddev-output-tokens 10 \
  --max-num-completed-requests 32 --timeout 1800 \
  --num-concurrent-requests 16 \
  --results-dir "qwen-4B" --llm-api litellm \
  --warmup-requests 2
```

### Bedrock

```bash
sm-benchmarker \
  --model "bedrock/anthropic.claude-3-sonnet-20240229-v1:0" \
  --mean-input-tokens 550 --stddev-input-tokens 150 \
  --mean-output-tokens 150 --stddev-output-tokens 10 \
  --max-num-completed-requests 32 --timeout 600 \
  --num-concurrent-requests 8 \
  --results-dir "bedrock-claude" --llm-api litellm \
  --warmup-requests 1
```

### OpenAI Compatible APIs

```bash
export OPENAI_API_KEY=<your-key>
export OPENAI_API_BASE="https://api.example.com/v1"

sm-benchmarker \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --mean-input-tokens 550 --stddev-input-tokens 150 \
  --mean-output-tokens 150 --stddev-output-tokens 10 \
  --max-num-completed-requests 32 --timeout 600 \
  --num-concurrent-requests 8 \
  --results-dir "result_outputs" --llm-api openai
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | Model identifier (e.g. `sagemaker/<endpoint>`, `bedrock/<model-id>`) |
| `--mean-input-tokens` | 550 | Mean number of input tokens per request |
| `--stddev-input-tokens` | 150 | Std dev of input tokens |
| `--mean-output-tokens` | 150 | Mean number of output tokens per request |
| `--stddev-output-tokens` | 80 | Std dev of output tokens |
| `--num-concurrent-requests` | 10 | Max concurrency. Runs power-of-2 sweep: 1, 2, 4, ..., max |
| `--max-num-completed-requests` | 10 | Number of requests to complete per concurrency step |
| `--timeout` | 90 | Timeout in seconds per concurrency step |
| `--llm-api` | openai | API backend: `openai`, `litellm`, `sagemaker`, `vertexai` |
| `--results-dir` | (none) | Directory to save JSON results |
| `--warmup-requests` | 0 | Warmup requests before benchmarking (first step only) |
| `--additional-sampling-params` | `{}` | Extra sampling params as JSON string |
| `--metadata` | (none) | Comma-separated key=value pairs for result metadata |

## How It Works

1. Generates prompts from Shakespeare sonnets, tokenized with LlamaTokenizer
2. Optionally sends warmup requests to prime cold endpoints
3. Runs a concurrency sweep at power-of-2 steps (1, 2, 4, 8, ..., max)
4. At each step, spawns concurrent Ray workers hitting the endpoint via LiteLLM
5. Collects per-request metrics: inter-token latency, TTFT, E2E latency, throughput
6. Prints rich formatted tables per step and a sweep summary at the end
7. Saves per-step JSON results to `--results-dir`

## Results

Results are saved as JSON files in the `--results-dir`:
- `*_c{N}_summary.json` — aggregate metrics for concurrency level N
- `*_c{N}_individual_responses.json` — per-request metrics

## License

Apache-2.0. See [LICENSE.txt](LICENSE.txt).
