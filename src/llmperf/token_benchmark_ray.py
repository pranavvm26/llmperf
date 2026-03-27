import threading
import argparse
from collections.abc import Iterable
import json
import logging
import os
from pathlib import Path
import re
import time
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)

from transformers import LlamaTokenizerFast
from llmperf.rich_output import print_results, console

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

# Suppress noisy loggers
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


def _suppress_ray_noise():
    """Set env vars to reduce Ray and LiteLLM log spam."""
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_DEDUP_LOGS_AGG_WINDOW_S", "60")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    # Suppress Ray worker logs from being forwarded to driver
    os.environ.setdefault("RAY_LOG_TO_DRIVER_ENABLED", "0")


def _make_progress_bar() -> Progress:
    """Create a clean rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=40, style="dim", complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _power_of_2_steps(max_concurrent: int) -> List[int]:
    """Generate power-of-2 concurrency steps up to max_concurrent.

    E.g. max_concurrent=16 -> [1, 2, 4, 8, 16]
         max_concurrent=10 -> [1, 2, 4, 8, 10]
    """
    steps = []
    c = 1
    while c < max_concurrent:
        steps.append(c)
        c *= 2
    steps.append(max_concurrent)
    return steps


def _run_warmup(
    model: str,
    llm_api: str,
    num_warmup_requests: int,
    prompts: List[str],
    num_output_tokens_list: List[int],
    additional_sampling_params: Dict[str, Any],
):
    """Send warmup requests to prime the endpoint before benchmarking."""
    console.print(f"\n[dim]Warming up endpoint with {num_warmup_requests} request(s)...[/dim]")

    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Warmup"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan"),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("warmup", total=num_warmup_requests)
        for i in range(num_warmup_requests):
            idx = i % len(prompts)
            default_sampling_params = {"max_tokens": num_output_tokens_list[idx]}
            default_sampling_params.update(additional_sampling_params)
            request_config = RequestConfig(
                model=model,
                prompt=prompts[idx],
                sampling_params=default_sampling_params,
                llm_api=llm_api,
            )
            req_launcher.launch_requests(request_config)
            outs = req_launcher.get_next_ready(block=True)
            progress.update(task, advance=1)

    console.print("[green]Warmup complete.[/green]\n")


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
    num_warmup_requests: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".
        num_warmup_requests: Number of warmup requests to send before benchmarking.

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0
    # make up prompts outside of send loop for faster benchmarking loop
    num_output_tokens_list = []
    prompts = []
    for i in range(max_num_completed_requests):
        num_output_tokens = (sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        ))
        num_output_tokens_list.append(num_output_tokens)

        prompts.append(randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=mean_input_tokens,
            prompt_tokens_stddev=stddev_input_tokens,
            expect_output_tokens=num_output_tokens,
            tokenizer=tokenizer
        ))

    # Run warmup if requested
    if num_warmup_requests > 0:
        _run_warmup(
            model=model,
            llm_api=llm_api,
            num_warmup_requests=num_warmup_requests,
            prompts=prompts,
            num_output_tokens_list=num_output_tokens_list,
            additional_sampling_params=additional_sampling_params,
        )

    start_time = time.monotonic()
    progress = _make_progress_bar()
    progress.start()
    task_id = progress.add_task(
        f"Benchmarking ({num_concurrent_requests} concurrent)",
        total=max_num_completed_requests,
    )

    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        request_index = thread_index % max_num_completed_requests

        while (
            time.monotonic() - start_time < test_timeout_s
            and num_completed_requests < max_num_completed_requests
        ):

            default_sampling_params = {"max_tokens": num_output_tokens_list[request_index]}
            default_sampling_params.update(additional_sampling_params)
            request_config = RequestConfig(
                model=model,
                prompt=prompts[request_index],
                sampling_params=default_sampling_params,
                llm_api=llm_api,
            )
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                num_output_tokens = get_token_length(gen_text)
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        if num_output_tokens:
                            request_metrics[common_metrics.INTER_TOKEN_LAT] /= request_metrics[common_metrics.NUM_OUTPUT_TOKENS]
                        else:
                            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                        e2e_lat = request_metrics[common_metrics.E2E_LAT]
                        if e2e_lat > 0:
                            request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / e2e_lat
                        else:
                            request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = 0
                        all_metrics.append(request_metrics)
                        completed_requests.extend(all_metrics)
                        progress.update(task_id, advance=len(all_metrics))
                        num_completed_requests += len(all_metrics)
                        request_index = (request_index + num_concurrent_requests) % max_num_completed_requests

    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    progress.stop()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        console.print("[yellow]Test timed out before all requests could be completed.[/yellow]")

    # check one last time that there are no remaining results to collect.
    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        num_output_tokens = get_token_length(gen_text)
        with completed_requests_lock:
            if num_completed_requests < max_num_completed_requests:
                if num_output_tokens:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                e2e_lat = request_metrics[common_metrics.E2E_LAT]
                if e2e_lat > 0:
                    request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / e2e_lat
                else:
                    request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = 0
                completed_requests.extend(request_metrics)

    ret = metrics_summary(completed_requests, start_time, end_time)

    print_results(
        model=model,
        llm_api=llm_api,
        results=ret,
        num_concurrent=num_concurrent_requests,
        mean_input=mean_input_tokens,
        mean_output=mean_output_tokens,
        max_requests=max_num_completed_requests,
    )

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret

    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS
    ]:
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        ret[key]["mean"] = series.mean()
        ret[key]["min"] = series.min()
        ret[key]["max"] = series.max()
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    error_code_frequency = dict(error_codes.value_counts())
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    num_warmup_requests: int = 0,
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The max number of concurrent requests. Benchmark runs
            at power-of-2 steps from 1 up to this value.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
        num_warmup_requests: Number of warmup requests before benchmarking.
    """
    if mean_input_tokens < 40:
        console.print(
            "[yellow]The minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now.[/yellow]"
        )

    concurrency_steps = _power_of_2_steps(num_concurrent_requests)
    console.print(
        f"\n[cyan]Concurrency sweep:[/cyan] {' → '.join(str(s) for s in concurrency_steps)}\n"
    )

    all_step_results = []

    for step_idx, concurrency in enumerate(concurrency_steps):
        console.rule(
            f"[cyan]Step {step_idx + 1}/{len(concurrency_steps)}: "
            f"{concurrency} concurrent request{'s' if concurrency > 1 else ''}[/cyan]"
        )

        # Only warmup on the first step
        warmup = num_warmup_requests if step_idx == 0 else 0

        summary, individual_responses = get_token_throughput_latencies(
            model=model,
            llm_api=llm_api,
            test_timeout_s=test_timeout_s,
            max_num_completed_requests=max_num_completed_requests,
            mean_input_tokens=mean_input_tokens,
            stddev_input_tokens=stddev_input_tokens,
            mean_output_tokens=mean_output_tokens,
            stddev_output_tokens=stddev_output_tokens,
            num_concurrent_requests=concurrency,
            additional_sampling_params=json.loads(additional_sampling_params),
            num_warmup_requests=warmup,
        )

        all_step_results.append((concurrency, summary, individual_responses))

        if results_dir:
            _save_results(
                summary=summary,
                individual_responses=individual_responses,
                model=model,
                mean_input_tokens=mean_input_tokens,
                mean_output_tokens=mean_output_tokens,
                concurrency=concurrency,
                results_dir=results_dir,
                user_metadata=user_metadata,
            )

    # Print concurrency sweep summary if more than one step
    if len(concurrency_steps) > 1:
        _print_sweep_summary(all_step_results)


def _save_results(
    summary: Dict,
    individual_responses: List,
    model: str,
    mean_input_tokens: int,
    mean_output_tokens: int,
    concurrency: int,
    results_dir: str,
    user_metadata: Dict[str, Any],
):
    """Save benchmark results to JSON files."""
    filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}_c{concurrency}"
    filename = re.sub(r"[^\w\d-]+", "-", filename)
    filename = re.sub(r"-{2,}", "-", filename)
    summary_filename = f"{filename}_summary"
    individual_responses_filename = f"{filename}_individual_responses"

    summary.update(user_metadata)

    results = LLMPerfResults(name=summary_filename, metadata=summary)
    results_dir_path = Path(results_dir)
    if not results_dir_path.exists():
        results_dir_path.mkdir(parents=True)
    elif not results_dir_path.is_dir():
        raise ValueError(f"{results_dir} is not a directory")

    try:
        with open(results_dir_path / f"{summary_filename}.json", "w") as f:
            json.dump(results.to_dict(), f, indent=4, default=str)
    except Exception as e:
        console.print_exception()
        raise e

    try:
        with open(results_dir_path / f"{individual_responses_filename}.json", "w") as f:
            json.dump(individual_responses, f, indent=4)
    except Exception as e:
        console.print_exception()
        raise e


def _print_sweep_summary(all_step_results: List[Tuple]):
    """Print a summary table comparing all concurrency steps."""
    from rich.table import Table
    from rich import box

    console.print()
    console.rule("[cyan]Concurrency Sweep Summary[/cyan]")

    table = Table(
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold bright_white",
        border_style="cyan",
        padding=(0, 1),
    )

    table.add_column("Concurrency", justify="center", style="bold yellow")
    table.add_column("Throughput (tok/s)", justify="right", style="bold cyan")
    table.add_column("Requests/min", justify="right", style="bright_white")
    table.add_column("E2E Latency p50 (s)", justify="right", style="bright_white")
    table.add_column("E2E Latency p99 (s)", justify="right", style="bright_white")
    table.add_column("TTFT p50 (s)", justify="right", style="bright_white")
    table.add_column("ITL p50 (s)", justify="right", style="bright_white")
    table.add_column("Errors", justify="right", style="bright_white")

    for concurrency, summary, _ in all_step_results:
        results = summary.get("results", {})
        throughput = results.get(common_metrics.OUTPUT_THROUGHPUT, 0)
        rpm = results.get(common_metrics.COMPLETED_REQUESTS_PER_MIN, 0)
        e2e = results.get(common_metrics.E2E_LAT, {})
        ttft = results.get(common_metrics.TTFT, {})
        itl = results.get(common_metrics.INTER_TOKEN_LAT, {})
        errors = results.get(common_metrics.NUM_ERRORS, 0)

        e2e_p50 = e2e.get("quantiles", {}).get("p50", 0)
        e2e_p99 = e2e.get("quantiles", {}).get("p99", 0)
        ttft_p50 = ttft.get("quantiles", {}).get("p50", 0)
        itl_p50 = itl.get("quantiles", {}).get("p50", 0)

        error_style = "bold red" if errors > 0 else "green"

        table.add_row(
            str(concurrency),
            f"{throughput:.1f}",
            f"{rpm:.1f}",
            f"{e2e_p50:.3f}",
            f"{e2e_p99:.3f}",
            f"{ttft_p50:.4f}",
            f"{itl_p50:.6f}",
            f"[{error_style}]{errors}[/]",
        )

    console.print(table)
    console.print()


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="sm-benchmarker",
        description="SageMaker / LLM endpoint token throughput and latency benchmark.",
    )

    parser.add_argument(
        "--model", type=str, required=True, help="The model to use for this load test."
    )
    parser.add_argument(
        "--mean-input-tokens",
        type=int,
        default=550,
        help=(
            "The mean number of tokens to send in the prompt for the request. "
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--stddev-input-tokens",
        type=int,
        default=150,
        help=(
            "The standard deviation of number of tokens to send in the prompt for the request. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--mean-output-tokens",
        type=int,
        default=150,
        help=(
            "The mean number of tokens to generate from each llm request. This is the max_tokens param "
            "for the completions API. Note that this is not always the number of tokens returned. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--stddev-output-tokens",
        type=int,
        default=80,
        help=(
            "The stdandard deviation on the number of tokens to generate per llm request. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=int,
        default=10,
        help=(
            "The max number of concurrent requests. Benchmark runs at power-of-2 steps "
            "from 1 up to this value (1, 2, 4, 8, ..., max). (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="The amount of time to run the load test for per concurrency step. (default: %(default)s)",
    )
    parser.add_argument(
        "--max-num-completed-requests",
        type=int,
        default=10,
        help=(
            "The number of requests to complete before finishing the test. Note "
            "that its possible for the test to timeout first. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--additional-sampling-params",
        type=str,
        default="{}",
        help=(
            "Additional sampling params to send with the each request to the LLM API. "
            "(default: %(default)s) No additional sampling params are sent."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help=(
            "The directory to save the results to. "
            "(`default: %(default)s`) No results are saved)"
        ),
    )
    parser.add_argument(
        "--llm-api",
        type=str,
        default="openai",
        help=(
            f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="",
        help=(
            "A comma separated list of metadata to include in the results, e.g. "
            "name=foo,bar=1. These will be added to the metadata field of the results. "
        ),
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help=(
            "Number of warmup requests to send before starting the benchmark. "
            "Useful for priming cold endpoints like SageMaker. (default: %(default)s)"
        ),
    )

    return parser


def main():
    """CLI entry point for sm-benchmarker."""
    _suppress_ray_noise()
    env_vars = dict(os.environ)
    ray.init(
        runtime_env={"env_vars": env_vars},
        logging_level=logging.ERROR,
        log_to_driver=False,
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run_token_benchmark(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        mean_input_tokens=args.mean_input_tokens,
        stddev_input_tokens=args.stddev_input_tokens,
        mean_output_tokens=args.mean_output_tokens,
        stddev_output_tokens=args.stddev_output_tokens,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
        num_warmup_requests=args.warmup_requests,
    )


if __name__ == "__main__":
    main()
