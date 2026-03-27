"""Rich CLI output for benchmark results."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

METRIC_LABELS = {
    "inter_token_latency_s": ("Inter-Token Latency", "s"),
    "ttft_s": ("Time to First Token", "s"),
    "end_to_end_latency_s": ("End-to-End Latency", "s"),
    "request_output_throughput_token_per_s": ("Output Throughput / Request", "tok/s"),
    "number_input_tokens": ("Input Tokens", ""),
    "number_output_tokens": ("Output Tokens", ""),
}

PERCENTILE_KEYS = ["p25", "p50", "p75", "p90", "p95", "p99"]


def _fmt(value, unit: str) -> str:
    """Format a numeric value with appropriate precision."""
    if isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.6f}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        elif abs(value) < 100:
            return f"{value:.2f}"
        else:
            return f"{value:.1f}"
    return str(value)


def print_benchmark_header(model: str, llm_api: str, num_concurrent: int,
                           mean_input: int, mean_output: int, max_requests: int):
    """Print a styled header with benchmark configuration."""
    header = Text()
    header.append("LLMPerf Benchmark", style="bold cyan")
    header.append("  │  ", style="dim")
    header.append(f"Model: ", style="dim")
    header.append(model, style="bold white")
    header.append("  │  ", style="dim")
    header.append(f"API: ", style="dim")
    header.append(llm_api, style="bold white")

    config_text = Text()
    config_text.append(f"Concurrent: ", style="dim")
    config_text.append(str(num_concurrent), style="bold yellow")
    config_text.append("  │  ", style="dim")
    config_text.append(f"Input tokens: ", style="dim")
    config_text.append(f"~{mean_input}", style="bold yellow")
    config_text.append("  │  ", style="dim")
    config_text.append(f"Output tokens: ", style="dim")
    config_text.append(f"~{mean_output}", style="bold yellow")
    config_text.append("  │  ", style="dim")
    config_text.append(f"Requests: ", style="dim")
    config_text.append(str(max_requests), style="bold yellow")

    content = Text()
    content.append_text(header)
    content.append("\n")
    content.append_text(config_text)

    console.print(Panel(content, border_style="cyan", box=box.HEAVY))


def print_latency_table(results: dict):
    """Print latency metrics (ITL, TTFT, E2E) in a styled table."""
    latency_keys = [
        "inter_token_latency_s",
        "ttft_s",
        "end_to_end_latency_s",
    ]
    _print_metrics_table("Latency Metrics", latency_keys, results)


def print_throughput_table(results: dict):
    """Print throughput and token count metrics in a styled table."""
    throughput_keys = [
        "request_output_throughput_token_per_s",
        "number_input_tokens",
        "number_output_tokens",
    ]
    _print_metrics_table("Throughput & Token Metrics", throughput_keys, results)


def _print_metrics_table(title: str, keys: list, results: dict):
    """Build and print a rich table for a set of metrics."""
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold bright_white",
        border_style="cyan",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("Metric", style="bold white", min_width=28)
    for p in PERCENTILE_KEYS:
        table.add_column(p, justify="right", style="bright_white", min_width=10)
    table.add_column("mean", justify="right", style="bold green", min_width=10)
    table.add_column("min", justify="right", style="dim", min_width=10)
    table.add_column("max", justify="right", style="dim", min_width=10)
    table.add_column("stddev", justify="right", style="yellow", min_width=10)

    for key in keys:
        if key not in results:
            continue
        label, unit = METRIC_LABELS.get(key, (key, ""))
        data = results[key]
        quantiles = data.get("quantiles", {})
        suffix = f" ({unit})" if unit else ""

        row = [f"{label}{suffix}"]
        for p in PERCENTILE_KEYS:
            row.append(_fmt(quantiles.get(p, "—"), unit))
        row.append(_fmt(data.get("mean", "—"), unit))
        row.append(_fmt(data.get("min", "—"), unit))
        row.append(_fmt(data.get("max", "—"), unit))
        row.append(_fmt(data.get("stddev", "—"), unit))
        table.add_row(*row)

    console.print(table)


def print_summary_panel(results: dict):
    """Print the overall summary stats in a panel."""
    num_errors = results.get("number_errors", 0)
    error_rate = results.get("error_rate", 0)
    throughput = results.get("mean_output_throughput_token_per_s", 0)
    completed = results.get("num_completed_requests", 0)
    rpm = results.get("num_completed_requests_per_min", 0)
    error_freq = results.get("error_code_frequency", "{}")

    grid = Table(box=None, show_header=False, padding=(0, 3))
    grid.add_column(style="dim", min_width=30)
    grid.add_column(style="bold bright_white", min_width=20)

    error_style = "bold red" if num_errors > 0 else "bold green"

    grid.add_row("Overall Output Throughput", f"[bold cyan]{_fmt(throughput, 'tok/s')}[/] tok/s")
    grid.add_row("Completed Requests", f"[bold cyan]{completed}[/]")
    grid.add_row("Requests Per Minute", f"[bold cyan]{_fmt(rpm, '')}[/]")
    grid.add_row("Errored Requests", f"[{error_style}]{num_errors}[/]  ({_fmt(error_rate * 100, '')}%)")

    if num_errors > 0:
        grid.add_row("Error Code Frequency", f"[yellow]{error_freq}[/]")

    console.print(Panel(grid, title="Summary", border_style="cyan",
                        title_align="left", box=box.HEAVY))


def print_results(model: str, llm_api: str, results: dict,
                  num_concurrent: int = 0, mean_input: int = 0,
                  mean_output: int = 0, max_requests: int = 0):
    """Print the full formatted benchmark results."""
    console.print()
    print_benchmark_header(model, llm_api, num_concurrent, mean_input,
                           mean_output, max_requests)
    console.print()
    print_latency_table(results)
    console.print()
    print_throughput_table(results)
    console.print()
    print_summary_panel(results)
    console.print()
