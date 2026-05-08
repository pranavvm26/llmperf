"""PDF report generator for benchmark results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from llmperf import common_metrics

# Consistent styling
COLORS = {
    "primary": "#00b4d8",
    "secondary": "#90e0ef",
    "accent": "#caf0f8",
    "error": "#e63946",
    "text": "#1d3557",
    "bg": "#ffffff",
}

METRIC_LABELS = {
    common_metrics.INTER_TOKEN_LAT: "ITL (s)",
    common_metrics.TTFT: "TTFT (s)",
    common_metrics.E2E_LAT: "E2E Latency (s)",
    common_metrics.REQ_OUTPUT_THROUGHPUT: "Throughput (tok/s)",
    common_metrics.NUM_INPUT_TOKENS: "Input Tokens",
    common_metrics.NUM_OUTPUT_TOKENS: "Output Tokens",
}

METRIC_DESCRIPTIONS = {
    common_metrics.INTER_TOKEN_LAT: "Inter-Token Latency",
    common_metrics.TTFT: "Time to First Token",
    common_metrics.E2E_LAT: "End-to-End Latency",
    common_metrics.REQ_OUTPUT_THROUGHPUT: "Per-Request Output Throughput",
    common_metrics.NUM_INPUT_TOKENS: "Input Token Count",
    common_metrics.NUM_OUTPUT_TOKENS: "Output Token Count",
}

PERCENTILES = ["p25", "p50", "p75", "p90", "p95", "p99"]


def generate_report(
    all_step_results: List[Tuple[int, Dict[str, Any], List]],
    model: str,
    llm_api: str,
    tokenizer_name: str,
    modality: str,
    mean_input_tokens: int,
    mean_output_tokens: int,
    max_num_completed_requests: int,
    num_warmup_requests: int,
    results_dir: str,
):
    """Generate a PDF benchmark report and save it to results_dir.

    Args:
        all_step_results: List of (concurrency, summary_dict, individual_responses).
        model: Model identifier string.
        llm_api: API backend used.
        tokenizer_name: Tokenizer used.
        modality: "text" or "vision".
        mean_input_tokens: Configured mean input tokens.
        mean_output_tokens: Configured mean output tokens.
        max_num_completed_requests: Requests per concurrency step.
        num_warmup_requests: Warmup requests sent.
        results_dir: Output directory.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    pdf_path = results_path / "benchmark_report.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # Page 1: Title + config
        _add_title_page(
            pdf, model, llm_api, tokenizer_name, modality,
            mean_input_tokens, mean_output_tokens,
            max_num_completed_requests, num_warmup_requests,
            all_step_results,
        )

        # Page 2: Sweep summary table
        if len(all_step_results) > 1:
            _add_sweep_table(pdf, all_step_results)

        # Page 3: Throughput & RPM vs concurrency chart
        if len(all_step_results) > 1:
            _add_throughput_chart(pdf, all_step_results)

        # Page 4: Latency vs concurrency chart
        if len(all_step_results) > 1:
            _add_latency_chart(pdf, all_step_results)

        # Per-step detail pages
        for concurrency, summary, _ in all_step_results:
            _add_step_detail_page(pdf, concurrency, summary)

    return pdf_path


def _add_title_page(pdf, model, llm_api, tokenizer_name, modality,
                    mean_input, mean_output, max_requests, warmup,
                    all_step_results):
    """Title page with benchmark configuration and high-level results."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.93, "LLM Benchmark Report", fontsize=26, fontweight="bold",
            ha="center", va="top", color=COLORS["text"])
    ax.text(0.5, 0.88, datetime.now().strftime("%Y-%m-%d %H:%M"),
            fontsize=10, ha="center", va="top", color="gray")

    # Config section as a table for clean alignment
    config_rows = [
        ["Model", model],
        ["API", llm_api],
        ["Tokenizer", tokenizer_name],
        ["Modality", modality],
        ["Mean Input Tokens", str(mean_input)],
        ["Mean Output Tokens", str(mean_output)],
        ["Requests / Step", str(max_requests)],
        ["Warmup Requests", str(warmup)],
        ["Concurrency Steps", ", ".join(str(c) for c, _, _ in all_step_results)],
    ]

    ax.text(0.08, 0.82, "Configuration", fontsize=14, fontweight="bold",
            color=COLORS["text"])

    config_table = ax.table(
        cellText=config_rows,
        colLabels=["Parameter", "Value"],
        loc="upper left",
        cellLoc="left",
        colWidths=[0.2, 0.7],
        bbox=[0.08, 0.38, 0.84, 0.42],
    )
    config_table.auto_set_font_size(False)
    config_table.set_fontsize(10)
    config_table.scale(1.0, 1.8)

    # Style config table
    for j in range(2):
        cell = config_table[0, j]
        cell.set_facecolor(COLORS["primary"])
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(len(config_rows)):
        config_table[i + 1, 0].set_text_props(color="gray", fontweight="bold")
        config_table[i + 1, 0].set_facecolor(COLORS["bg"])
        config_table[i + 1, 1].set_facecolor(COLORS["accent"] if i % 2 == 0 else COLORS["bg"])

    # Best results highlight
    if all_step_results:
        best_throughput = 0
        best_c = 0
        for c, summary, _ in all_step_results:
            results = summary.get("results", {})
            tp = results.get(common_metrics.OUTPUT_THROUGHPUT, 0)
            if tp > best_throughput:
                best_throughput = tp
                best_c = c

        total_errors = sum(
            s.get("results", {}).get(common_metrics.NUM_ERRORS, 0)
            for _, s, _ in all_step_results
        )

        ax.text(0.08, 0.34, "Highlights", fontsize=14, fontweight="bold",
                color=COLORS["text"])

        highlight_rows = [
            ["Peak Throughput", f"{best_throughput:.1f} tok/s @ concurrency {best_c}"],
            ["Total Errors", str(total_errors)],
        ]
        hl_table = ax.table(
            cellText=highlight_rows,
            loc="upper left",
            cellLoc="left",
            colWidths=[0.2, 0.7],
            bbox=[0.08, 0.2, 0.84, 0.12],
        )
        hl_table.auto_set_font_size(False)
        hl_table.set_fontsize(11)
        hl_table.scale(1.0, 2.0)

        for i in range(len(highlight_rows)):
            hl_table[i, 0].set_text_props(color="gray", fontweight="bold")
            hl_table[i, 0].set_facecolor(COLORS["bg"])
            hl_table[i, 1].set_facecolor(COLORS["bg"])
            color = COLORS["error"] if "Error" in highlight_rows[i][0] and total_errors > 0 else COLORS["primary"]
            hl_table[i, 1].set_text_props(color=color, fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def _add_sweep_table(pdf, all_step_results):
    """Concurrency sweep summary as a table page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.92, "Concurrency Sweep Summary", fontsize=18, fontweight="bold",
            ha="center", va="top", color=COLORS["text"])

    headers = ["Concurrency", "Throughput\n(tok/s)", "Req/min",
               "E2E p50\n(s)", "E2E p99\n(s)", "TTFT p50\n(s)",
               "ITL p50\n(s)", "Errors"]

    rows = []
    for c, summary, _ in all_step_results:
        r = summary.get("results", {})
        e2e = r.get(common_metrics.E2E_LAT, {})
        ttft = r.get(common_metrics.TTFT, {})
        itl = r.get(common_metrics.INTER_TOKEN_LAT, {})
        rows.append([
            str(c),
            f"{r.get(common_metrics.OUTPUT_THROUGHPUT, 0):.1f}",
            f"{r.get(common_metrics.COMPLETED_REQUESTS_PER_MIN, 0):.1f}",
            f"{e2e.get('quantiles', {}).get('p50', 0):.3f}",
            f"{e2e.get('quantiles', {}).get('p99', 0):.3f}",
            f"{ttft.get('quantiles', {}).get('p50', 0):.4f}",
            f"{itl.get('quantiles', {}).get('p50', 0):.6f}",
            str(r.get(common_metrics.NUM_ERRORS, 0)),
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        bbox=[0.08, 0.2, 0.84, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor(COLORS["primary"])
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_facecolor(COLORS["accent"] if i % 2 == 0 else COLORS["bg"])

    pdf.savefig(fig)
    plt.close(fig)


def _add_throughput_chart(pdf, all_step_results):
    """Throughput and RPM vs concurrency chart."""
    concurrencies = [c for c, _, _ in all_step_results]
    throughputs = [
        s.get("results", {}).get(common_metrics.OUTPUT_THROUGHPUT, 0)
        for _, s, _ in all_step_results
    ]
    rpms = [
        s.get("results", {}).get(common_metrics.COMPLETED_REQUESTS_PER_MIN, 0)
        for _, s, _ in all_step_results
    ]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

    ax1.set_xlabel("Concurrency", fontsize=12)
    ax1.set_ylabel("Throughput (tok/s)", fontsize=12, color=COLORS["primary"])
    line1 = ax1.plot(concurrencies, throughputs, "o-", color=COLORS["primary"],
                     linewidth=2, markersize=8, label="Throughput (tok/s)")
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
    ax1.set_xticks(concurrencies)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Requests / min", fontsize=12, color=COLORS["error"])
    line2 = ax2.plot(concurrencies, rpms, "s--", color=COLORS["error"],
                     linewidth=2, markersize=8, label="Requests / min")
    ax2.tick_params(axis="y", labelcolor=COLORS["error"])

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    ax1.set_title("Throughput & Request Rate vs Concurrency",
                  fontsize=14, fontweight="bold", color=COLORS["text"])
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)


def _add_latency_chart(pdf, all_step_results):
    """Latency percentiles vs concurrency chart."""
    concurrencies = [c for c, _, _ in all_step_results]

    e2e_p50 = []
    e2e_p99 = []
    ttft_p50 = []
    for _, s, _ in all_step_results:
        r = s.get("results", {})
        e2e = r.get(common_metrics.E2E_LAT, {}).get("quantiles", {})
        ttft = r.get(common_metrics.TTFT, {}).get("quantiles", {})
        e2e_p50.append(e2e.get("p50", 0))
        e2e_p99.append(e2e.get("p99", 0))
        ttft_p50.append(ttft.get("p50", 0))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(concurrencies, e2e_p50, "o-", color=COLORS["primary"],
            linewidth=2, markersize=8, label="E2E Latency p50")
    ax.plot(concurrencies, e2e_p99, "s--", color=COLORS["error"],
            linewidth=2, markersize=8, label="E2E Latency p99")
    ax.plot(concurrencies, ttft_p50, "^:", color="#2a9d8f",
            linewidth=2, markersize=8, label="TTFT p50")

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Latency (seconds)", fontsize=12)
    ax.set_xticks(concurrencies)
    ax.set_title("Latency vs Concurrency",
                 fontsize=14, fontweight="bold", color=COLORS["text"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)


def _add_step_detail_page(pdf, concurrency, summary):
    """Detail page for a single concurrency step with metrics table."""
    results = summary.get("results", {})

    # Landscape orientation for wide tables
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.95, f"Concurrency: {concurrency}",
            fontsize=18, fontweight="bold", ha="center", va="top",
            color=COLORS["text"])

    # Summary stats row
    throughput = results.get(common_metrics.OUTPUT_THROUGHPUT, 0)
    completed = results.get(common_metrics.NUM_COMPLETED_REQUESTS, 0)
    rpm = results.get(common_metrics.COMPLETED_REQUESTS_PER_MIN, 0)
    errors = results.get(common_metrics.NUM_ERRORS, 0)

    summary_text = (
        f"Throughput: {throughput:.1f} tok/s    |    "
        f"Completed: {completed}    |    "
        f"Requests/min: {rpm:.1f}    |    "
        f"Errors: {errors}"
    )
    ax.text(0.5, 0.89, summary_text, fontsize=10, ha="center", va="top",
            color=COLORS["text"])

    # Metrics table
    metric_keys = [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS,
    ]

    headers = ["Metric"] + PERCENTILES + ["mean", "min", "max", "stddev"]
    rows = []
    for key in metric_keys:
        if key not in results:
            continue
        data = results[key]
        q = data.get("quantiles", {})
        label = METRIC_LABELS.get(key, key)
        row = [label]
        for p in PERCENTILES:
            row.append(_fmt_val(q.get(p, 0)))
        row.append(_fmt_val(data.get("mean", 0)))
        row.append(_fmt_val(data.get("min", 0)))
        row.append(_fmt_val(data.get("max", 0)))
        row.append(_fmt_val(data.get("stddev", 0)))
        rows.append(row)

    if rows:
        # Column widths: wider first column for metric name
        col_widths = [0.16] + [0.076] * (len(headers) - 1)

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colWidths=col_widths,
            bbox=[0.02, 0.12, 0.96, 0.72],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 2.0)

        # Style header
        for j in range(len(headers)):
            cell = table[0, j]
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)

        # Style metric name column left-aligned, data cells centered
        for i in range(len(rows)):
            for j in range(len(headers)):
                cell = table[i + 1, j]
                cell.set_facecolor(COLORS["accent"] if i % 2 == 0 else COLORS["bg"])
                if j == 0:
                    cell.set_text_props(ha="left", fontweight="bold", fontsize=9)

    # Legend at bottom
    y = 0.08
    for key in metric_keys:
        short = METRIC_LABELS.get(key, key)
        full = METRIC_DESCRIPTIONS.get(key, "")
        if full:
            ax.text(0.04, y, f"{short}", fontsize=7, fontweight="bold",
                    color=COLORS["text"])
            ax.text(0.18, y, f"— {full}", fontsize=7, color="gray")
            y -= 0.018

    pdf.savefig(fig)
    plt.close(fig)


def _fmt_val(value) -> str:
    """Format a numeric value for the table."""
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
