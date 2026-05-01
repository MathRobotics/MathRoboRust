#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from test_support.python_impl_bench import comparison_cases, rust_only_cases
from test_support.python_impl_bench import SKIPPED_COMPARISON_CASES


def bench_us(fn, loops: int, repeat: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        for _ in range(loops):
            fn()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter() - start) * 1e6 / loops)

    return {
        "mean_us": statistics.fmean(samples),
        "std_us": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def print_comparison(rows: list[dict[str, object]]) -> None:
    print("MathRoboRust(PyO3) vs vendor/MathRobo")
    print("-------------------------------------")
    print(
        f"{'name':18s} {'mathrobors[us]':>16s} {'mathrobo[us]':>14s} {'ratio':>10s} {'loops':>8s}"
    )
    for row in rows:
        rust = row["mathrobors"]
        vendor = row["mathrobo"]
        ratio = vendor["mean_us"] / rust["mean_us"] if rust["mean_us"] else float("inf")
        print(
            f"{row['name']:18s} "
            f"{rust['mean_us']:16.3f} "
            f"{vendor['mean_us']:14.3f} "
            f"{ratio:10.2f} "
            f"{row['loops']:8d}"
        )


def print_rust_only(rows: list[dict[str, object]]) -> None:
    print("\nMathRoboRust-only APIs")
    print("----------------------")
    print(f"{'name':18s} {'mathrobors[us]':>16s} {'std[us]':>10s} {'loops':>8s}")
    for row in rows:
        rust = row["mathrobors"]
        print(
            f"{row['name']:18s} "
            f"{rust['mean_us']:16.3f} "
            f"{rust['std_us']:10.3f} "
            f"{row['loops']:8d}"
        )


def print_skipped_cases() -> None:
    if not SKIPPED_COMPARISON_CASES:
        return
    print("\nSkipped Vendor Cases")
    print("--------------------")
    for name, note in SKIPPED_COMPARISON_CASES:
        print(f"{name}: {note}")


def render_report(
    comparison_rows: list[dict[str, object]],
    rust_only_rows: list[dict[str, object]],
    repeat: int,
    warmup: int,
    loop_scale: float,
) -> str:
    ratios = [
        row["mathrobo"]["mean_us"] / row["mathrobors"]["mean_us"]
        for row in comparison_rows
        if row["mathrobors"]["mean_us"]
    ]
    median_ratio = statistics.median(ratios) if ratios else 0.0
    fastest = max(
        comparison_rows,
        key=lambda row: row["mathrobo"]["mean_us"] / row["mathrobors"]["mean_us"],
    )
    regressions = [
        row
        for row in comparison_rows
        if row["mathrobo"]["mean_us"] / row["mathrobors"]["mean_us"] < 1.0
    ]

    lines = [
        "# MathRoboRust Benchmark Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Command settings: `repeat={repeat}`, `warmup={warmup}`, `loop_scale={loop_scale}`",
        "",
        "## Summary",
        "",
        f"- Compared `{len(comparison_rows)}` Rust-vs-vendor cases.",
        f"- Median speedup over vendor `mathrobo`: `{median_ratio:.2f}x`.",
        f"- Largest observed speedup: `{fastest['name']}` at "
        f"`{fastest['mathrobo']['mean_us'] / fastest['mathrobors']['mean_us']:.2f}x`.",
    ]
    if regressions:
        lines.append(
            "- Slower-than-vendor cases: "
            + ", ".join(
                f"`{row['name']}` ({row['mathrobo']['mean_us'] / row['mathrobors']['mean_us']:.2f}x)"
                for row in regressions
            )
            + "."
        )
    else:
        lines.append("- Slower-than-vendor cases: none.")

    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| case | mathrobors us | mathrobo us | speedup x | loops |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        ratio = row["mathrobo"]["mean_us"] / row["mathrobors"]["mean_us"]
        lines.append(
            f"| {row['name']} | {row['mathrobors']['mean_us']:.3f} | "
            f"{row['mathrobo']['mean_us']:.3f} | {ratio:.2f} | {row['loops']} |"
        )

    lines.extend(
        [
            "",
            "## Rust-only",
            "",
            "| case | mathrobors us | std us | loops |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rust_only_rows:
        lines.append(
            f"| {row['name']} | {row['mathrobors']['mean_us']:.3f} | "
            f"{row['mathrobors']['std_us']:.3f} | {row['loops']} |"
        )

    if SKIPPED_COMPARISON_CASES:
        lines.extend(["", "## Skipped Vendor Cases", ""])
        for name, note in SKIPPED_COMPARISON_CASES:
            lines.append(f"- `{name}`: {note}.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mathrobors against the vendored MathRobo Python implementation."
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--loop-scale",
        type=float,
        default=1.0,
        help="Multiply each benchmark case loop count by this factor.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    try:
        compare_rows = []
        for case in comparison_cases():
            loops = max(1, int(round(case.loops * args.loop_scale)))
            compare_rows.append(
                {
                    "name": case.name,
                    "loops": loops,
                    "mathrobors": bench_us(case.rust_fn, loops, args.repeat, args.warmup),
                    "mathrobo": bench_us(case.vendor_fn, loops, args.repeat, args.warmup),
                }
            )

        rust_only_rows = []
        for case in rust_only_cases():
            loops = max(1, int(round(case.loops * args.loop_scale)))
            rust_only_rows.append(
                {
                    "name": case.name,
                    "loops": loops,
                    "mathrobors": bench_us(case.rust_fn, loops, args.repeat, args.warmup),
                }
            )
    except ModuleNotFoundError as exc:
        missing = exc.name or "required module"
        print(f"Missing Python dependency: {missing}", file=sys.stderr)
        print("Build/install the extension and dev deps first:", file=sys.stderr)
        print("  cd python", file=sys.stderr)
        print("  uv sync --dev", file=sys.stderr)
        print("  uv run maturin develop --release", file=sys.stderr)
        raise SystemExit(1) from exc

    payload = {
        "comparison": compare_rows,
        "rust_only": rust_only_rows,
        "skipped_vendor_cases": [
            {"name": name, "reason": note} for name, note in SKIPPED_COMPARISON_CASES
        ],
    }
    print_comparison(compare_rows)
    print_rust_only(rust_only_rows)
    print_skipped_cases()

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.report_out:
        args.report_out.write_text(
            render_report(compare_rows, rust_only_rows, args.repeat, args.warmup, args.loop_scale),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
