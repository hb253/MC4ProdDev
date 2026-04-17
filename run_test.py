#!/usr/bin/env python3
"""
End-to-End Test Runner
Orchestrates the full Monte Carlo pipeline:
  1. generate_samples.py  → data.csv          (100 records)
  2. monte_carlo.py       → data_mc_results.csv (1000 simulations)
  3. plot_results.py      → mc_chart.png       (visualisation)

Usage:
    python run_test.py [--records 100] [--simulations 1000] [--seed 42]

All three scripts must be in the same directory as this file.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(msg):  print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'─'*60}{RESET}")
def ok(msg):      print(f"  {GREEN}✓  {msg}{RESET}")
def info(msg):    print(f"  {YELLOW}→  {msg}{RESET}")
def err(msg):     print(f"  {RED}✗  {msg}{RESET}", file=sys.stderr)


# ── Helpers ──────────────────────────────────────────────────────────────────

def here() -> Path:
    """Directory containing this script."""
    return Path(__file__).parent.resolve()


def require_script(name: str) -> Path:
    path = here() / name
    if not path.exists():
        err(f"Required script not found: {path}")
        sys.exit(1)
    return path


def run_step(label: str, cmd: list[str]) -> float:
    """Run a subprocess, stream its output, and return elapsed seconds."""
    info(f"Running: {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        err(f"Step '{label}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    ok(f"'{label}' completed in {elapsed:.2f}s")
    return elapsed


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full Monte Carlo test pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--records",     "-r", type=int, default=100,  help="Rows to generate (default: 100)")
    parser.add_argument("--simulations", "-n", type=int, default=1000, help="MC simulations (default: 1000)")
    parser.add_argument("--seed",        "-s", type=int, default=42,   help="Random seed (default: 42)")
    parser.add_argument("--data",              type=str, default="data.csv",            help="Intermediate data file")
    parser.add_argument("--results",           type=str, default="data_mc_results.csv", help="MC results file")
    parser.add_argument("--chart",             type=str, default="mc_chart.png",        help="Output chart file")
    args = parser.parse_args()

    # ── Locate scripts ────────────────────────────────────────────────────────
    gen_script  = require_script("generate_samples.py")
    mc_script   = require_script("monte_carlo.py")
    plot_script = require_script("plot_results.py")

    python = sys.executable   # use the same interpreter that launched this script

    wall_start = time.perf_counter()

    # ── Step 1: Generate sample data ─────────────────────────────────────────
    banner("STEP 1 / 3 — Generate Sample Data")
    run_step(
        "generate_samples",
        [python, str(gen_script),
         "--records", str(args.records),
         "--output",  args.data,
         "--seed",    str(args.seed)],
    )
    data_path = here() / args.data
    if not data_path.exists():
        err(f"Expected output file not found: {data_path}")
        sys.exit(1)
    ok(f"Data file present: {data_path}  ({data_path.stat().st_size:,} bytes)")

    # ── Step 2: Run Monte Carlo simulations ───────────────────────────────────
    banner("STEP 2 / 3 — Run Monte Carlo Simulations")
    run_step(
        "monte_carlo",
        [python, str(mc_script),
         str(data_path),
         "--simulations", str(args.simulations),
         "--output",      args.results,
         "--seed",        str(args.seed)],
    )
    results_path = here() / args.results
    if not results_path.exists():
        err(f"Expected results file not found: {results_path}")
        sys.exit(1)

    import csv
    with open(results_path) as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1   # subtract header
    ok(f"Results file present: {results_path}  ({row_count:,} simulation rows)")

    # ── Step 3: Plot results ──────────────────────────────────────────────────
    banner("STEP 3 / 3 — Generate Chart")
    run_step(
        "plot_results",
        [python, str(plot_script),
         str(results_path),
         "--output", args.chart],
    )
    chart_path = here() / args.chart
    if not chart_path.exists():
        err(f"Expected chart file not found: {chart_path}")
        sys.exit(1)
    ok(f"Chart file present: {chart_path}  ({chart_path.stat().st_size:,} bytes)")

    # ── Summary ───────────────────────────────────────────────────────────────
    wall_elapsed = time.perf_counter() - wall_start
    banner("PIPELINE COMPLETE")
    print(f"  {'Data file':<20} {args.data}")
    print(f"  {'Results file':<20} {args.results}")
    print(f"  {'Chart file':<20} {args.chart}")
    print(f"  {'Records generated':<20} {args.records:,}")
    print(f"  {'Simulations run':<20} {args.simulations:,}")
    print(f"  {'Total time':<20} {wall_elapsed:.2f}s")
    print()


if __name__ == "__main__":
    main()
