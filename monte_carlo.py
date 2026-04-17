#!/usr/bin/env python3
"""
Monte Carlo Simulation Runner
Runs 1000 simulations on numerical data from a CSV or JSON input file.

Usage:
    python monte_carlo.py <input_file> [--simulations 1000] [--output results.csv]

Supported file formats: CSV, JSON
Expected data: Numeric columns representing variables/parameters.
"""

import argparse
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────
# File Loading
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV or JSON file into a DataFrame."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Use .csv or .json")

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    dropped = set(df.columns) - set(numeric_df.columns)
    if dropped:
        print(f"  [info] Non-numeric columns ignored: {', '.join(dropped)}")

    if numeric_df.empty:
        raise ValueError("No numeric columns found in the input file.")

    return numeric_df


# ──────────────────────────────────────────────
# Monte Carlo Core
# ──────────────────────────────────────────────

def fit_distributions(df: pd.DataFrame) -> dict:
    """
    Estimate mean and std for each column to parameterise
    a normal distribution used in sampling.
    """
    stats = {}
    for col in df.columns:
        data = df[col].dropna()
        stats[col] = {
            "mean": float(data.mean()),
            "std": float(data.std()) if data.std() > 0 else float(abs(data.mean()) * 0.01 + 1e-9),
            "min": float(data.min()),
            "max": float(data.max()),
            "count": int(len(data)),
        }
    return stats


def run_simulations(df: pd.DataFrame, n_simulations: int, seed: int = 42) -> pd.DataFrame:
    """
    Run Monte Carlo simulations by sampling from fitted distributions.

    For each simulation:
      - Draw a random sample row from the input data (with replacement).
      - Add Gaussian noise scaled to each column's std deviation.
      - Compute a composite score (normalised weighted sum across columns).

    Returns a DataFrame with one row per simulation.
    """
    rng = np.random.default_rng(seed)
    dist_params = fit_distributions(df)
    col_names = list(df.columns)
    data_matrix = df[col_names].dropna().values  # shape: (rows, cols)

    results = []

    print(f"\n  Running {n_simulations:,} simulations across "
          f"{len(col_names)} variable(s)...\n")

    start = time.perf_counter()

    for i in range(n_simulations):
        # Sample a base row from the empirical data
        base_idx = rng.integers(0, len(data_matrix))
        base_row = data_matrix[base_idx].copy()

        # Perturb each variable using its fitted std
        noise = np.array([
            rng.normal(0, dist_params[col]["std"])
            for col in col_names
        ])
        sim_row = base_row + noise

        # Clamp to observed range (optional but realistic)
        for j, col in enumerate(col_names):
            sim_row[j] = np.clip(
                sim_row[j],
                dist_params[col]["min"] - 3 * dist_params[col]["std"],
                dist_params[col]["max"] + 3 * dist_params[col]["std"],
            )

        # Composite score: mean of z-scored variables
        z_scores = np.array([
            (sim_row[j] - dist_params[col]["mean"]) / dist_params[col]["std"]
            for j, col in enumerate(col_names)
        ])
        composite_score = float(z_scores.mean())

        row = {"simulation_id": i + 1}
        for j, col in enumerate(col_names):
            row[col] = round(float(sim_row[j]), 6)
        row["composite_score"] = round(composite_score, 6)
        results.append(row)

        # Progress indicator every 10 %
        if (i + 1) % max(1, n_simulations // 10) == 0:
            pct = (i + 1) / n_simulations * 100
            elapsed = time.perf_counter() - start
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"  [{bar}] {pct:5.1f}%  —  {i+1:,} / {n_simulations:,}  "
                  f"({elapsed:.2f}s elapsed)")

    elapsed_total = time.perf_counter() - start
    print(f"\n  ✓ Completed in {elapsed_total:.3f}s "
          f"({n_simulations / elapsed_total:,.0f} sims/sec)\n")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# Summary Statistics
# ──────────────────────────────────────────────

def summarise(results: pd.DataFrame, dist_params: dict) -> None:
    """Print a formatted summary of simulation outcomes."""
    numeric_cols = [c for c in results.columns
                    if c not in ("simulation_id",) and pd.api.types.is_numeric_dtype(results[c])]

    print("=" * 60)
    print("  SIMULATION SUMMARY")
    print("=" * 60)

    for col in numeric_cols:
        series = results[col]
        label = "composite_score" if col == "composite_score" else col
        print(f"\n  Variable : {label}")
        print(f"    Mean   : {series.mean():.4f}")
        print(f"    Std    : {series.std():.4f}")
        print(f"    Min    : {series.min():.4f}")
        print(f"    P5     : {series.quantile(0.05):.4f}")
        print(f"    P25    : {series.quantile(0.25):.4f}")
        print(f"    Median : {series.median():.4f}")
        print(f"    P75    : {series.quantile(0.75):.4f}")
        print(f"    P95    : {series.quantile(0.95):.4f}")
        print(f"    Max    : {series.max():.4f}")

    # Probability that composite score > 0 (baseline)
    prob_positive = (results["composite_score"] > 0).mean()
    print(f"\n  P(composite_score > 0) : {prob_positive:.2%}")
    print("=" * 60)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations on a numeric data file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_file", help="Path to input CSV or JSON file")
    parser.add_argument(
        "--simulations", "-n",
        type=int, default=1000,
        help="Number of simulations to run (default: 1000)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Path to save simulation results CSV (default: <input>_mc_results.csv)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the summary statistics table",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  MONTE CARLO SIMULATION RUNNER")
    print("=" * 60)
    print(f"  Input file  : {args.input_file}")
    print(f"  Simulations : {args.simulations:,}")
    print(f"  Random seed : {args.seed}")

    # ── Load data ──────────────────────────────
    print("\n  Loading data...")
    try:
        df = load_data(args.input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  [error] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} numeric columns: "
          f"{', '.join(df.columns)}")

    # ── Fit distributions ──────────────────────
    dist_params = fit_distributions(df)

    # ── Run simulations ────────────────────────
    results = run_simulations(df, n_simulations=args.simulations, seed=args.seed)

    # ── Summary ────────────────────────────────
    if not args.no_summary:
        summarise(results, dist_params)

    # ── Save results ───────────────────────────
    if args.output:
        out_path = args.output
    else:
        stem = Path(args.input_file).stem
        out_path = f"{stem}_mc_results.csv"

    results.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")
    print(f"  Shape: {results.shape[0]:,} rows × {results.shape[1]} columns\n")


if __name__ == "__main__":
    main()
