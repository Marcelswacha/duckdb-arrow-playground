"""Benchmark DuckDB queries against a multi-file Parquet dataset.

Each SQL query lives in its own file under ./queries/.

Examples:
    uv run python benchmark.py --query top_clickable_users
    uv run python benchmark.py --query content_engagement --threads 8
    uv run python benchmark.py --query all --runs 5 --warmup 2
    uv run python benchmark.py --list
    uv run python benchmark.py --query daily_user_retention --explain

The generator creates:
    data/{dataset_name}/2025-01-01.parquet
    data/{dataset_name}/2025-01-02.parquet
    ...

DuckDB reads all matching Parquet files through a single SQL view
called "events". SQL files can reference that view directly.
"""

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
QUERIES_DIR = ROOT / "queries"

DEFAULT_DATASET = "benchmark_10m"
DEFAULT_THREADS = 4
DEFAULT_RUNS = 5
DEFAULT_WARMUP = 1


def heading(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def load_queries() -> dict[str, Path]:
    """Discover SQL queries from ./queries/*.sql."""
    return {
        path.stem: path
        for path in sorted(QUERIES_DIR.glob("*.sql"))
    }


def parse_args() -> argparse.Namespace:
    queries = load_queries()

    parser = argparse.ArgumentParser(
        description="Benchmark DuckDB queries on daily Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET,
        help="Dataset directory under ./data",
    )

    parser.add_argument(
        "--query",
        default="all",
        choices=["all", *queries.keys()],
        help="SQL query to execute, or 'all'",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="DuckDB execution threads",
    )

    parser.add_argument(
        "--memory-limit",
        default="2GB",
        help="DuckDB memory limit, e.g. 512MB, 2GB, 8GB",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Measured executions per query",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Unmeasured warmup executions per query",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print DuckDB EXPLAIN plan before benchmarking",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available SQL queries and exit",
    )

    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")

    if args.runs < 1:
        parser.error("--runs must be positive")

    if args.warmup < 0:
        parser.error("--warmup must not be negative")

    if not args.list and not queries:
        parser.error(f"No SQL queries found in {QUERIES_DIR}")

    return args


def benchmark_query(
    con: duckdb.DuckDBPyConnection,
    name: str,
    sql: str,
    runs: int,
    warmup: int,
    explain: bool,
) -> dict:
    """Benchmark a SQL query, including result materialization."""

    heading(f"Query: {name}")

    print(sql.strip())

    if explain:
        heading(f"EXPLAIN: {name}")
        plan = con.execute(f"EXPLAIN {sql}").fetchone()
        print(plan[1])

    # Warmup runs are not included in measurements.
    for i in range(warmup):
        con.execute(sql).fetchall()
        print(f"  Warmup {i + 1}/{warmup}: completed")

    timings = []
    result_rows = 0
    sample = []

    for i in range(runs):
        start = time.perf_counter()

        result = con.execute(sql).fetchall()

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

        result_rows = len(result)
        sample = result[:3]

        print(
            f"  Run {i + 1}/{runs}: "
            f"{elapsed:.4f}s "
            f"({result_rows:,} result rows)"
        )

    minimum = min(timings)
    median = statistics.median(timings)
    maximum = max(timings)
    mean = statistics.mean(timings)

    print()
    print(f"  Min:    {minimum:.4f}s")
    print(f"  Median: {median:.4f}s")
    print(f"  Mean:   {mean:.4f}s")
    print(f"  Max:    {maximum:.4f}s")

    if sample:
        print("\n  Sample results:")
        for row in sample:
            print(f"    {row}")

    return {
        "query": name,
        "min_seconds": minimum,
        "median_seconds": median,
        "mean_seconds": mean,
        "max_seconds": maximum,
        "result_rows": result_rows,
    }


def main() -> None:
    args = parse_args()
    available_queries = load_queries()

    if args.list:
        heading("Available SQL Queries")
        for name in available_queries:
            print(f"  {name}")
        return

    dataset_dir = DATA_DIR / args.dataset_name

    if not dataset_dir.is_dir():
        raise SystemExit(
            f"Dataset not found: {dataset_dir}\n"
            f"Generate it first with:\n"
            f"  make generate DATASET_NAME={args.dataset_name}"
        )

    parquet_files = sorted(dataset_dir.glob("*.parquet"))

    if not parquet_files:
        raise SystemExit(
            f"No Parquet files found in {dataset_dir}"
        )

    selected_queries = (
        available_queries
        if args.query == "all"
        else {args.query: available_queries[args.query]}
    )

    heading("DuckDB Parquet Benchmark")

    print(f"Dataset:        {args.dataset_name}")
    print(f"Parquet files:  {len(parquet_files)}")
    print(f"Threads:        {args.threads}")
    print(f"Memory limit:   {args.memory_limit}")
    print(f"Runs:           {args.runs}")
    print(f"Warmup:         {args.warmup}")
    print(f"Queries:        {', '.join(selected_queries)}")

    total_size = sum(p.stat().st_size for p in parquet_files)
    print(f"Dataset size:   {total_size / 1024**3:.3f} GB")

    results = []

    with duckdb.connect(":memory:") as con:
        # Thread count is validated as a positive integer.
        con.execute(f"SET threads = {args.threads}")

        # Set memory limit using a parameter.
        con.execute(
            "SET memory_limit = ?",
            [args.memory_limit],
        )

        # Register ALL daily Parquet files as one logical SQL view.
        # Files remain on disk; no dataset import is required.
        parquet_glob = str(dataset_dir / "*.parquet")

        con.from_parquet(parquet_glob).create_view("events")

        heading("Dataset Metadata")

        metadata = con.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT filename) AS files
            FROM read_parquet(?, filename = true)
            """,
            [parquet_glob],
        ).fetchone()

        print(f"Total rows: {metadata[0]:,}")
        print(f"Files:      {metadata[1]:,}")

        for name, query_file in selected_queries.items():
            sql = query_file.read_text(encoding="utf-8").strip()
            sql = sql.rstrip(";").strip()

            if not sql:
                raise ValueError(f"Empty query: {query_file}")

            result = benchmark_query(
                con=con,
                name=name,
                sql=sql,
                runs=args.runs,
                warmup=args.warmup,
                explain=args.explain,
            )

            result["dataset"] = args.dataset_name
            result["threads"] = args.threads
            result["memory_limit"] = args.memory_limit
            result["timestamp"] = datetime.now().isoformat(
                timespec="seconds"
            )

            results.append(result)

    heading("Benchmark Summary")

    print(
        f"{'Query':<30} "
        f"{'Min (s)':>10} "
        f"{'Median (s)':>12} "
        f"{'Max (s)':>10}"
    )

    print("-" * 66)

    for result in results:
        print(
            f"{result['query']:<30} "
            f"{result['min_seconds']:>10.4f} "
            f"{result['median_seconds']:>12.4f} "
            f"{result['max_seconds']:>10.4f}"
        )

    if args.csv:
        csv_path = (
            args.csv
            if args.csv.is_absolute()
            else ROOT / args.csv
        )

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()