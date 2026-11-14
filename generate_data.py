"""Generate a large synthetic event dataset using PyArrow and multiprocessing.

Each day is stored in a separate Parquet file.

Default configuration:
    Start date:       2025-01-01
    Days:             5
    Registered users: 100,000
    DAU:              75%
    Events/user/day:  100 (Poisson distribution)
    CTR:              10%
    Contents:         3,000,000
    Languages:        200
    Countries:        200
    Workers:          8
    Batch size:       250,000 target events

Output:
    data/{dataset_name}/2025-01-01.parquet
    data/{dataset_name}/2025-01-02.parquet
    ...

Installation:
    uv add numpy pyarrow

Usage:
    uv run python generate_data.py

    uv run python generate_data.py \\
        --dataset-name benchmark_10m \\
        --start-date 2025-01-01 \\
        --days 5 \\
        --users 100000 \\
        --contents 3000000 \\
        --events-per-user 100 \\
        --dau 0.75 \\
        --ctr 0.10 \\
        --workers 8 \\
        --batch-size 250000

Notes:
    - Existing Parquet files in the selected dataset directory are removed.
    - Data is generated in batches to limit memory consumption.
    - Multiple days are generated concurrently.
    - User UUIDs are deterministic and stable across days.
    - Exactly floor(users * dau) users are selected per day.
    - Events per active user follow a Poisson distribution.
    - Each event has an independent 10% probability of being CLICKED.
    - Timestamps have millisecond precision.
"""

import argparse
import math
import os
import re
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_DAYS = 5
DEFAULT_USERS = 100_000
DEFAULT_CONTENTS = 3_000_000
DEFAULT_EVENTS_PER_USER = 100
DEFAULT_DAU = 0.75
DEFAULT_CTR = 0.10
DEFAULT_WORKERS = 8
DEFAULT_BATCH_SIZE = 250_000
DEFAULT_SEED = 42

MILLISECONDS_PER_DAY = 86_400_000
MAX_SERVER_DELAY_MS = 120_000

LANGUAGES = np.array(
    [f"lang_{i:03d}" for i in range(200)]
)

COUNTRIES = np.array(
    [f"country_{i:03d}" for i in range(200)]
)

EVENT_TYPES = np.array(["EXPOSED", "CLICKED"])


# ============================================================
# Arrow schema
# ============================================================

SCHEMA = pa.schema(
    [
        ("device_time", pa.timestamp("ms")),
        ("server_time", pa.timestamp("ms")),
        ("user_id", pa.string()),
        ("content_id", pa.int32()),
        ("language", pa.string()),
        ("country", pa.string()),
        ("type", pa.string()),
    ]
)


# ============================================================
# Helpers
# ============================================================

def format_number(value: int | float) -> str:
    return f"{value:,.0f}"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"

    hours, minutes = divmod(minutes, 60)

    return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"


def user_id_to_uuid(user_id: int) -> str:
    """Generate a stable UUID string from a numeric user ID.

    The same numeric user ID always produces the same UUID.

    Supports up to 2^48 distinct IDs.
    """
    return f"00000000-0000-4000-8000-{user_id:012x}"


def choose_permutation(
    rng: np.random.Generator,
    users: int,
) -> tuple[int, int]:
    """Choose parameters for a bijective mapping over user IDs.

    The mapping:

        user_id = (a * index + b) % users

    is a permutation when gcd(a, users) == 1.

    This lets us select exactly the requested number of active users
    without allocating an array containing all registered users.
    """
    if users == 1:
        return 1, 0

    while True:
        a = int(rng.integers(1, users))

        if math.gcd(a, users) == 1:
            break

    b = int(rng.integers(0, users))

    return a, b


def heading(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


# ============================================================
# Generate one Arrow batch
# ============================================================

def generate_batch(
    rng: np.random.Generator,
    day: date,
    user_ids: np.ndarray,
    event_counts: np.ndarray,
    contents: int,
    ctr: float,
) -> pa.Table:
    """Generate an Arrow table containing events for a batch of users."""

    total_events = int(event_counts.sum())

    # --------------------------------------------------------
    # User IDs
    # --------------------------------------------------------

    # Generate one stable UUID per user, then repeat according
    # to the number of events produced by that user.
    user_uuids = [
        user_id_to_uuid(int(user_id))
        for user_id in user_ids
    ]

    repeated_user_ids = np.repeat(
        user_uuids,
        event_counts,
    )

    # --------------------------------------------------------
    # Device timestamps
    # --------------------------------------------------------

    # Random millisecond within the requested day.
    device_offsets = rng.integers(
        0,
        MILLISECONDS_PER_DAY,
        size=total_events,
        dtype=np.int64,
    )

    day_start = np.datetime64(day.isoformat(), "ms")

    device_times = (
        day_start
        + device_offsets.astype("timedelta64[ms]")
    )

    # --------------------------------------------------------
    # Server timestamps
    # --------------------------------------------------------

    # Add random delay between 0 and 120,000 milliseconds.
    # Server timestamps may cross midnight into the next day.
    server_delays = rng.integers(
        0,
        MAX_SERVER_DELAY_MS + 1,
        size=total_events,
        dtype=np.int64,
    )

    server_times = (
        device_times
        + server_delays.astype("timedelta64[ms]")
    )

    # --------------------------------------------------------
    # Content IDs
    # --------------------------------------------------------

    content_ids = rng.integers(
        0,
        contents,
        size=total_events,
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # Languages
    # --------------------------------------------------------

    language_indices = rng.integers(
        0,
        len(LANGUAGES),
        size=total_events,
        dtype=np.uint8,
    )

    languages = LANGUAGES[language_indices]

    # --------------------------------------------------------
    # Countries
    # --------------------------------------------------------

    country_indices = rng.integers(
        0,
        len(COUNTRIES),
        size=total_events,
        dtype=np.uint8,
    )

    countries = COUNTRIES[country_indices]

    # --------------------------------------------------------
    # Event types
    # --------------------------------------------------------

    # Each event is independently CLICKED with probability ctr.
    clicked = rng.random(total_events) < ctr

    event_types = EVENT_TYPES[clicked.astype(np.uint8)]

    # --------------------------------------------------------
    # Create Arrow table
    # --------------------------------------------------------

    return pa.Table.from_arrays(
        [
            pa.array(
                device_times,
                type=pa.timestamp("ms"),
            ),
            pa.array(
                server_times,
                type=pa.timestamp("ms"),
            ),
            pa.array(
                repeated_user_ids,
                type=pa.string(),
            ),
            pa.array(
                content_ids,
                type=pa.int32(),
            ),
            pa.array(
                languages,
                type=pa.string(),
            ),
            pa.array(
                countries,
                type=pa.string(),
            ),
            pa.array(
                event_types,
                type=pa.string(),
            ),
        ],
        schema=SCHEMA,
    )


# ============================================================
# Generate one day
# ============================================================

def generate_day(
    day_index: int,
    start_date: date,
    output_dir: Path,
    users: int,
    contents: int,
    events_per_user: float,
    dau: float,
    ctr: float,
    batch_size: int,
    seed: int,
    progress_every: int,
) -> dict:
    """Generate one day's events and write them to Parquet.

    Designed to execute inside a separate worker process.
    """

    started = time.perf_counter()

    current_day = start_date + timedelta(days=day_index)

    output_file = output_dir / f"{current_day.isoformat()}.parquet"

    # A unique, reproducible random stream for each day.
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, day_index])
    )

    # Exactly this many users will be selected for the day.
    active_users = int(users * dau)

    # Estimated events, used for informational output.
    estimated_events = active_users * events_per_user

    # Instead of storing every active user in memory,
    # select them using a deterministic permutation.
    a, b = choose_permutation(rng, users)

    # Approximate target batch size in terms of active users.
    users_per_batch = max(
        1,
        int(batch_size / events_per_user),
    )

    rows_written = 0
    batches_written = 0
    next_progress = progress_every

    writer = None

    # Write to a temporary file first so an interrupted worker
    # does not leave a partially written final Parquet file.
    temp_file = output_file.with_suffix(
        f".{os.getpid()}.tmp"
    )

    try:
        writer = pq.ParquetWriter(
            temp_file,
            schema=SCHEMA,
            compression="zstd",
            use_dictionary=[
                "language",
                "country",
                "type",
            ],
        )

        for start in range(0, active_users, users_per_batch):
            end = min(
                start + users_per_batch,
                active_users,
            )

            # ------------------------------------------------
            # Select active users
            # ------------------------------------------------

            active_indices = np.arange(
                start,
                end,
                dtype=np.int64,
            )

            user_ids = (
                a * active_indices + b
            ) % users

            # ------------------------------------------------
            # Generate event counts per user
            # ------------------------------------------------

            event_counts = rng.poisson(
                lam=events_per_user,
                size=len(user_ids),
            )

            # Remove users who happened to generate zero events.
            nonzero = event_counts > 0

            user_ids = user_ids[nonzero]
            event_counts = event_counts[nonzero]

            if len(user_ids) == 0:
                continue

            # ------------------------------------------------
            # Generate events
            # ------------------------------------------------

            table = generate_batch(
                rng=rng,
                day=current_day,
                user_ids=user_ids,
                event_counts=event_counts,
                contents=contents,
                ctr=ctr,
            )

            # ------------------------------------------------
            # Write Arrow batch to Parquet
            # ------------------------------------------------

            writer.write_table(table)

            rows_written += table.num_rows
            batches_written += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if rows_written >= next_progress:
                elapsed = time.perf_counter() - started

                percentage = (
                    100 * rows_written / estimated_events
                    if estimated_events
                    else 100
                )

                throughput = (
                    rows_written / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"[{current_day}] "
                    f"{format_number(rows_written)} rows "
                    f"(~{percentage:.1f}%) | "
                    f"{format_number(throughput)} rows/s | "
                    f"{format_duration(elapsed)}",
                    flush=True,
                )

                next_progress = (
                    (rows_written // progress_every + 1)
                    * progress_every
                )

        writer.close()
        writer = None

        # Atomically replace the existing daily file.
        os.replace(temp_file, output_file)

    except Exception:
        if writer is not None:
            writer.close()

        if temp_file.exists():
            temp_file.unlink()

        raise

    elapsed = time.perf_counter() - started

    file_size = output_file.stat().st_size

    return {
        "day": current_day.isoformat(),
        "file": str(output_file),
        "rows": rows_written,
        "batches": batches_written,
        "size_bytes": file_size,
        "elapsed": elapsed,
    }


# ============================================================
# CLI arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic event data as daily Parquet files "
            "using PyArrow and multiprocessing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="events",
        help="Dataset directory name under ./data",
    )

    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
        help="First day to generate (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of consecutive days to generate",
    )

    parser.add_argument(
        "--users",
        type=int,
        default=DEFAULT_USERS,
        help="Total registered users",
    )

    parser.add_argument(
        "--contents",
        type=int,
        default=DEFAULT_CONTENTS,
        help="Number of distinct content IDs",
    )

    parser.add_argument(
        "--events-per-user",
        type=float,
        default=DEFAULT_EVENTS_PER_USER,
        help="Average daily events per active user",
    )

    parser.add_argument(
        "--dau",
        type=float,
        default=DEFAULT_DAU,
        help="Fraction of registered users active daily (0-1)",
    )

    parser.add_argument(
        "--ctr",
        type=float,
        default=DEFAULT_CTR,
        help="Probability that an event is CLICKED (0-1)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Target number of events per Arrow batch",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible generation",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print worker progress every N generated rows",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.dataset_name):
        parser.error(
            "--dataset-name must contain only letters, "
            "numbers, underscores, or hyphens"
        )

    if args.days <= 0:
        parser.error("--days must be greater than zero")

    if not 1 <= args.users <= 2**48:
        parser.error("--users must be between 1 and 2^48")

    if not 1 <= args.contents <= 2**31 - 1:
        parser.error(
            "--contents must be between 1 and 2^31 - 1"
        )

    if args.events_per_user <= 0:
        parser.error("--events-per-user must be positive")

    if not math.isfinite(args.events_per_user):
        parser.error("--events-per-user must be finite")

    if not 0 < args.dau <= 1:
        parser.error("--dau must be in the interval (0, 1]")

    if not 0 <= args.ctr <= 1:
        parser.error("--ctr must be in the interval [0, 1]")

    if args.workers <= 0:
        parser.error("--workers must be greater than zero")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    if args.progress_every <= 0:
        parser.error("--progress-every must be greater than zero")

    if args.seed < 0:
        parser.error("--seed must not be negative")

    if int(args.users * args.dau) == 0:
        parser.error(
            "The selected --users and --dau result in zero active users"
        )

    try:
        args.start_date + timedelta(days=args.days - 1)
    except OverflowError:
        parser.error(
            "--start-date and --days exceed the supported date range"
        )

    return args


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    output_dir = DATA_DIR / args.dataset_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    active_users = int(args.users * args.dau)

    estimated_daily_events = (
        active_users * args.events_per_user
    )

    estimated_total_events = (
        estimated_daily_events * args.days
    )

    heading("PyArrow / Parquet Dataset Generator")

    print(f"Dataset:          {args.dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Start date:       {args.start_date}")
    print(f"Days:             {args.days}")
    print(f"Registered users: {format_number(args.users)}")
    print(f"Daily active:     {format_number(active_users)}")
    print(f"DAU:              {args.dau:.0%}")
    print(f"Events/user/day:  {args.events_per_user}")
    print(f"CTR:              {args.ctr:.0%}")
    print(f"Contents:         {format_number(args.contents)}")
    print(f"Languages:        {len(LANGUAGES)}")
    print(f"Countries:        {len(COUNTRIES)}")
    print(f"Workers:          {args.workers}")
    print(f"Batch size:       {format_number(args.batch_size)}")
    print(f"Compression:      ZSTD")

    print()
    print(
        f"Estimated events/day: "
        f"{format_number(estimated_daily_events)}"
    )

    print(
        f"Estimated total rows: "
        f"{format_number(estimated_total_events)}"
    )

    # --------------------------------------------------------
    # Remove previous dataset files
    # --------------------------------------------------------

    existing_files = list(output_dir.glob("*.parquet"))

    if existing_files:
        print()
        print(
            f"Removing {len(existing_files)} existing "
            f"Parquet file(s)..."
        )

        for file in existing_files:
            file.unlink()

    # --------------------------------------------------------
    # Generate days in parallel
    # --------------------------------------------------------

    heading("Generating Parquet files")

    started = time.perf_counter()

    total_rows = 0
    total_bytes = 0
    completed_days = 0
    results = []

    worker_count = min(
        args.workers,
        args.days,
    )

    with ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:

        futures = [
            executor.submit(
                generate_day,
                day_index,
                args.start_date,
                output_dir,
                args.users,
                args.contents,
                args.events_per_user,
                args.dau,
                args.ctr,
                args.batch_size,
                args.seed,
                args.progress_every,
            )
            for day_index in range(args.days)
        ]

        for future in as_completed(futures):
            result = future.result()

            results.append(result)

            completed_days += 1
            total_rows += result["rows"]
            total_bytes += result["size_bytes"]

            elapsed = time.perf_counter() - started

            print()
            print(
                f"✓ [{completed_days}/{args.days}] "
                f"{result['day']} completed"
            )

            print(
                f"  Rows:     {format_number(result['rows'])}"
            )

            print(
                f"  File:     {result['file']}"
            )

            print(
                f"  Size:     {result['size_bytes'] / 1024**2:.2f} MB"
            )

            print(
                f"  Duration: {format_duration(result['elapsed'])}"
            )

            print(
                f"  Total elapsed: {format_duration(elapsed)}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    elapsed = time.perf_counter() - started

    throughput = (
        total_rows / elapsed
        if elapsed > 0
        else 0
    )

    heading("Generation Complete")

    print(f"Dataset:       {args.dataset_name}")
    print(f"Days:          {completed_days}")
    print(f"Total rows:    {format_number(total_rows)}")
    print(f"Total size:    {total_bytes / 1024**3:.3f} GB")
    print(f"Elapsed:       {format_duration(elapsed)}")

    print(
        f"Throughput:    "
        f"{format_number(throughput)} rows/s"
    )

    print(f"Output:        {output_dir}")

    print()
    print("Generated files:")

    for result in sorted(
        results,
        key=lambda item: item["day"],
    ):
        print(
            f"  {Path(result['file']).name:<25} "
            f"{format_number(result['rows']):>15} rows "
            f"{result['size_bytes'] / 1024**2:>10.2f} MB"
        )

    print()
    print("Example DuckDB query:")
    print()
    print(
        f"  SELECT type, COUNT(*) "
        f"FROM read_parquet("
        f"'{output_dir.as_posix()}/*.parquet') "
        f"GROUP BY type;"
    )


if __name__ == "__main__":
    main()