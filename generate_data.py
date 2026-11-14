"""Generate synthetic daily Parquet data using PyArrow and multiprocessing.

Each day is written into its own Hive partition directory:
    data/{dataset_name}/day=2025-01-01/events.parquet

Run:
    uv run python generate_data.py --days 5 --users 100000 --workers 8

Dependencies: numpy, pyarrow
Existing Parquet files under the chosen dataset are overwritten on rerun.
"""

import argparse
import math
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

LANGUAGES = np.array([f"lang_{i:03d}" for i in range(200)])
COUNTRIES = np.array([f"country_{i:03d}" for i in range(200)])
EVENT_TYPES = np.array(["EXPOSED", "CLICKED"])

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
    """Generate a stable UUID-formatted string for each numeric user ID."""
    return f"00000000-0000-4000-8000-{user_id:012x}"


def choose_permutation(rng: np.random.Generator, users: int) -> tuple[int, int]:
    """Choose a bijection over numeric IDs without storing all IDs in RAM."""
    if users == 1:
        return 1, 0
    while True:
        a = int(rng.integers(1, users))
        if math.gcd(a, users) == 1:
            break
    b = int(rng.integers(0, users))
    return a, b


def heading(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


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
    user_uuids = [user_id_to_uuid(int(user_id)) for user_id in user_ids]
    repeated_user_ids = np.repeat(user_uuids, event_counts)

    device_offsets = rng.integers(
        0, MILLISECONDS_PER_DAY, size=total_events, dtype=np.int64
    )
    day_start = np.datetime64(day.isoformat(), "ms")
    device_times = day_start + device_offsets.astype("timedelta64[ms]")

    server_delays = rng.integers(
        0, MAX_SERVER_DELAY_MS + 1, size=total_events, dtype=np.int64
    )
    server_times = device_times + server_delays.astype("timedelta64[ms]")

    content_ids = rng.integers(0, contents, size=total_events, dtype=np.int32)
    language_indices = rng.integers(0, len(LANGUAGES), size=total_events, dtype=np.uint8)
    country_indices = rng.integers(0, len(COUNTRIES), size=total_events, dtype=np.uint8)
    clicked = rng.random(total_events) < ctr

    return pa.Table.from_arrays(
        [
            pa.array(device_times, type=pa.timestamp("ms")),
            pa.array(server_times, type=pa.timestamp("ms")),
            pa.array(repeated_user_ids, type=pa.string()),
            pa.array(content_ids, type=pa.int32()),
            pa.array(LANGUAGES[language_indices], type=pa.string()),
            pa.array(COUNTRIES[country_indices], type=pa.string()),
            pa.array(EVENT_TYPES[clicked.astype(np.uint8)], type=pa.string()),
        ],
        schema=SCHEMA,
    )


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
    """Generate one day into its own Hive partition, in a worker process."""
    started = time.perf_counter()
    current_day = start_date + timedelta(days=day_index)

    # CHANGED: Hive partition layout, one file per day.
    partition_dir = output_dir / f"day={current_day.isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_file = partition_dir / "events.parquet"

    rng = np.random.default_rng(np.random.SeedSequence([seed, day_index]))
    active_users = int(users * dau)
    estimated_events = active_users * events_per_user
    a, b = choose_permutation(rng, users)
    users_per_batch = max(1, int(batch_size / events_per_user))

    rows_written = 0
    batches_written = 0
    next_progress = progress_every
    writer = None
    temp_file = output_file.with_suffix(f".{os.getpid()}.tmp")

    try:
        writer = pq.ParquetWriter(
            temp_file,
            schema=SCHEMA,
            compression="zstd",
            use_dictionary=["language", "country", "type"],
        )

        for start in range(0, active_users, users_per_batch):
            end = min(start + users_per_batch, active_users)
            active_indices = np.arange(start, end, dtype=np.int64)
            user_ids = (a * active_indices + b) % users
            event_counts = rng.poisson(lam=events_per_user, size=len(user_ids))
            nonzero = event_counts > 0
            user_ids = user_ids[nonzero]
            event_counts = event_counts[nonzero]
            if len(user_ids) == 0:
                continue

            table = generate_batch(
                rng=rng,
                day=current_day,
                user_ids=user_ids,
                event_counts=event_counts,
                contents=contents,
                ctr=ctr,
            )
            writer.write_table(table)
            rows_written += table.num_rows
            batches_written += 1

            if rows_written >= next_progress:
                elapsed = time.perf_counter() - started
                percentage = 100 * rows_written / estimated_events if estimated_events else 100
                throughput = rows_written / elapsed if elapsed > 0 else 0
                print(
                    f"[{current_day}] {format_number(rows_written)} rows "
                    f"(~{percentage:.1f}%) | {format_number(throughput)} rows/s | "
                    f"{format_duration(elapsed)}",
                    flush=True,
                )
                next_progress = (rows_written // progress_every + 1) * progress_every

        writer.close()
        writer = None
        os.replace(temp_file, output_file)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_file.exists():
            temp_file.unlink()
        raise

    elapsed = time.perf_counter() - started
    return {
        "day": current_day.isoformat(),
        "file": str(output_file),
        "rows": rows_written,
        "batches": batches_written,
        "size_bytes": output_file.stat().st_size,
        "elapsed": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hive-partitioned daily Parquet event data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", default="events")
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--contents", type=int, default=DEFAULT_CONTENTS)
    parser.add_argument("--events-per-user", type=float, default=DEFAULT_EVENTS_PER_USER)
    parser.add_argument("--dau", type=float, default=DEFAULT_DAU)
    parser.add_argument("--ctr", type=float, default=DEFAULT_CTR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.dataset_name):
        parser.error("--dataset-name must contain only letters, numbers, underscores, or hyphens")
    if args.days <= 0:
        parser.error("--days must be greater than zero")
    if not 1 <= args.users <= 2**48:
        parser.error("--users must be between 1 and 2^48")
    if not 1 <= args.contents <= 2**31 - 1:
        parser.error("--contents must be between 1 and 2^31 - 1")
    if not 0 < args.events_per_user or not math.isfinite(args.events_per_user):
        parser.error("--events-per-user must be finite and positive")
    if not 0 < args.dau <= 1:
        parser.error("--dau must be in the interval (0, 1]")
    if not 0 <= args.ctr <= 1:
        parser.error("--ctr must be in the interval [0, 1]")
    if args.workers <= 0 or args.batch_size <= 0 or args.progress_every <= 0:
        parser.error("--workers, --batch-size and --progress-every must be positive")
    if args.seed < 0:
        parser.error("--seed must not be negative")
    if int(args.users * args.dau) == 0:
        parser.error("The selected --users and --dau result in zero active users")
    try:
        args.start_date + timedelta(days=args.days - 1)
    except OverflowError:
        parser.error("--start-date and --days exceed the supported date range")
    return args


def main() -> None:
    args = parse_args()
    output_dir = DATA_DIR / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    active_users = int(args.users * args.dau)
    estimated_daily_events = active_users * args.events_per_user
    heading("PyArrow / Parquet Dataset Generator (Hive partitioned)")
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
    print(f"Estimated events/day: {format_number(estimated_daily_events)}")
    print(f"Estimated total rows: {format_number(estimated_daily_events * args.days)}")

    # CHANGED: Remove old flat Parquet files and all previous Hive partitions.
    # Restrict deletion to this selected dataset directory.
    existing_files = list(output_dir.glob("*.parquet"))
    partition_dirs = [p for p in output_dir.glob("day=*") if p.is_dir()]
    if existing_files or partition_dirs:
        print(f"Removing {len(existing_files)} flat file(s) and {len(partition_dirs)} partition(s)...")
        for file in existing_files:
            file.unlink()
        for partition_dir in partition_dirs:
            shutil.rmtree(partition_dir)

    heading("Generating Parquet files")
    started = time.perf_counter()
    total_rows = 0
    total_bytes = 0
    completed_days = 0
    results = []

    with ProcessPoolExecutor(max_workers=min(args.workers, args.days)) as executor:
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
            print(f"\n✓ [{completed_days}/{args.days}] {result['day']} completed")
            print(f"  Rows:          {format_number(result['rows'])}")
            print(f"  File:          {result['file']}")
            print(f"  Size:          {result['size_bytes'] / 1024**2:.2f} MB")
            print(f"  Duration:      {format_duration(result['elapsed'])}")
            print(f"  Total elapsed: {format_duration(elapsed)}")

    elapsed = time.perf_counter() - started
    heading("Generation Complete")
    print(f"Dataset:       {args.dataset_name}")
    print(f"Days:          {completed_days}")
    print(f"Total rows:    {format_number(total_rows)}")
    print(f"Total size:    {total_bytes / 1024**3:.3f} GB")
    print(f"Elapsed:       {format_duration(elapsed)}")
    print(f"Throughput:    {format_number(total_rows / elapsed if elapsed else 0)} rows/s")
    print(f"Output:        {output_dir}")
    print("\nGenerated files:")
    for result in sorted(results, key=lambda item: item["day"]):
        print(
            f"  {Path(result['file']).relative_to(output_dir)!s:<38} "
            f"{format_number(result['rows']):>15} rows "
            f"{result['size_bytes'] / 1024**2:>10.2f} MB"
        )
    print("\nExample DuckDB query:")
    print(
        f"  SELECT day, type, COUNT(*) FROM "
        f"read_parquet('{output_dir.as_posix()}/day=*/*.parquet', "
        f"hive_partitioning = true) GROUP BY day, type;"
    )


if __name__ == "__main__":
    main()
