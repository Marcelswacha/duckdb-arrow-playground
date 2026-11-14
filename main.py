"""A one-file playground for PyArrow, Parquet, and DuckDB.

Run with: uv run python main.py
All output files are written under ./data (relative to this script).
"""

from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ORDERS_FILE = DATA_DIR / "orders.parquet"
SUMMARY_FILE = DATA_DIR / "customer_summary.parquet"


def heading(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Create an Arrow table
    # ============================================================
    heading("1. Create an Arrow table")

    orders = pa.table(
        {
            "order_id": pa.array(
                [1, 2, 3, 4, 5, 6, 7, 8],
                type=pa.int64(),
            ),
            "ordered_at": pa.array(
                [
                    date(2026, 1, 5),
                    date(2026, 1, 6),
                    date(2026, 2, 2),
                    date(2026, 2, 4),
                    date(2026, 2, 8),
                    date(2026, 3, 1),
                    date(2026, 3, 3),
                    date(2026, 3, 5),
                ],
                type=pa.date32(),
            ),
            "customer": [
                "Alice", "Bob", "Alice", "Charlie",
                "Bob", "Alice", "Charlie", "Bob",
            ],
            "product": [
                "Laptop", "Mouse", "Keyboard", "Laptop",
                "Monitor", "Mouse", "Keyboard", "Laptop",
            ],
            "quantity": pa.array(
                [1, 2, 1, 1, 2, 3, 2, 1],
                type=pa.int32(),
            ),
            "unit_price": pa.array(
                [1200.0, 25.0, 80.0, 1100.0, 300.0, 25.0, 80.0, 1250.0],
                type=pa.float64(),
            ),
        }
    )

    print(orders.schema)
    print(f"Rows: {orders.num_rows}")

    # ============================================================
    # 2. Write Parquet using PyArrow
    # ============================================================
    heading("2. Write Parquet with PyArrow")

    pq.write_table(
        orders,
        ORDERS_FILE,
        compression="zstd",
    )

    print(
        f"Created: {ORDERS_FILE.relative_to(ROOT)} "
        f"({ORDERS_FILE.stat().st_size} bytes)"
    )

    metadata = pq.read_metadata(ORDERS_FILE)

    print(
        f"Parquet metadata: {metadata.num_rows} rows, "
        f"{metadata.num_row_groups} row group(s)"
    )

    # ============================================================
    # 3. Connect to DuckDB
    # ============================================================
    with duckdb.connect(":memory:") as con:

        # ========================================================
        # 3. Query Parquet directly
        # ========================================================
        heading("3. SQL directly on a Parquet file")

        con.sql(
            """
            SELECT
                customer,
                COUNT(*) AS orders,
                ROUND(SUM(quantity * unit_price), 2) AS total_spent
            FROM read_parquet(?)
            GROUP BY customer
            ORDER BY total_spent DESC
            """,
            params=[str(ORDERS_FILE)],
        ).show()

        # ========================================================
        # 4. Query Arrow directly
        # ========================================================
        heading("4. SQL directly on Arrow in memory")

        con.register("orders_arrow", orders)

        con.sql(
            """
            SELECT
                product,
                SUM(quantity) AS units_sold,
                ROUND(SUM(quantity * unit_price), 2) AS revenue
            FROM orders_arrow
            GROUP BY product
            ORDER BY revenue DESC
            """
        ).show()

        # ========================================================
        # 5. DuckDB -> Arrow
        # ========================================================
        heading("5. DuckDB query result -> Arrow")

        arrow_result = con.sql(
            """
            SELECT
                customer,
                SUM(quantity * unit_price) AS revenue
            FROM orders_arrow
            GROUP BY customer
            ORDER BY revenue DESC
            """
        ).to_arrow_table()

        print(f"Result type: {type(arrow_result).__name__}")
        print(f"Schema:\n{arrow_result.schema}")
        print(f"Data:\n{arrow_result.to_pydict()}")

        # ========================================================
        # 6. DuckDB -> Parquet
        # ========================================================
        heading("6. DuckDB SQL -> Parquet")

        con.execute(
            """
            COPY (
                SELECT
                    customer,
                    COUNT(*) AS orders,
                    SUM(quantity * unit_price) AS total_spent
                FROM orders_arrow
                GROUP BY customer
            )
            TO ?
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(SUMMARY_FILE)],
        )

        print(f"Created: {SUMMARY_FILE.relative_to(ROOT)}")

        # ========================================================
        # 7. Read generated Parquet
        # ========================================================
        heading("7. Read generated Parquet")

        con.sql(
            """
            SELECT *
            FROM read_parquet(?)
            ORDER BY total_spent DESC
            """,
            params=[str(SUMMARY_FILE)],
        ).show()

        # ========================================================
        # 8. EXPLAIN
        # ========================================================
        heading("8. EXPLAIN - Parquet scan and projection pushdown")

        plan = con.execute(
            """
            EXPLAIN
            SELECT
                customer,
                unit_price
            FROM read_parquet(?)
            WHERE unit_price > 1000
            """,
            [str(ORDERS_FILE)],
        ).fetchone()

        print(plan[1])

    heading("Done!")
    print("Generated files are under ./data; `make clean` deletes them.")


if __name__ == "__main__":
    main()