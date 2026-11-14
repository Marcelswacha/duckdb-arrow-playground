.DEFAULT_GOAL := help

# ------------------------------------------------------------
# Python
# ------------------------------------------------------------

UV ?= uv
PYTHON := $(UV) run python

# ------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------

DATASET_NAME ?= benchmark_1m_partitions
START_DATE ?= 2025-01-01
DAYS ?= 90
USERS ?= 1000000
CONTENTS ?= 3000000
EVENTS_PER_USER ?= 100
DAU ?= 0.75
CTR ?= 0.10
WORKERS ?= 8
BATCH_SIZE ?= 250000

# ------------------------------------------------------------
# Benchmark configuration
# ------------------------------------------------------------

QUERY ?= users_per_day
THREADS ?= 4
MEMORY_LIMIT ?= 32GB
RUNS ?= 1
WARMUP ?= 1
EXPLAIN ?= 0

EXPLAIN_FLAG := $(if $(filter 1 true yes,$(EXPLAIN)),--explain,)

.PHONY: help setup run generate benchmark benchmark-1 benchmark-4 benchmark-8 list-queries clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-22s %s\n", $$1, $$2}'

setup: ## Install dependencies using uv
	$(UV) sync

run: ## Run original DuckDB + Arrow playground
	$(PYTHON) main.py

generate: ## Generate daily Parquet files (overwrites existing dataset)
	$(PYTHON) generate_data.py \
		--dataset-name $(DATASET_NAME) \
		--start-date $(START_DATE) \
		--days $(DAYS) \
		--users $(USERS) \
		--contents $(CONTENTS) \
		--events-per-user $(EVENTS_PER_USER) \
		--dau $(DAU) \
		--ctr $(CTR) \
		--workers $(WORKERS) \
		--batch-size $(BATCH_SIZE)

benchmark: ## Run selected SQL query (QUERY=all by default)
	$(PYTHON) benchmark.py \
		--dataset-name $(DATASET_NAME) \
		--query $(QUERY) \
		--threads $(THREADS) \
		--memory-limit $(MEMORY_LIMIT) \
		--runs $(RUNS) \
		--warmup $(WARMUP) \
		$(EXPLAIN_FLAG)

benchmark-1: ## Benchmark using 1 thread
	$(MAKE) benchmark THREADS=1

benchmark-4: ## Benchmark using 4 threads
	$(MAKE) benchmark THREADS=4

benchmark-8: ## Benchmark using 8 threads
	$(MAKE) benchmark THREADS=8

list-queries: ## List available SQL queries
	$(PYTHON) benchmark.py --list

clean: ## Remove generated data
	rm -rf data/