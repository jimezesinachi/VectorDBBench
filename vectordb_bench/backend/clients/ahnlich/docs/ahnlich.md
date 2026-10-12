# Ahnlich

The adapter benchmarks dataset loading and unfiltered dense-vector search using
HNSW or exact search. It offers direct synchronous gRPC and Python SDK
implementations; no AI server is required. Connect to a standalone DB with unauthenticated, plaintext
gRPC on localhost or a trusted network.

## Install

From the VectorDBBench checkout, using Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ahnlich]'
```

## Run locally

Building a managed server requires Rust/Cargo and an Ahnlich source checkout.
Set `AHNLICH_ROOT` to that checkout's root. A prebuilt binary or an already
running DB can be used instead, as described below.

The runner builds a release DB from your Ahnlich checkout, starts it, and stops
it after benchmarking. By default it selects all registered datasets with both
HNSW and linear/exact search, at concurrency 1, 2, 4, 8, and 16. The current
catalog has 36 datasets: 72 dataset/algorithm runs and 360 throughput measurements.
Preset cases and registered names referring to the same dataset are counted once.

Each dataset keeps its original vector count, dimensions, distance metric, queries,
and ground truth. Each dataset/algorithm pair loads a fresh store once, then runs
its selected concurrency phases for 30 seconds each and measures serial latency
and recall. Loading uses one worker and batches of 100; searches request 100
neighbors. Initial downloads require internet access and are cached for later runs.

For a bounded local run, include datasets with at most 100,000 vectors and at most
1,536 dimensions. Both limits are inclusive; this selects Small Cohere and Small
OpenAI, producing four dataset/algorithm runs and 20 throughput measurements:

```bash
RESULTS_DIR="$PWD/.data/results/local-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
set -o pipefail
AHNLICH_ROOT=/absolute/path/to/ahnlich RESULTS_DIR="$RESULTS_DIR" \
  bash vectordb_bench/backend/clients/ahnlich/tools/run_ahnlich.sh --max-vectors 100000 --max-dimensions 1536 \
  2>&1 | tee "$RESULTS_DIR/run.txt"
```

Add `--dry-run` to preview the selection without building or starting a server,
downloading data, or running benchmarks. Omit both size limits for the full matrix.

To skip building, replace `AHNLICH_ROOT` with `SERVER_BIN=/absolute/path/to/ahnlich-db`
and `SERVER_REF=YOUR_DB_REVISION`. Defaults are port 1369, four Rayon threads, an
8 GiB allocator cap, and 64 MiB messages. The generator and DB need additional RAM
beyond that cap. Adjust `THREADS` and `ALLOCATOR_SIZE` for the machine.

## Run on a provisioned machine

Install the same checkout and dependencies on the provisioned machine. Use the
local command above to build and run there, or use `SERVER_BIN` for a copied
release binary built for that machine's operating system and architecture.

For an already running DB, omit `AHNLICH_ROOT` and `SERVER_BIN` and set its endpoint:

```bash
RESULTS_DIR="$PWD/.data/results/provisioned-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
set -o pipefail
HOST=YOUR_DB_HOST PORT=1369 SERVER_REF=YOUR_DB_REVISION RESULTS_DIR="$RESULTS_DIR" \
  bash vectordb_bench/backend/clients/ahnlich/tools/run_ahnlich.sh \
  2>&1 | tee "$RESULTS_DIR/run.txt"
```

The existing server remains running. Record its resources and settings, and
whether the generator shares its machine, when comparing results. Check its logs
for index errors; the runner can check server logs automatically only for servers
it starts.

## Configure

The shared workload defaults apply to every selected dataset/algorithm pair:

| Property | Default |
|---|---|
| Results per query | 100 (`k`) |
| Duration | 30 seconds per concurrency level |
| Loading workers | 1 |
| Insert batch size | 100 |
| Filters | None |
| Accuracy evaluation | Dataset-supplied queries and matching ground truth |
| HNSW construction | `m=16`, `m_zero=32`, `ef_construction=200`; both optional construction flags disabled |
| HNSW search | `ef_search=100`; override with `--ef-search N` (server raises it to at least `k`) |
| Linear search | Exact search using the dataset's metric; no HNSW index |

Select matrix workloads through runner options:

| Option | Selection |
|---|---|
| `--algorithm hnsw` or `--algorithm flat` | One algorithm across all matching datasets; default is both |
| `--dataset NAME` | One registered dataset by its exact catalog name |
| `--case-type PRESET` | The dataset associated with one fixed unfiltered performance preset |
| `--concurrency N` | One positive concurrency level, including values above 16; default is 1, 2, 4, 8, 16 |
| `--max-vectors N` | Datasets with at most N vectors |
| `--max-dimensions N` | Datasets with at most N dimensions |
| `--dry-run` | Preview selected workloads without executing them |

Dataset and preset selections are alternatives. All other filters combine. Empty
selections and option-parsing errors fail before the server is built or started.
Checks inside the adapter command, such as rejecting `--nq 2`, run when the
benchmark command executes, after managed-server startup.
For example, adding `--case-type Performance1536D50K --algorithm flat --concurrency 32`
to the local command selects one exact-cosine run at concurrency 32. Selecting only
an algorithm runs all matching datasets with that algorithm.

Edit [the example YAML](../examples/ahnlich.yaml), or select
another file with `CONFIG_FILE`, for shared workload, store, and schema settings.
The matrix runner overrides the YAML's case, algorithm, and search concurrency
with its selection. The wrapper also supplies the host, port, message-size limit,
and server revision; configure those with `HOST`, `PORT`, `MESSAGE_SIZE`, and
`SERVER_REF` environment variables rather than the YAML. When building the DB,
`SERVER_REF` is populated automatically from the Ahnlich checkout. Other adapter
CLI options, such as `--k` and `--concurrency-duration`, are forwarded to every
selected run. Use
`bash vectordb_bench/backend/clients/ahnlich/tools/run_ahnlich.sh --help` for matrix options. The standard
`vectordbbench ahnlich` CLI still runs one case and algorithm per invocation.

- Loading drops the configured store by default. Use a dedicated benchmark store.
- Loading uses one worker until Ahnlich's concurrent HNSW insertion race is fixed
  ([#425](https://github.com/deven96/ahnlich/issues/425)). Search concurrency is independent.
- For search-only runs against an existing populated server, select exactly one
  dataset/algorithm pair and add `--skip-load --skip-drop-old`. Keep the dataset,
  store, dimension, and index settings unchanged.
- Managed servers have persistence disabled and are stopped after the run, so
  their in-memory stores cannot be reused in a later run.

## Results and limitations

Use a fresh timestamped results directory for each invocation. The runner saves:

- `plan.json`: selected datasets, algorithms, dimensions, metrics, and concurrency.
- `config.yaml` and `environment.txt`: shared configuration and environment notes.
- One named directory per dataset/algorithm pair, containing `benchmark.txt`,
  `vectordbbench.log`, the actual command in `command.json`, and result JSON under `json/`.
- `results.json`: passed, failed, and pending runs, with metrics for completed runs.
- Managed runs also save build and server logs in the parent directory.

The runner stops at the first failed task or logged managed-server index error,
returns a nonzero status, and leaves later runs marked pending. Existing servers
remain running; managed servers are stopped on success or failure. Loading time
is single-writer loading time, not maximum concurrent ingestion throughput.

Failed insert RPCs are not retried because their commit status may be uncertain.

Index construction runs inside Set. The adapter checks for an empty or oversized store and performs a
search probe after loading, but the DB has no full index-coverage endpoint and
can swallow index errors. Inspect recall and server logs as well as QPS.

The adapter passes `ef_search=100` by default for HNSW; `--ef-search N` overrides
it and has no effect on linear search. The DB must include the configurable search
breadth change ([#436](https://github.com/deven96/ahnlich/pull/436)). When the field
is omitted by other clients, the server defaults to `max(k, 50)`. Although the adapter returns only IDs to VectorDBBench, Ahnlich's wire
response includes vectors and metadata, and that transfer cost is measured.
Capacity cases replay identical vectors under new IDs, which become updates in
Ahnlich and cannot measure its capacity correctly. The CLI lists only supported
unfiltered performance cases in `--case-type`.
Filters, streaming writes, full-text search, tenants, alternative payloads, and
insert-readiness scenarios are outside this adapter's scope.

## Compare direct gRPC and the Python SDK

The adapter defaults to the current synchronous gRPC implementation. Select
`--implementation sdk` to use `ahnlich-client-py` through a persistent event loop
and channel per worker. The selector belongs to `vectordbbench ahnlich`; the shell
runner forwards it unchanged. Result configurations record the implementation.

The SDK candidate remains experimental. In the October 6, 2026 local comparison
on the 50K-vector, 1536-dimensional OpenAI dataset, its peak HNSW QPS was about
59 times lower and peak linear QPS about 3.1 times lower than direct gRPC, with
identical recall. These are results from one local comparison, not a general
performance guarantee. The cause remains under investigation; direct gRPC stays
the default.

For this comparison, install the SDK from the same Ahnlich checkout used by the
server (`python -m pip install /absolute/path/to/ahnlich/sdk/ahnlich-client-py`).
The SDK must include the new `GetSimN.ef_search` field; older installed versions
are rejected before connecting.
Run each implementation separately against the same dataset, algorithms, and
concurrency settings, using fresh timestamped result directories. For example,
append `--case-type Performance1536D50K --implementation grpc` for the control,
and replace `grpc` with `sdk` for the candidate. This runs both HNSW and linear
search at concurrency 1, 2, 4, 8, and 16. Each run loads fresh stores.

## Native upserts and recall

Ahnlich uses float32 vectors as primary keys. Identical vectors with different
dataset IDs are processed as native upserts: one stored entry remains, and its
benchmark ID metadata is overwritten. This applies within batches and across
batches, for both HNSW and linear search. The adapter accepts these upserts and
reports processed rows to the loader, so datasets containing duplicates can run.
The stored corpus can therefore contain fewer entries than the original dataset.
QPS measures searches over that resulting corpus; loading time includes inserts
and updates. No additional upsert counters are recorded.

Recall is calculated against the dataset's original, unchanged ground truth.
Overwritten IDs can still appear among its expected neighbors, lowering recall
even for exact linear search. HNSW recall can reflect both approximate search
and IDs lost through upserts. The linear result on the same dataset provides a
reference, but does not isolate or correct either contribution. These recall
values describe Ahnlich's native storage behavior against the original dataset.

## Validate and regenerate bindings

Correctness tests use a small gRPC fixture, with optional four-vector tests
against a real DB. They do not run performance benchmarks.

```bash
python -m pip install -e '.[ahnlich,test]'
mkdir -p .data/validation
set -o pipefail
IR_DATASETS_HOME="$PWD/.data/ir_datasets" MPLCONFIGDIR="$PWD/.data/matplotlib" \
  python -m pytest tests/test_ahnlich.py vectordb_bench/backend/clients/ahnlich/tests/test_ahnlich_runner.py -q \
  2>&1 | tee .data/validation/tests.txt
```

Set `AHNLICH_DB_BIN=/absolute/path/to/ahnlich-db` to enable the real DB tests.
Generated bindings are checked in; protoc is needed only when updating them:

```bash
python -m pip install 'grpcio-tools==1.67.1' 'protobuf==5.29.5'
python vectordb_bench/backend/clients/ahnlich/tools/generate_ahnlich_bindings.py --ahnlich-root /absolute/path/to/ahnlich
```

The generator reads the DB service's transitive protobuf dependencies from the
Ahnlich checkout, qualifies imports within the adapter, and records source hashes
and revision in `proto_manifest.json`. Keep the Ahnlich MIT license alongside the
bindings.
