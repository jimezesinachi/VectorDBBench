"""Select and run baseline Ahnlich workloads using VectorDBBench's dataset catalog."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# ruff: noqa: T201 -- Command-line plans and streamed benchmark output go to stdout.
from vectordb_bench.backend.cases import type2case
from vectordb_bench.backend.clients.ahnlich.cli import Ahnlich
from vectordb_bench.backend.dataset import REGISTERED_DATASETS, BaseDataset, DatasetManager

DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def dataset_key(data: BaseDataset) -> tuple:
    return data.name, data.size, data.dim, data.metric_type.value


def preset_datasets() -> dict[str, str]:
    """Map fixed unfiltered presets to the registry, without preparing datasets."""
    names = {dataset_key(manager.data): name for name, manager in REGISTERED_DATASETS.items()}
    presets = {}
    for case, case_class in type2case.items():
        field = case_class.model_fields.get("dataset")
        if (
            case.name.startswith("Performance")
            and case_class.model_fields["filter_rate"].default is None
            and field is not None
            and isinstance(field.default, DatasetManager)
        ):
            presets[case.name] = names[dataset_key(field.default.data)]
    return presets


def select_runs(
    dataset: str | None = None,
    case_type: str | None = None,
    algorithm: str | None = None,
    concurrency: int | None = None,
    max_vectors: int | None = None,
    max_dimensions: int | None = None,
) -> list[dict]:
    presets = preset_datasets()
    selected = dataset or (presets[case_type] if case_type else None)
    case_names = {name: case for case, name in presets.items()}
    runs = []
    for name, manager in REGISTERED_DATASETS.items():
        data = manager.data
        if selected is not None and name != selected:
            continue
        if max_vectors is not None and data.size > max_vectors:
            continue
        if max_dimensions is not None and data.dim > max_dimensions:
            continue
        for method in [algorithm] if algorithm else ["hnsw", "flat"]:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            runs.append(
                {
                    "id": f"{slug}-{method}",
                    "dataset": name,
                    "case_type": case_names.get(name, "Performance"),
                    "vectors": data.size,
                    "dimensions": data.dim,
                    "metric": data.metric_type.value,
                    "algorithm": method,
                    "concurrency": [concurrency] if concurrency is not None else DEFAULT_CONCURRENCY,
                }
            )
    if not runs:
        raise ValueError("No datasets match the selection and size limits")
    return runs


def benchmark_arguments(run: dict, config_file: Path, environment_file: Path, extra: list[str]) -> list[str]:
    arguments = [
        "--config-file",
        str(config_file),
        "--host",
        os.environ.get("HOST", "127.0.0.1"),
        "--port",
        os.environ.get("PORT", "1369"),
        "--message-size",
        os.environ.get("MESSAGE_SIZE", "67108864"),
        "--server-version",
        os.environ.get("SERVER_REF", "unknown"),
        *(
            ["--note-file", str(environment_file)]
            if environment_file.exists()
            else ["--note", "Matrix selection preview"]
        ),
        "--task-label",
        run["id"],
        *extra,
        "--case-type",
        run["case_type"],
        "--algorithm",
        run["algorithm"],
        "--num-concurrency",
        ",".join(map(str, run["concurrency"])),
    ]
    if run["case_type"] == "Performance":
        arguments.extend(["--dataset-name", run["dataset"]])
    return arguments


def read_result(directory: Path) -> dict:
    files = list(directory.glob("**/result_*.json"))
    if len(files) != 1:
        message = f"Expected one result JSON, found {len(files)} in {directory}"
        raise RuntimeError(message)
    results = json.loads(files[0].read_text()).get("results", [])
    if len(results) != 1 or results[0].get("label") != ":)":
        message = f"Benchmark task failed; inspect {files[0]}"
        raise RuntimeError(message)
    return results[0]


def run_matrix(plan_file: Path):
    plan = json.loads(plan_file.read_text())
    root = plan_file.parent
    summary = [{**run, "status": "pending"} for run in plan["runs"]]
    summary_file = root / "results.json"
    for result in summary:
        directory = root / result["id"]
        directory.mkdir()
        env = dict(os.environ, RESULTS_LOCAL_DIR=str(directory / "json"), LOG_FILE=str(directory / "vectordbbench.log"))
        arguments = benchmark_arguments(result, root / "config.yaml", root / "environment.txt", plan["extra_arguments"])
        command = [sys.executable, "-m", "vectordb_bench.cli.vectordbbench", "ahnlich", *arguments]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(
            f"\nRunning {result['id']}: {result['vectors']:,} vectors, {result['dimensions']} dimensions, "
            f"{result['metric']}, concurrency {result['concurrency']}",
            flush=True,
        )
        try:
            with (
                (directory / "benchmark.txt").open("w") as log,
                subprocess.Popen(
                    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                ) as process,
            ):
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                if process.wait():
                    message = f"Benchmark exited with status {process.returncode}; inspect {directory}"
                    raise RuntimeError(message)  # noqa: TRY301 -- Record the failed run before aborting.
            outcome = read_result(directory / "json")
            server_log = os.environ.get("SERVER_LOG")
            if server_log and re.search(
                r"Error inserting into index|Error deleting from index|Neighbor not found",
                Path(server_log).read_text(),
            ):
                message = f"Index errors were logged; discard results and inspect {server_log}"
                raise RuntimeError(message)  # noqa: TRY301 -- Record the failed run before aborting.
            result.update(status="passed", metrics=outcome["metrics"])
        except Exception as error:
            result.update(status="failed", error=str(error))
            raise
        finally:
            summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nResults saved to {root}")


def main():
    parser = argparse.ArgumentParser(
        prog="bash vectordb_bench/backend/clients/ahnlich/tools/run_ahnlich.sh", description=__doc__
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-file", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--run-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config-file", type=Path, help=argparse.SUPPRESS)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", "--dataset-name", choices=list(REGISTERED_DATASETS), metavar="NAME")
    selection.add_argument("--case-type", choices=list(preset_datasets()), metavar="PRESET")
    parser.add_argument("--algorithm", choices=["hnsw", "flat"], help="Default: both algorithms")
    parser.add_argument(
        "--concurrency", type=positive_int, help="Default: 1, 2, 4, 8, 16; any positive value is allowed"
    )
    parser.add_argument("--max-vectors", type=positive_int, help="Inclusive maximum dataset size")
    parser.add_argument("--max-dimensions", type=positive_int, help="Inclusive maximum vector dimension")
    parser.add_argument("--dry-run", action="store_true", help="List selected runs without building or benchmarking")
    args, extra = parser.parse_known_args()
    if "--num-concurrency" in extra:
        parser.error("Use --concurrency to select a level, or omit it for the default sweep")
    if args.run_plan:
        run_matrix(args.run_plan)
        return
    try:
        runs = select_runs(
            args.dataset, args.case_type, args.algorithm, args.concurrency, args.max_vectors, args.max_dimensions
        )
    except ValueError as error:
        parser.error(str(error))
    config = yaml.safe_load(args.config_file.read_text())["ahnlich"]
    if len(runs) > 1 and (
        "--skip-load" in extra
        or "--skip-drop-old" in extra
        or not config.get("load", True)
        or not config.get("drop_old", True)
    ):
        parser.error("Multiple dataset/algorithm runs require fresh loading; select one pair for search-only runs")
    # Validate forwarded options with the adapter CLI without invoking any workload.
    try:
        for run in runs:
            with Ahnlich.make_context(
                "ahnlich", benchmark_arguments(run, args.config_file, args.plan_file.parent / "environment.txt", extra)
            ):
                pass
    except Exception as error:
        parser.error(str(error))
    plan = {"runs": runs, "extra_arguments": extra, "dry_run": args.dry_run}
    args.plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"Selected {len(runs)} dataset/algorithm runs and "
        f"{sum(len(run['concurrency']) for run in runs)} throughput measurements"
    )
    for run in runs:
        print(
            f"  {run['id']}: {run['case_type']}; {run['metric']}; {run['vectors']:,} vectors; "
            f"{run['dimensions']} dimensions; concurrency={run['concurrency']}"
        )


if __name__ == "__main__":
    main()
