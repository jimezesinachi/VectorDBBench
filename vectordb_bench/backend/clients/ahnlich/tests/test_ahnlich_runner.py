"""Matrix selection and result handling without executing performance workloads."""

# ruff: noqa: ANN001

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vectordb_bench.backend.clients.ahnlich.tools import run_ahnlich_matrix as matrix

ROOT = Path(__file__).resolve().parents[5]
BASH = shutil.which("bash")


def test_full_matrix_has_no_preset_duplicates():
    runs = matrix.select_runs()
    assert len(runs) == 72
    assert len({run["id"] for run in runs}) == 72
    assert sum(len(run["concurrency"]) for run in runs) == 360
    assert {run["metric"] for run in runs} == {"L2", "COSINE", "IP"}
    assert all(run["concurrency"] == [1, 2, 4, 8, 16] for run in runs)


def test_inclusive_local_limits():
    runs = matrix.select_runs(max_vectors=100_000, max_dimensions=1536)
    assert len(runs) == 4
    assert {run["dataset"] for run in runs} == {
        "Small Cohere (768dim, 100K)",
        "Small OpenAI (1536dim, 50K)",
    }
    assert len(matrix.select_runs(max_vectors=100_000, max_dimensions=768)) == 2
    with pytest.raises(ValueError, match="No datasets match"):
        matrix.select_runs(max_vectors=49_999)


def test_algorithm_preset_and_high_concurrency_selection():
    assert len(matrix.select_runs(algorithm="flat")) == 36
    preset = matrix.select_runs(case_type="Performance768D1M", algorithm="hnsw", concurrency=64)
    registered = matrix.select_runs(dataset="Medium Cohere (768dim, 1M)", algorithm="hnsw", concurrency=64)
    assert preset == registered
    assert len(preset) == 1
    assert preset[0]["concurrency"] == [64]


def test_arguments_use_matching_case_and_dataset(tmp_path):
    runs = matrix.select_runs(max_vectors=100_000)
    for run in runs:
        arguments = matrix.benchmark_arguments(run, tmp_path / "config.yaml", tmp_path / "environment.txt", [])
        assert arguments[arguments.index("--algorithm") + 1] == run["algorithm"]
        assert arguments[arguments.index("--case-type") + 1] == run["case_type"]
        assert arguments[arguments.index("--num-concurrency") + 1] == "1,2,4,8,16"
        if run["case_type"] == "Performance":
            assert arguments[arguments.index("--dataset-name") + 1] == run["dataset"]
        else:
            assert "--dataset-name" not in arguments


@pytest.mark.skipif(BASH is None, reason="The shell runner requires bash")
def test_shell_preview_does_not_build_or_start_server(tmp_path):
    env = dict(os.environ, PYTHON=sys.executable, RESULTS_DIR=str(tmp_path), AHNLICH_ROOT="/not-a-checkout")
    completed = subprocess.run(
        [
            BASH,
            "vectordb_bench/backend/clients/ahnlich/tools/run_ahnlich.sh",
            "--dry-run",
            "--max-vectors",
            "100000",
            "--max-dimensions",
            "1536",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "4 dataset/algorithm runs and 20 throughput measurements" in completed.stdout
    assert not (tmp_path / "server.log").exists()
    assert not (tmp_path / "build.txt").exists()
    assert len(json.loads((tmp_path / "plan.json").read_text())["runs"]) == 4


@pytest.mark.skipif(BASH is None, reason="The shell runner requires bash")
@pytest.mark.parametrize("options", [["--concurrency", "0"], ["--max-vectors", "49999"], ["--skip-load"]])
def test_invalid_plan_fails_before_execution(tmp_path, options):
    completed = subprocess.run(
        [BASH, "vectordb_bench/backend/clients/ahnlich/tools/run_ahnlich.sh", "--dry-run", *options],
        cwd=ROOT,
        env=dict(os.environ, PYTHON=sys.executable, RESULTS_DIR=str(tmp_path)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not (tmp_path / "plan.json").exists()


@pytest.mark.parametrize("label", [":)", "x"])
def test_matrix_records_success_or_failure_and_stops(tmp_path, monkeypatch, label):
    plan = {"runs": matrix.select_runs(max_vectors=100_000), "extra_arguments": []}
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))
    calls = []

    class FixtureProcess:
        def __init__(self, command, env, **kwargs):
            calls.append(command)
            self.stdout = io.StringIO("fixture result, no workload executed\n")
            self.returncode = 0
            directory = Path(env["RESULTS_LOCAL_DIR"])
            directory.mkdir()
            (directory / "result_fixture.json").write_text(
                json.dumps(
                    {
                        "results": [{"label": label, "metrics": {"qps": 1}}],
                    }
                )
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return self.returncode

    monkeypatch.setattr(matrix.subprocess, "Popen", FixtureProcess)
    monkeypatch.delenv("SERVER_LOG", raising=False)
    if label == ":)":
        matrix.run_matrix(plan_file)
        assert len(calls) == 4
        assert all(run["status"] == "passed" for run in json.loads((tmp_path / "results.json").read_text()))
    else:
        with pytest.raises(RuntimeError, match="Benchmark task failed"):
            matrix.run_matrix(plan_file)
        assert len(calls) == 1
        assert [run["status"] for run in json.loads((tmp_path / "results.json").read_text())] == [
            "failed",
            "pending",
            "pending",
            "pending",
        ]


@pytest.mark.skipif(BASH is None, reason="The shell runner requires bash")
@pytest.mark.parametrize("implementation", ["grpc", "sdk"])
def test_shell_forwards_implementation_to_standard_cli(tmp_path, implementation):
    subprocess.run(
        [
            BASH,
            "vectordb_bench/backend/clients/ahnlich/tools/run_ahnlich.sh",
            "--dry-run",
            "--case-type",
            "Performance1536D50K",
            "--implementation",
            implementation,
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTHON=sys.executable, RESULTS_DIR=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads((tmp_path / "plan.json").read_text())
    arguments = matrix.benchmark_arguments(
        plan["runs"][0], tmp_path / "config.yaml", tmp_path / "environment.txt", plan["extra_arguments"]
    )
    assert arguments[arguments.index("--implementation") + 1] == implementation
