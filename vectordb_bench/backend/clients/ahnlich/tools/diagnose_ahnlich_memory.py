"""Run one vector case with one loading worker and sample DB/system memory."""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import grpc
import psutil

from .._grpc.db import query_pb2
from .._grpc.services.db_service_pb2_grpc import DBServiceStub
from .run_ahnlich_matrix import read_result

# ruff: noqa: T201 -- Stream diagnostic output to the terminal and run log.
DATASET = "Small Cohere (768dim, 100K)"


def memory_sample(stub: DBServiceStub, process: psutil.Process, phase: str) -> dict:
    sample = {
        "time_utc": datetime.now(UTC).isoformat(),
        "phase": phase,
        "stored_vectors": "",
        "allocator_limit_bytes": "",
        "allocator_used_bytes": "",
        "rss_bytes": "",
        "system_available_bytes": psutil.virtual_memory().available,
        "system_swap_used_bytes": psutil.swap_memory().used,
        "error": "",
    }
    try:
        sample["rss_bytes"] = process.memory_info().rss
        info = stub.InfoServer(query_pb2.InfoServer(), timeout=2).info
        sample["allocator_limit_bytes"] = info.limit
        sample["allocator_used_bytes"] = info.limit - info.remaining
        try:
            store = stub.GetStore(query_pb2.GetStore(store="vdbbench_memory", schema="public"), timeout=2)
            sample["stored_vectors"] = store.len
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.NOT_FOUND:
                raise
            sample["stored_vectors"] = 0
    except (grpc.RpcError, psutil.Error) as error:
        sample["error"] = str(error)
    return sample


def monitor_memory(stub: DBServiceStub, process: psutil.Process, root: Path, phase: list[str], stop: threading.Event):
    with (root / "memory.csv").open("w", newline="") as output:
        writer = None
        while not stop.is_set():
            sample = memory_sample(stub, process, phase[0])
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(sample))
                writer.writeheader()
            writer.writerow(sample)
            output.flush()
            if stop.wait(5):
                break


def validate_load(root: Path, server: subprocess.Popen):
    # VectorDBBench can exit zero even when the task failed. Require its success label.
    read_result(root / "load" / "json")
    if server.poll() is not None:
        raise RuntimeError("DB exited during loading; inspect server.log and memory.csv")


def terminate_process(process: subprocess.Popen):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_load(command: list[str], env: dict, root: Path, stage: str = "load"):
    """Stream one CLI stage to its own log and preserve its exit code."""
    with (
        (root / f"{stage}.txt").open("w") as load_log,
        subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as loader,
    ):
        try:
            for line in loader.stdout:
                load_log.write(line)
                load_log.flush()
                print(line, end="", flush=True)
            code = loader.wait()
            print(f"{stage} CLI exit code: {code}", flush=True)
            if code:
                message = f"{stage} exited with code {code}; inspect {stage}.txt"
                raise RuntimeError(message)
        finally:
            terminate_process(loader)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["flat", "hnsw"], required=True)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--server-ref", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--port", type=int, default=1369)
    parser.add_argument("--case-type", help="Preset case; omit for the original Small Cohere 100K diagnostic")
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--m-zero", type=int, default=30)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=180)
    parser.add_argument("--search", action="store_true", help="Run serial and concurrent search after loading")
    parser.add_argument("--num-concurrency", default="12,14,16,18,20")
    return parser.parse_args()


def main():  # noqa: PLR0915 -- Keep the diagnostic process lifecycle in one place.
    args = parse_arguments()
    binary = args.server_bin.resolve(strict=True)
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ["memory.csv", "server.log", "load"]):
        raise ValueError("Use a fresh timestamped results directory")
    endpoint = f"127.0.0.1:{args.port}"
    # Fail before startup if another DB is already bound to the selected port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    stop = threading.Event()
    phase = ["baseline"]
    status = {"state": "running", "phase": "baseline"}
    monitor = None
    server_command = [
        str(binary),
        "run",
        "--port",
        str(args.port),
        "--threadpool-size",
        "16",
        "--allocator-size",
        "60129542144",
        "--message-size",
        "67108864",
        "--log-level",
        "info",
    ]
    command = [
        sys.executable,
        "-m",
        "vectordb_bench.cli.vectordbbench",
        "ahnlich",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--server-version",
        args.server_ref,
        "--db-label",
        "memory-diagnostic",
        "--case-type",
        args.case_type or "Performance",
        *([] if args.case_type else ["--dataset-name", DATASET]),
        "--algorithm",
        args.algorithm,
        "--collection-name",
        "vdbbench_memory",
        "--implementation",
        "grpc",
        "--load-concurrency",
        "1",
        "--insert-batch-size",
        "100",
        "--m",
        str(args.m),
        "--m-zero",
        str(args.m_zero),
        "--ef-construction",
        str(args.ef_construction),
        "--ef-search",
        str(args.ef_search),
        "--k",
        "100",
        "--num-concurrency",
        args.num_concurrency,
        "--concurrency-duration",
        "30",
        "--message-size",
        "67108864",
        "--drop-old",
        "--load",
        "--skip-search-serial",
        "--skip-search-concurrent",
    ]
    search_command = [*command[:-4], "--skip-drop-old", "--skip-load", "--search-serial", "--search-concurrent"]
    commands = {"server": server_command, "load": command}
    if args.search:
        commands["search"] = search_command
    (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT)
        try:
            channel = grpc.insecure_channel(endpoint)
            try:
                grpc.channel_ready_future(channel).result(timeout=30)
                stub = DBServiceStub(channel)
                monitor = threading.Thread(
                    target=monitor_memory, args=(stub, psutil.Process(server.pid), root, phase, stop), daemon=True
                )
                monitor.start()
                stop.wait(5)
                phase[0] = "loading"
                env = dict(os.environ, RESULTS_LOCAL_DIR=str(root / "load" / "json"), LOG_FILE=str(root / "load.log"))
                run_load(command, env, root)
                validate_load(root, server)
                phase[0] = "settling"
                print("Loading passed. Sampling idle memory for 60 seconds.", flush=True)
                for _ in range(60):
                    if server.poll() is not None:
                        raise RuntimeError("DB exited while settling; inspect server.log")
                    stop.wait(1)
                if "Error inserting into index" in (root / "server.log").read_text():
                    raise RuntimeError("DB logged an index insertion error; discard the load")
                if args.search:
                    phase[0] = "searching"
                    env = dict(
                        os.environ, RESULTS_LOCAL_DIR=str(root / "search" / "json"), LOG_FILE=str(root / "search.log")
                    )
                    run_load(search_command, env, root, "search")
                    read_result(root / "search" / "json")
                    if server.poll() is not None:
                        raise RuntimeError("DB exited during search; inspect server.log and memory.csv")
                status["state"] = "completed"
                print(f"Diagnostic complete: {root / 'memory.csv'}", flush=True)
            finally:
                stop.set()
                if monitor is not None:
                    monitor.join(timeout=5)
                channel.close()
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            status.update(
                phase=phase[0],
                time_utc=datetime.now(UTC).isoformat(),
                server_returncode_before_cleanup=server.poll(),
            )
            (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            terminate_process(server)


if __name__ == "__main__":
    main()
