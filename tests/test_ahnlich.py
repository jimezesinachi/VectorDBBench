"""Adapter correctness through real sync gRPC serialization, without timing workloads."""

# ruff: noqa: ANN001

import os
import pickle
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import numpy as np
import pytest
from click.testing import CliRunner

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.ahnlich._grpc import keyval_pb2, similarity_pb2
from vectordb_bench.backend.clients.ahnlich._grpc.algorithm import algorithm_pb2
from vectordb_bench.backend.clients.ahnlich._grpc.db import server_pb2
from vectordb_bench.backend.clients.ahnlich._grpc.services import db_service_pb2_grpc
from vectordb_bench.backend.clients.ahnlich.ahnlich import Ahnlich
from vectordb_bench.backend.clients.ahnlich.cli import Ahnlich as AhnlichCli
from vectordb_bench.backend.clients.ahnlich.config import AhnlichConfig, AhnlichIndexConfig
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.filter import IntFilter
from vectordb_bench.backend.payload import PayloadProfile

VECTORS = [[1, 0.1, 0], [0.8, 0.3, 0.1], [0, 1, 0], [-1, 0, 0]]
IDS = [19, 7, 103, 42]
QUERY = [1, 0, 0]


class FixtureService(db_service_pb2_grpc.DBServiceServicer):
    """Small protocol fixture; it does not stand in for production HNSW validation."""

    def __init__(self):
        self.store = None
        self.entries = {}
        self.fail_insert = False
        self.missing_id = False
        self.request_count = 0

    def GetStore(self, request, context):
        if self.store is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "Store not found")
        return server_pb2.StoreInfo(
            name=request.store,
            dimension=self.store.dimension,
            len=len(self.entries),
            non_linear_indices=self.store.non_linear_indices,
        )

    def DropStore(self, request, context):
        self.store = None
        self.entries = {}
        return server_pb2.Del()

    def CreateStore(self, request, context):
        self.store = request
        return server_pb2.Unit()

    def Set(self, request, context):
        self.request_count += 1
        if self.fail_insert:
            context.abort(grpc.StatusCode.INTERNAL, "Injected insertion failure")
        inserted = updated = 0
        for entry in request.inputs:
            key = tuple(entry.key.key)
            if key in self.entries:
                updated += 1
            else:
                inserted += 1
            self.entries[key] = entry
        response = server_pb2.Set()
        response.upsert.inserted = inserted
        response.upsert.updated = updated
        return response

    def GetSimN(self, request, context):
        self.last_search = request
        metric = request.algorithm
        if metric == algorithm_pb2.HNSW:
            metric = {
                algorithm_pb2.Euclidean: algorithm_pb2.EuclideanDistance,
                algorithm_pb2.Cosine: algorithm_pb2.CosineSimilarity,
                algorithm_pb2.DotProduct: algorithm_pb2.DotProductSimilarity,
            }[self.store.non_linear_indices[0].hnsw.distance]
        query = np.asarray(request.search_input.key)
        ranked = []
        for entry in self.entries.values():
            vector = np.asarray(entry.key.key)
            if metric == algorithm_pb2.EuclideanDistance:
                score = -np.linalg.norm(vector - query)
            elif metric == algorithm_pb2.CosineSimilarity:
                score = np.dot(vector, query) / (np.linalg.norm(vector) * np.linalg.norm(query))
            else:
                score = np.dot(vector, query)
            ranked.append((float(score), entry))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return server_pb2.GetSimN(
            entries=[
                server_pb2.GetSimNEntry(
                    key=entry.key,
                    value=keyval_pb2.StoreValue() if self.missing_id else entry.value,
                    similarity=similarity_pb2.Similarity(value=score),
                )
                for score, entry in ranked[: request.closest_n]
            ]
        )


@pytest.fixture
def service():
    fixture = FixtureService()
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    db_service_pb2_grpc.add_DBServiceServicer_to_server(fixture, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield fixture, port
    finally:
        server.stop(0).wait()


def make_client(port: int, metric=MetricType.COSINE, algorithm="hnsw", **overrides):
    return Ahnlich(
        dim=3,
        db_config={"host": "127.0.0.1", "port": port, "rpc_timeout": 10, **overrides},
        db_case_config=AhnlichIndexConfig(metric_type=metric, algorithm=algorithm),
        drop_old=True,
    )


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE, MetricType.IP, MetricType.DP])
@pytest.mark.parametrize("algorithm", ["hnsw", "flat"])
def test_wire_roundtrip_ranked_ids(service, metric, algorithm):
    fixture, port = service
    client = make_client(port, metric, algorithm)
    with client.init():
        assert client.insert_embeddings(np.asarray(VECTORS), np.asarray(IDS)) == (4, None)
        assert client.search_embedding(QUERY, k=4) == IDS
        client.optimize(data_size=4)
    assert client._stub is None
    assert fixture.request_count == 1
    restored = pickle.loads(pickle.dumps(client))  # noqa: S301 -- locally created worker state
    with restored.init():
        assert restored.search_embedding(QUERY, k=1) == [19]


def test_shared_channel_concurrent_search(service):
    _, port = service
    client = make_client(port)
    with client.init():
        assert client.insert_embeddings(VECTORS, IDS) == (4, None)
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: client.search_embedding(QUERY, k=2), range(8)))
        assert results == [[19, 7]] * 8


@pytest.mark.parametrize(
    ("vectors", "ids"),
    [
        ([[1, 0, 0], [0, 1, 0]], [1, 1]),
        ([[1, 0, 0]], []),
        ([[float("nan"), 1, 0]], [1]),
        ([[3.5e38, 1, 0]], [1]),  # Finite Python float overflows float32.
        ([[0, 0, 0]], [1]),
        ([[1, 0]], [1]),
    ],
)
def test_invalid_batch_rejected_before_rpc(service, vectors, ids):
    fixture, port = service
    client = make_client(port)
    with client.init():
        count, error = client.insert_embeddings(vectors, ids)
    assert count == 0
    assert error.non_retryable
    assert fixture.request_count == 0


def test_duplicate_across_batches_uses_native_upsert(service):
    _, port = service
    client = make_client(port)
    with client.init():
        assert client.insert_embeddings([VECTORS[0]], [19]) == (1, None)
        assert client.insert_embeddings([VECTORS[0], VECTORS[1]], [20, 7]) == (2, None)
        client.optimize(data_size=3)
        assert client.search_embedding(QUERY, k=2) == [20, 7]


def test_float32_duplicate_within_batch_uses_native_upsert(service):
    fixture, port = service
    client = make_client(port)
    with client.init():
        assert client.insert_embeddings([[1, 0, 0], [1 + 1e-9, 0, 0]], [1, 2]) == (2, None)
        client.optimize(data_size=2)
        assert len(client.search_embedding(QUERY, k=2)) == 1
    assert len(fixture.entries) == 1


def test_loading_nonempty_dataset_cannot_leave_empty_store(service):
    _, port = service
    client = make_client(port)
    with client.init(), pytest.raises(RuntimeError, match="empty store"):
        client.optimize(data_size=1)


def test_rpc_failure_not_retried(service):
    fixture, port = service
    client = make_client(port)
    fixture.fail_insert = True
    with client.init():
        count, error = client.insert_embeddings(VECTORS, IDS)
    assert count == 0
    assert error.non_retryable
    assert fixture.request_count == 1


def test_missing_id_and_wrong_count_fail(service):
    fixture, port = service
    client = make_client(port)
    with client.init():
        assert client.insert_embeddings(VECTORS, IDS) == (4, None)
        with pytest.raises(RuntimeError, match="Expected at most 3"):
            client.optimize(data_size=3)
        fixture.missing_id = True
        with pytest.raises(RuntimeError, match="missing its benchmark ID"):
            client.search_embedding(QUERY)


def test_reused_store_validates_configuration(service):
    _, port = service
    client = make_client(port)
    reused = Ahnlich(dim=3, db_config=client.connection.to_dict(), db_case_config=client.case)
    assert reused._channel is None
    with pytest.raises(ValueError, match="HNSW settings differ"):
        Ahnlich(
            dim=3,
            db_config=client.connection.to_dict(),
            db_case_config=AhnlichIndexConfig(metric_type=MetricType.L2),
        )


def test_unsupported_operations_and_limits(service):
    fixture, port = service
    client = make_client(port, message_size=100)
    with client.init():
        count, error = client.insert_embeddings(VECTORS, IDS)
        assert count == 0
        assert "message_size" in str(error)
        with pytest.raises(ValueError, match="unfiltered"):
            client.prepare_filter(IntFilter(int_value=10))
        with pytest.raises(ValueError, match="ids_only"):
            client.search_embedding(QUERY, payload_profile=PayloadProfile.VECTOR)
        with pytest.raises(NotImplementedError, match="coverage"):
            client.poll_insert_readiness(4)
    assert fixture.request_count == 0


def test_registration_and_cli():
    assert DB.Ahnlich.init_cls is Ahnlich
    assert DB.Ahnlich.config_cls is AhnlichConfig
    assert DB.Ahnlich.case_config_cls() is AhnlichIndexConfig
    runner = CliRunner()
    help_result = runner.invoke(AhnlichCli, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--ef-search" in help_result.output
    refused = runner.invoke(AhnlichCli, ["--load-concurrency", "2"])
    assert refused.exit_code == 2
    dry_run = runner.invoke(AhnlichCli, ["--case-type", "Performance1536D50K", "--dry-run"])
    assert dry_run.exit_code == 0, dry_run.output


@pytest.mark.parametrize(
    "options",
    [
        ["--case-type", "Performance768D1M1P"],
        ["--case-type", "CapacityDim128"],
        ["--case-type", "StreamingPerformanceCase"],
        ["--payload-profile", "vector"],
        ["--nq", "2"],
    ],
)
def test_cli_rejects_nonbaseline_cases(options):
    result = CliRunner().invoke(AhnlichCli, [*options, "--dry-run"])
    assert result.exit_code == 2, result.output
    if "--case-type" in options:
        assert "Invalid value for '--case-type'" in result.output
    else:
        assert "Ahnlich supports only" in result.output


@pytest.fixture
def live_port(tmp_path):
    binary = os.environ.get("AHNLICH_DB_BIN")
    if not binary:
        pytest.skip("Set AHNLICH_DB_BIN to enable the small real-server correctness fixture")
    binary = str(Path(binary).resolve(strict=True))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with (tmp_path / "ahnlich.log").open("w") as log:
        process = subprocess.Popen(
            [binary, "run", "--port", str(port), "--allocator-size", "536870912", "--message-size", "67108864"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                grpc.channel_ready_future(channel).result(timeout=30)
            yield port
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE, MetricType.IP])
def test_real_db_exact_and_hnsw_agree(live_port, metric):
    client = make_client(live_port, metric)
    with client.init():
        assert client.insert_embeddings(VECTORS, IDS) == (4, None)
        hnsw_ids = client.search_embedding(QUERY, k=4)
        client.case.algorithm = "flat"
        assert client.search_embedding(QUERY, k=4) == hnsw_ids == IDS
        client.case.algorithm = "hnsw"
        client.optimize(data_size=4)
        with ThreadPoolExecutor(max_workers=4) as workers:
            assert list(workers.map(lambda _: client.search_embedding(QUERY, k=1), range(8))) == [[19]] * 8
        assert client.insert_embeddings([VECTORS[0], VECTORS[0]], [200, 201]) == (2, None)
        client.optimize(data_size=6)
        updated_ids = client.search_embedding(QUERY, k=4)
        assert len(updated_ids) == 4
        assert updated_ids[0] in {200, 201}
        assert 19 not in updated_ids
        client.case.algorithm = "flat"
        assert client.search_embedding(QUERY, k=4) == updated_ids


@pytest.mark.parametrize("algorithm", ["hnsw", "flat"])
@pytest.mark.parametrize("ef_search", [100, 180, 118])
def test_search_breadth_on_wire(service, algorithm, ef_search):
    fixture, port = service
    client = make_client(port, algorithm=algorithm)
    client.case.ef_search = ef_search
    with client.init():
        client.insert_embeddings(VECTORS, IDS)
        client.search_embedding(QUERY, k=4)
    assert fixture.last_search.HasField("ef_search") == (algorithm == "hnsw")
    if algorithm == "hnsw":
        assert fixture.last_search.ef_search == ef_search


@pytest.mark.parametrize("value", ["0", "-1", "4294967296"])
def test_cli_rejects_invalid_search_breadth(value):
    result = CliRunner().invoke(AhnlichCli, ["--ef-search", value, "--dry-run"])
    assert result.exit_code == 2


def test_cli_search_breadth_config(monkeypatch):
    captured = {}
    monkeypatch.setattr("vectordb_bench.backend.clients.ahnlich.cli.run", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(AhnlichCli, ["--ef-search", "180", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert captured["db_case_config"].search_param()["ef_search"] == 180
