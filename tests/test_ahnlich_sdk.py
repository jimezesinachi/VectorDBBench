"""SDK candidate correctness against the same wire fixture as the control."""

# ruff: noqa: ANN001

import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest
from click.testing import CliRunner
from test_ahnlich import IDS, QUERY, VECTORS, make_client
from test_ahnlich import live_port as live_port  # noqa: PLC0414 -- Re-export pytest fixture.
from test_ahnlich import service as service  # noqa: PLC0414 -- Re-export pytest fixture.

from vectordb_bench.backend.clients.ahnlich.cli import Ahnlich as AhnlichCli
from vectordb_bench.backend.clients.ahnlich.sdk import AhnlichSDK
from vectordb_bench.backend.clients.api import MetricType


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE, MetricType.IP, MetricType.DP])
@pytest.mark.parametrize("algorithm", ["hnsw", "flat"])
def test_sdk_roundtrip_and_upserts(service, metric, algorithm):
    _, port = service
    client = make_client(port, metric, algorithm, implementation="sdk")
    assert isinstance(client, AhnlichSDK)
    with client.init():
        assert client.insert_embeddings(VECTORS, IDS) == (4, None)
        assert client.search_embedding(QUERY, k=4) == IDS
        assert client.insert_embeddings([VECTORS[0], VECTORS[0]], [200, 201]) == (2, None)
        client.optimize(data_size=6)
        assert client.search_embedding(QUERY, k=4) == [201, *IDS[1:]]
        loop = client._runner.get_loop()
        client.search_embedding(QUERY, k=1)
        assert client._runner.get_loop() is loop
    assert client._runner is client._channel is client._stub is None
    restored = pickle.loads(pickle.dumps(client))  # noqa: S301 -- Locally created worker state.
    with restored.init():
        assert restored.search_embedding(QUERY, k=1) == [201]


def test_sdk_failures_and_message_limits(service):
    fixture, port = service
    client = make_client(port, implementation="sdk")
    with client.init():
        fixture.fail_insert = True
        count, error = client.insert_embeddings(VECTORS, IDS)
        assert count == 0
        assert error.non_retryable
        fixture.fail_insert = False
        assert client.insert_embeddings(VECTORS, IDS) == (4, None)
        fixture.missing_id = True
        with pytest.raises(RuntimeError, match="benchmark ID"):
            client.search_embedding(QUERY)
        fixture.missing_id = False
        client._channel._codec.limit = 100
        with pytest.raises(ValueError, match="Response exceeds"):
            client.search_embedding(QUERY, k=4)
        count, error = client.insert_embeddings(VECTORS, IDS)
        assert count == 0
        assert "message_size" in str(error)


def test_sdk_float32_and_thread_ownership(service):
    _, port = service
    client = make_client(port, implementation="sdk")
    with client.init():
        assert client.insert_embeddings([[1, 0, 0], [1 + 1e-9, 0, 0]], [1, 2]) == (2, None)
        client.optimize(data_size=2)
        assert client.search_embedding(QUERY, k=2) == [2]
        for vector in ([1e100, 0, 0], [1e-100, 0, 0]):
            count, error = client.insert_embeddings([vector], [3])
            assert count == 0
            assert error.non_retryable
        copies = [client.copy_for_thread() for _ in range(4)]

        def search(worker):
            with worker.init():
                return worker.search_embedding(QUERY, k=1)

        with ThreadPoolExecutor(max_workers=4) as workers:
            assert list(workers.map(search, copies)) == [[2]] * 4
        assert not client.thread_safe


def test_sdk_cli_selection():
    result = CliRunner().invoke(AhnlichCli, ["--implementation", "sdk", "--dry-run"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE, MetricType.IP])
def test_sdk_real_db_matches_control(live_port, metric):
    control = make_client(live_port, metric)
    with control.init():
        assert control.insert_embeddings(VECTORS, IDS) == (4, None)
        expected = control.search_embedding(QUERY, k=4)
    sdk = AhnlichSDK(
        dim=3,
        db_config={**control.connection.to_dict(), "implementation": "sdk"},
        db_case_config=control.case,
    )
    with sdk.init():
        assert sdk.search_embedding(QUERY, k=4) == expected
        assert sdk.insert_embeddings([VECTORS[0], VECTORS[0]], [200, 201]) == (2, None)
        sdk.optimize(data_size=6)
        updated = sdk.search_embedding(QUERY, k=4)
        assert updated[0] in {200, 201}
        assert 19 not in updated
    with control.init():
        assert control.search_embedding(QUERY, k=4) == updated


def test_sdk_search_breadth_on_wire(service):
    fixture, port = service
    client = make_client(port, implementation="sdk")
    client.case.ef_search = 180
    with client.init():
        client.insert_embeddings(VECTORS, IDS)
        client.search_embedding(QUERY, k=4)
    assert fixture.last_search.HasField("ef_search")
    assert fixture.last_search.ef_search == 180
