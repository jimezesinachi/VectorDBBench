"""Memory diagnostic sampling and failure detection without loading workloads."""

# ruff: noqa: ANN001
import json
from types import SimpleNamespace

import grpc
import psutil
import pytest

from vectordb_bench.backend.clients.ahnlich.tools import diagnose_ahnlich_memory as diagnostic


def test_memory_sample_records_allocator_rss_and_count():
    stub = SimpleNamespace(
        InfoServer=lambda _request, **_kwargs: SimpleNamespace(info=SimpleNamespace(limit=1000, remaining=250)),
        GetStore=lambda _request, **_kwargs: SimpleNamespace(len=100),
    )
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=800))
    sample = diagnostic.memory_sample(stub, process, "loading")
    assert sample["allocator_used_bytes"] == 750
    assert sample["allocator_limit_bytes"] == 1000
    assert sample["rss_bytes"] == 800
    assert sample["stored_vectors"] == 100
    assert sample["error"] == ""


def test_baseline_without_store_has_zero_count():
    class MissingStore(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.NOT_FOUND

    def get_store(_request, **_kwargs):
        raise MissingStore

    stub = SimpleNamespace(
        InfoServer=lambda _request, **_kwargs: SimpleNamespace(info=SimpleNamespace(limit=1000, remaining=900)),
        GetStore=get_store,
    )
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=200))
    sample = diagnostic.memory_sample(stub, process, "baseline")
    assert sample["stored_vectors"] == 0
    assert sample["error"] == ""


def test_dead_process_is_recorded():
    def memory_info():
        raise psutil.NoSuchProcess(123)

    sample = diagnostic.memory_sample(None, SimpleNamespace(memory_info=memory_info), "loading")
    assert sample["error"]
    assert sample["allocator_used_bytes"] == ""


@pytest.mark.parametrize(("label", "exit_code", "passes"), [(":)", None, True), ("x", None, False), (":)", 1, False)])
def test_zero_loader_exit_cannot_hide_task_or_server_failure(tmp_path, label, exit_code, passes):
    directory = tmp_path / "load" / "json"
    directory.mkdir(parents=True)
    (directory / "result_fixture.json").write_text(json.dumps({"results": [{"label": label}]}))
    server = SimpleNamespace(poll=lambda: exit_code)
    if passes:
        diagnostic.validate_load(tmp_path, server)
    else:
        with pytest.raises(RuntimeError):
            diagnostic.validate_load(tmp_path, server)
