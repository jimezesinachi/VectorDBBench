"""Thin synchronous transport adapter; all indexing and scoring run in ahnlich-db."""

import math
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import grpc

from vectordb_bench.backend.payload import PayloadProfile

from ..api import MetricType, NonRetryableInsertError, PartialInsertError, VectorDB
from ._grpc import keyval_pb2, metadata_pb2
from ._grpc.algorithm import algorithm_pb2, nonlinear_pb2
from ._grpc.db import query_pb2
from ._grpc.services.db_service_pb2_grpc import DBServiceStub
from .config import AhnlichConfig, AhnlichIndexConfig

if TYPE_CHECKING:
    from google.protobuf.message import Message

    from vectordb_bench.backend.filter import Filter

ID_FIELD = "vdbbench_id"


class Ahnlich(VectorDB):
    name = "Ahnlich"
    thread_safe = True  # grpc.Channel supports concurrent synchronous calls.

    def __new__(cls, *args, **kwargs):
        db_config = kwargs.get("db_config", args[1] if len(args) > 1 else {})
        if cls is Ahnlich and db_config.get("implementation", "grpc") == "sdk":
            from .sdk import AhnlichSDK

            return object.__new__(AhnlichSDK)
        return object.__new__(cls)

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AhnlichIndexConfig,
        collection_name: str = "vdbbench_ahnlich",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        if dim < 1:
            raise ValueError("Ahnlich requires a positive vector dimension")
        if with_scalar_labels:
            raise ValueError("This adapter currently supports unfiltered vector cases only")
        self.dim = dim
        self.connection = AhnlichConfig(**db_config)
        self.collection_name = db_config.get("collection_name", collection_name)
        self.case = db_case_config
        self._metric, self._flat_algorithm = self._parse_metric(self.case.metric_type)
        self._channel = None
        self._stub = None
        # Setup has its own short-lived channel; no live channel crosses a fork.
        with self.init():
            self._setup_store(drop_old)

    @staticmethod
    def _parse_metric(metric: MetricType | None) -> tuple[int, int]:
        mapping = {
            MetricType.L2: (algorithm_pb2.Euclidean, algorithm_pb2.EuclideanDistance),
            MetricType.COSINE: (algorithm_pb2.Cosine, algorithm_pb2.CosineSimilarity),
            MetricType.IP: (algorithm_pb2.DotProduct, algorithm_pb2.DotProductSimilarity),
            MetricType.DP: (algorithm_pb2.DotProduct, algorithm_pb2.DotProductSimilarity),
        }
        if metric not in mapping:
            message = f"Unsupported Ahnlich metric: {metric}"
            raise ValueError(message)
        return mapping[metric]

    @contextmanager
    def init(self):
        if self._channel is not None:
            raise RuntimeError("Ahnlich init() contexts must not be nested")
        target = f"{self.connection.host}:{self.connection.port}"
        options = [
            ("grpc.max_send_message_length", self.connection.message_size),
            ("grpc.max_receive_message_length", self.connection.message_size),
        ]
        channel = grpc.insecure_channel(target, options=options)
        try:
            grpc.channel_ready_future(channel).result(timeout=self.connection.rpc_timeout)
            self._channel = channel
            self._stub = DBServiceStub(channel)
            yield
        finally:
            self._stub = None
            self._channel = None
            channel.close()

    def _call(self, method: str, request: "Message"):
        if self._stub is None:
            raise RuntimeError("Call Ahnlich methods inside init()")
        return getattr(self._stub, method)(request, timeout=self.connection.rpc_timeout)

    def _hnsw_config(self):
        return nonlinear_pb2.HNSWConfig(
            distance=self._metric,
            ef_construction=self.case.ef_construction,
            maximum_connections=self.case.m,
            maximum_connections_zero=self.case.m_zero,
            extend_candidates=self.case.extend_candidates,
            keep_pruned_connections=self.case.keep_pruned_connections,
        )

    def _setup_store(self, drop_old: bool):
        scope = {"store": self.collection_name, "schema": self.connection.namespace}
        if drop_old:
            self._call("DropStore", query_pb2.DropStore(**scope, error_if_not_exists=False))
        try:
            info = self._call("GetStore", query_pb2.GetStore(**scope))
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.NOT_FOUND:
                raise
            indices = []
            if self.case.algorithm == "hnsw":
                indices.append(nonlinear_pb2.NonLinearIndex(hnsw=self._hnsw_config()))
            self._call(
                "CreateStore",
                query_pb2.CreateStore(**scope, dimension=self.dim, non_linear_indices=indices, error_if_exists=True),
            )
        else:
            if info.dimension != self.dim:
                message = f"Existing store dimension {info.dimension} differs from dataset dimension {self.dim}"
                raise ValueError(message)
            if self.case.algorithm == "hnsw":
                configs = [index.hnsw for index in info.non_linear_indices if index.WhichOneof("index") == "hnsw"]
                if configs != [self._hnsw_config()]:
                    raise ValueError("Existing store HNSW settings differ; load a fresh store with --drop-old")

    def prepare_filter(self, filters: "Filter"):
        if not self.filter_supported(filters):
            raise ValueError("This adapter currently supports unfiltered vector cases only")

    def _validate_vector(self, vector: Sequence[float]):
        if len(vector) != self.dim or not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("Vector must have the configured dimension and finite components")
        if self._metric == algorithm_pb2.Cosine and not any(vector):
            raise ValueError("Cosine search requires nonzero vectors")

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            return self._insert_embeddings(embeddings, metadata, labels_data, tenant_labels_data)
        except Exception as error:
            # A failed RPC may already have applied writes. Replaying can hide
            # partial work and overwrite dataset IDs; abort instead.
            failure = NonRetryableInsertError(f"Ahnlich insert failed: {error}")
            failure.__cause__ = error
            return 0, failure

    def _insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
        tenant_labels_data: list[str] | None,
    ) -> tuple[int, Exception | None]:
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and benchmark IDs must have equal lengths")
        if labels_data is not None or tenant_labels_data is not None:
            raise ValueError("Label and tenant cases are not supported")
        entries = []
        ids = set()
        for vector, identifier in zip(embeddings, metadata, strict=True):
            self._validate_vector(vector)
            if int(identifier) != identifier or int(identifier) < 0:
                raise ValueError("Benchmark IDs must be nonnegative integers")
            key = keyval_pb2.StoreKey(key=vector)
            self._validate_vector(key.key)
            if int(identifier) in ids:
                raise ValueError("Benchmark IDs must be unique within a batch")
            ids.add(int(identifier))
            entries.append(
                keyval_pb2.DbStoreEntry(
                    key=key,
                    value=keyval_pb2.StoreValue(
                        value={ID_FIELD: metadata_pb2.MetadataValue(raw_string=str(int(identifier)))},
                    ),
                ),
            )
        if not entries:
            return 0, None
        request = query_pb2.Set(store=self.collection_name, schema=self.connection.namespace, inputs=entries)
        if request.ByteSize() > self.connection.message_size:
            raise ValueError("Set request exceeds message_size; reduce --insert-batch-size")
        result = self._call("Set", request).upsert
        processed = result.inserted + result.updated
        if processed != len(entries):
            error = PartialInsertError(
                "Set returned an unexpected processed count; stop this run and reload a fresh store",
                inserted_count=processed,
            )
            return processed, error
        # The loader counts processed dataset rows, including native upserts.
        return processed, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ) -> list[int]:
        if tenant is not None or payload_profile != PayloadProfile.IDS_ONLY:
            raise ValueError("This adapter supports single-tenant ids_only cases")
        if k < 1:
            raise ValueError("k must be positive")
        self._validate_vector(query)
        algorithm = algorithm_pb2.HNSW if self.case.algorithm == "hnsw" else self._flat_algorithm
        response = self._call(
            "GetSimN",
            query_pb2.GetSimN(
                store=self.collection_name,
                schema=self.connection.namespace,
                search_input=keyval_pb2.StoreKey(key=query),
                closest_n=k,
                ef_search=self.case.ef_search if self.case.algorithm == "hnsw" else None,
                algorithm=algorithm,
            ),
        )
        if len(response.entries) > k:
            raise RuntimeError("Search returned more results than requested")
        identifiers = []
        for entry in response.entries:
            value = entry.value.value.get(ID_FIELD)
            if value is None or value.WhichOneof("value") != "raw_string":
                raise RuntimeError("Search result is missing its benchmark ID")
            identifier = int(value.raw_string)
            if identifier < 0:
                raise RuntimeError("Search returned a negative benchmark ID")
            identifiers.append(identifier)
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError("Search returned duplicate benchmark IDs")
        return identifiers

    def optimize(self, data_size: int | None = None):
        # Index construction happens synchronously inside Set. GetStore checks
        # the primary count only; the DB has no index-coverage endpoint.
        info = self._call(
            "GetStore",
            query_pb2.GetStore(store=self.collection_name, schema=self.connection.namespace),
        )
        if data_size is not None and info.len > data_size:
            message = f"Expected at most {data_size} stored vectors after upserts, found {info.len}"
            raise RuntimeError(message)
        if data_size and not info.len:
            raise RuntimeError("Loading a nonempty dataset left an empty store")
        if info.len and not self.search_embedding([1.0] * self.dim, k=1):
            raise RuntimeError("Nonempty store produced no search results")

    def poll_insert_readiness(self, expected_count: int) -> dict:
        raise NotImplementedError("Ahnlich has no index coverage endpoint; insert-readiness cases are unsupported")
