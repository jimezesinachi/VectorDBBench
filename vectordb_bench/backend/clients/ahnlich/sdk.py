"""SDK candidate for comparison with the synchronous grpcio control."""

import asyncio
import copy
import struct
from collections.abc import Sequence
from contextlib import contextmanager

import betterproto
from ahnlich_client_py.grpc import keyval, metadata
from ahnlich_client_py.grpc.algorithm import algorithms, nonlinear
from ahnlich_client_py.grpc.db import query as db_query
from ahnlich_client_py.grpc.services.db_service import DbServiceStub
from grpclib.client import Channel
from grpclib.const import Status
from grpclib.encoding.proto import ProtoCodec
from grpclib.exceptions import GRPCError

from vectordb_bench.backend.payload import PayloadProfile

from ..api import MetricType, PartialInsertError
from .ahnlich import ID_FIELD, Ahnlich


class LimitedProtoCodec(ProtoCodec):
    """Enforce the configured message limit without an extra serialization pass."""

    def __init__(self, limit: int):
        self.limit = limit

    def encode(self, message: betterproto.Message, message_type: type[betterproto.Message]):
        data = super().encode(message, message_type)
        if len(data) > self.limit:
            raise ValueError("Request exceeds message_size; reduce --insert-batch-size")
        return data

    def decode(self, data: bytes, message_type: type[betterproto.Message]):
        if len(data) > self.limit:
            raise ValueError("Response exceeds message_size")
        return super().decode(data, message_type)


class AhnlichSDK(Ahnlich):
    thread_safe = False

    def copy_for_thread(self):
        client = copy.copy(self)
        client._runner = None
        client._channel = None
        client._stub = None
        return client

    @staticmethod
    def _parse_metric(metric: MetricType | None) -> tuple[int, int]:
        mapping = {
            MetricType.L2: (algorithms.DistanceMetric.Euclidean, algorithms.Algorithm.EuclideanDistance),
            MetricType.COSINE: (algorithms.DistanceMetric.Cosine, algorithms.Algorithm.CosineSimilarity),
            MetricType.IP: (algorithms.DistanceMetric.DotProduct, algorithms.Algorithm.DotProductSimilarity),
            MetricType.DP: (algorithms.DistanceMetric.DotProduct, algorithms.Algorithm.DotProductSimilarity),
        }
        if metric not in mapping:
            message = f"Unsupported Ahnlich metric: {metric}"
            raise ValueError(message)
        # betterproto enums are not pickle-safe; worker state stores plain integers.
        distance, algorithm = mapping[metric]
        return int(distance), int(algorithm)

    async def _open_channel(self):
        self._channel = Channel(
            host=self.connection.host,
            port=self.connection.port,
            codec=LimitedProtoCodec(self.connection.message_size),
        )
        self._stub = DbServiceStub(self._channel)

    @contextmanager
    def init(self):
        if self._channel is not None:
            raise RuntimeError("Ahnlich init() contexts must not be nested")
        with asyncio.Runner() as runner:
            self._runner = runner
            try:
                runner.run(self._open_channel())
                yield
            finally:
                if self._channel is not None:
                    self._channel.close()
                self._stub = None
                self._channel = None
                self._runner = None

    def _call(self, method: str, request: betterproto.Message):
        if self._stub is None:
            raise RuntimeError("Call Ahnlich methods inside init()")
        return self._runner.run(getattr(self._stub, method)(request, timeout=self.connection.rpc_timeout))

    def _hnsw_config(self):
        return nonlinear.HnswConfig(
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
            self._call("drop_store", db_query.DropStore(**scope, error_if_not_exists=False))
        try:
            info = self._call("get_store", db_query.GetStore(**scope))
        except GRPCError as error:
            if error.status != Status.NOT_FOUND:
                raise
            indices = []
            if self.case.algorithm == "hnsw":
                indices.append(nonlinear.NonLinearIndex(hnsw=self._hnsw_config()))
            self._call(
                "create_store",
                db_query.CreateStore(**scope, dimension=self.dim, non_linear_indices=indices, error_if_exists=True),
            )
        else:
            if info.dimension != self.dim:
                message = f"Existing store dimension {info.dimension} differs from dataset dimension {self.dim}"
                raise ValueError(message)
            if self.case.algorithm == "hnsw":
                configs = [
                    index.hnsw
                    for index in info.non_linear_indices
                    if betterproto.which_one_of(index, "index")[0] == "hnsw"
                ]
                if configs != [self._hnsw_config()]:
                    raise ValueError("Existing store HNSW settings differ; load a fresh store with --drop-old")

    def _wire_vector(self, vector: Sequence[float]):
        self._validate_vector(vector)
        # Match the control's float32 conversion, including overflow/underflow validation.
        rounded = list(struct.unpack(f"<{self.dim}f", struct.pack(f"<{self.dim}f", *vector)))
        self._validate_vector(rounded)
        return rounded

    def _insert_embeddings(
        self,
        embeddings: list[list[float]],
        identifiers: list[int],
        labels_data: list[str] | None,
        tenant_labels_data: list[str] | None,
    ):
        if len(embeddings) != len(identifiers):
            raise ValueError("Embeddings and benchmark IDs must have equal lengths")
        if labels_data is not None or tenant_labels_data is not None:
            raise ValueError("Label and tenant cases are not supported")
        entries = []
        ids = set()
        for vector, identifier in zip(embeddings, identifiers, strict=True):
            rounded = self._wire_vector(vector)
            if int(identifier) != identifier or int(identifier) < 0:
                raise ValueError("Benchmark IDs must be nonnegative integers")
            if int(identifier) in ids:
                raise ValueError("Benchmark IDs must be unique within a batch")
            ids.add(int(identifier))
            entries.append(
                keyval.DbStoreEntry(
                    key=keyval.StoreKey(key=rounded),
                    value=keyval.StoreValue(
                        value={ID_FIELD: metadata.MetadataValue(raw_string=str(int(identifier)))},
                    ),
                ),
            )
        if not entries:
            return 0, None
        response = self._call(
            "set", db_query.Set(store=self.collection_name, schema=self.connection.namespace, inputs=entries)
        )
        processed = response.upsert.inserted + response.upsert.updated
        if processed != len(entries):
            error = PartialInsertError(
                "Set returned an unexpected processed count; stop this run and reload a fresh store",
                inserted_count=processed,
            )
            return processed, error
        return processed, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ):
        if tenant is not None or payload_profile != PayloadProfile.IDS_ONLY:
            raise ValueError("This adapter supports single-tenant ids_only cases")
        if k < 1:
            raise ValueError("k must be positive")
        response = self._call(
            "get_sim_n",
            db_query.GetSimN(
                store=self.collection_name,
                schema=self.connection.namespace,
                search_input=keyval.StoreKey(key=self._wire_vector(query)),
                closest_n=k,
                algorithm=algorithms.Algorithm.HNSW if self.case.algorithm == "hnsw" else self._flat_algorithm,
            ),
        )
        if len(response.entries) > k:
            raise RuntimeError("Search returned more results than requested")
        identifiers = []
        for entry in response.entries:
            value = entry.value.value.get(ID_FIELD)
            if value is None or betterproto.which_one_of(value, "value")[0] != "raw_string":
                raise RuntimeError("Search result is missing its benchmark ID")
            identifier = int(value.raw_string)
            if identifier < 0:
                raise RuntimeError("Search returned a negative benchmark ID")
            identifiers.append(identifier)
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError("Search returned duplicate benchmark IDs")
        return identifiers

    def optimize(self, data_size: int | None = None):
        info = self._call("get_store", db_query.GetStore(store=self.collection_name, schema=self.connection.namespace))
        if data_size is not None and info.len > data_size:
            message = f"Expected at most {data_size} stored vectors after upserts, found {info.len}"
            raise RuntimeError(message)
        if data_size and not info.len:
            raise RuntimeError("Loading a nonempty dataset left an empty store")
        if info.len and not self.search_embedding([1.0] * self.dim, k=1):
            raise RuntimeError("Nonempty store produced no search results")
