from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..api import DBCaseConfig, DBConfig, MetricType


class AhnlichConfig(DBConfig):
    implementation: Literal["grpc", "sdk"] = "grpc"
    host: str = "127.0.0.1"
    port: int = Field(default=1369, ge=1, le=65535)
    collection_name: str = "vdbbench_ahnlich"
    namespace: str = Field(default="public", alias="schema")
    rpc_timeout: float = Field(default=60, gt=0)
    message_size: int = Field(default=64 * 1024 * 1024, gt=0)

    def to_dict(self) -> dict:
        return self.model_dump(by_alias=True, exclude={"db_label", "version", "note"})


class AhnlichIndexConfig(BaseModel, DBCaseConfig):
    model_config = ConfigDict(extra="forbid")

    metric_type: MetricType | None = None  # Set from the dataset by Assembler.
    algorithm: Literal["hnsw", "flat"] = "hnsw"
    m: int = Field(default=16, ge=2)
    m_zero: int = Field(default=32, ge=2)
    ef_construction: int = Field(default=200, ge=1)
    ef_search: int = Field(default=100, ge=1, le=2**32 - 1)
    extend_candidates: bool = False
    keep_pruned_connections: bool = False

    def index_param(self) -> dict:
        return self.model_dump()

    def search_param(self) -> dict:
        return {"algorithm": self.algorithm, "ef_search": self.ef_search}
