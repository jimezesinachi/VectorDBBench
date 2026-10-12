from typing import Annotated, Unpack

import click

from ....cli.cli import CommonTypedDict, click_parameter_decorators_from_typed_dict, run
from ...cases import type2case
from .. import DB


class AhnlichTypedDict(CommonTypedDict):
    case_type: Annotated[
        str,
        click.option(
            "--case-type",
            type=click.Choice(
                [
                    case.name
                    for case, case_class in type2case.items()
                    if case.name.startswith("Performance") and case_class.model_fields["filter_rate"].default is None
                ]
            ),
            is_eager=True,
            default="Performance1536D50K",
            help="Unfiltered vector performance case",
        ),
    ]
    implementation: Annotated[
        str,
        click.option("--implementation", type=click.Choice(["grpc", "sdk"]), default="grpc", show_default=True),
    ]
    host: Annotated[str, click.option("--host", default="127.0.0.1", show_default=True)]
    port: Annotated[int, click.option("--port", type=click.IntRange(1, 65535), default=1369, show_default=True)]
    collection_name: Annotated[str, click.option("--collection-name", default="vdbbench_ahnlich", show_default=True)]
    schema: Annotated[str, click.option("--schema", default="public", show_default=True)]
    server_version: Annotated[str, click.option("--server-version", default="", help="DB revision recorded in results")]
    rpc_timeout: Annotated[
        float, click.option("--rpc-timeout", type=click.FloatRange(min=0, min_open=True), default=60)
    ]
    message_size: Annotated[int, click.option("--message-size", type=click.IntRange(min=1), default=64 * 1024 * 1024)]
    algorithm: Annotated[str, click.option("--algorithm", type=click.Choice(["hnsw", "flat"]), default="hnsw")]
    m: Annotated[int, click.option("--m", type=click.IntRange(min=2), default=16)]
    m_zero: Annotated[int, click.option("--m-zero", type=click.IntRange(min=2), default=32)]
    ef_construction: Annotated[int, click.option("--ef-construction", type=click.IntRange(min=1), default=200)]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=click.IntRange(1, 2**32 - 1),
            default=100,
            show_default=True,
            help="HNSW search breadth; raised to at least k by the server; ignored for flat search",
        ),
    ]
    extend_candidates: Annotated[bool, click.option("--extend-candidates/--no-extend-candidates", default=False)]
    keep_pruned_connections: Annotated[
        bool,
        click.option("--keep-pruned-connections/--no-keep-pruned-connections", default=False),
    ]
    # Override upstream's 0=CPU count default until graph writes are coordinated.
    load_concurrency: Annotated[int, click.option("--load-concurrency", type=click.IntRange(1, 1), default=1)]


@click.command(name="ahnlich")
@click_parameter_decorators_from_typed_dict(AhnlichTypedDict)
def Ahnlich(**parameters: Unpack[AhnlichTypedDict]):
    """Run unfiltered vector cases against a standalone, unauthenticated Ahnlich DB."""
    from .config import AhnlichConfig, AhnlichIndexConfig

    if parameters["payload_profile"] != "ids_only" or parameters["nq"] != 1:
        raise click.UsageError("Ahnlich supports only ids_only responses and one query vector per request")

    run(
        db=DB.Ahnlich,
        db_config=AhnlichConfig(
            db_label=parameters["db_label"],
            version=parameters["server_version"],
            implementation=parameters["implementation"],
            host=parameters["host"],
            port=parameters["port"],
            collection_name=parameters["collection_name"],
            schema=parameters["schema"],
            rpc_timeout=parameters["rpc_timeout"],
            message_size=parameters["message_size"],
        ),
        db_case_config=AhnlichIndexConfig(
            algorithm=parameters["algorithm"],
            m=parameters["m"],
            m_zero=parameters["m_zero"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            extend_candidates=parameters["extend_candidates"],
            keep_pruned_connections=parameters["keep_pruned_connections"],
        ),
        **parameters,
    )
