import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench import config
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    IVFFlatTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from .config import LB_STRATEGIES, LB_STRATEGY_SMART_DRIVER


def set_default_quantized_fetch_limit(ctx: any, param: any, value: any):  # noqa: ARG001
    if ctx.params.get("reranking") and value is None:
        # ef_search is the default value for quantized_fetch_limit as it's bound by ef_search.
        # 100 is default value for quantized_fetch_limit for IVFFlat.
        return ctx.params["ef_search"] if ctx.command.name == "pgvectorhnsw" else 100
    return value


class PgVectorTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Postgres database password",
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]

    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Postgres database port",
            default=5432,
            show_default=True,
            required=False,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]
    load_balance: Annotated[
        bool,
        click.option(
            "--load-balance/--skip-load-balance",
            type=bool,
            default=True,
            show_default=True,
            help="Distribute connections across the YugabyteDB cluster instead of sending every "
            "concurrent-search worker to the configured host. On by default, so it is always on "
            "for YugabyteDB; it is auto-disabled when the server is detected to be vanilla "
            "PostgreSQL. Pass --skip-load-balance to pin every connection to --host. "
            "See --lb-strategy for how the connections are placed.",
        ),
    ]
    load_balance_strategy: Annotated[
        str,
        click.option(
            # perfservice keys config_file.yaml by the click parameter name, and its
            # payload already spells the sibling flag out as `load_balance`, so the
            # long name is what the pipeline writes; --lb-strategy is a typing alias.
            "--load-balance-strategy",
            "--lb-strategy",
            type=click.Choice(list(LB_STRATEGIES)),
            default=LB_STRATEGY_SMART_DRIVER,
            show_default=True,
            help="How --load-balance places connections. 'smart_driver': the YugabyteDB smart "
            "driver discovers nodes via yb_servers() and places them (requires the "
            "psycopg-yugabytedb driver). 'round_robin': VectorDBBench assigns connections to "
            "nodes itself in strict round-robin order, so they split as evenly as the counts "
            "allow (8 connections over 3 nodes -> 3/3/2). round_robin takes its node list from "
            "a comma-separated --host, else from yb_servers().",
        ),
    ]
    topology_keys: Annotated[
        str | None,
        click.option(
            "--topology-keys",
            type=str,
            default=None,
            required=False,
            help="Optional YB topology hint (cloud.region.zone, comma-separated; '*' wildcard "
            "allowed). Restricts load balancing to matching nodes under either --lb-strategy; "
            "omit to balance across all nodes.",
        ),
    ]
    create_index_before_load: Annotated[
        bool,
        click.option(
            "--create-index-before-load/--skip-create-index-before-load",
            type=bool,
            default=False,
            show_default=True,
            help="Create the index before loading data (empty index, then insert)",
        ),
    ]
    create_index_after_load: Annotated[
        bool,
        click.option(
            "--create-index-after-load/--skip-create-index-after-load",
            type=bool,
            default=True,
            show_default=True,
            help="Create the index after loading data (insert, then build index)",
        ),
    ]
    maintenance_work_mem: Annotated[
        str | None,
        click.option(
            "--maintenance-work-mem",
            type=str,
            help="Sets the maximum memory to be used for maintenance operations (index creation). "
            "Can be entered as string with unit like '64GB' or as an integer number of KB."
            "This will set the parameters: max_parallel_maintenance_workers,"
            " max_parallel_workers & table(parallel_workers)",
            required=False,
        ),
    ]
    max_parallel_workers: Annotated[
        int | None,
        click.option(
            "--max-parallel-workers",
            type=int,
            help="Sets the maximum number of parallel processes per maintenance operation (index creation)",
            required=False,
        ),
    ]
    quantization_type: Annotated[
        str | None,
        click.option(
            "--quantization-type",
            type=click.Choice(["none", "bit", "halfvec"]),
            help="quantization type for vectors (in index)",
            required=False,
        ),
    ]
    table_quantization_type: Annotated[
        str | None,
        click.option(
            "--table-quantization-type",
            type=click.Choice(["none", "bit", "halfvec"]),
            help="quantization type for vectors (in table). "
            "If equal to bit, the parameter quantization_type will be set to bit too.",
            required=False,
        ),
    ]
    reranking: Annotated[
        bool | None,
        click.option(
            "--reranking/--skip-reranking",
            type=bool,
            help="Enable reranking for HNSW search for binary quantization",
            default=False,
        ),
    ]
    reranking_metric: Annotated[
        str | None,
        click.option(
            "--reranking-metric",
            type=click.Choice(
                [metric.value for metric in MetricType if metric.value not in ["HAMMING", "JACCARD"]],
            ),
            help="Distance metric for reranking",
            default="COSINE",
            show_default=True,
        ),
    ]
    quantized_fetch_limit: Annotated[
        int | None,
        click.option(
            "--quantized-fetch-limit",
            type=int,
            help="Limit of fetching quantized vector ranked by distance for reranking \
                -- bound by ef_search",
            required=False,
            callback=set_default_quantized_fetch_limit,
        ),
    ]


class PgVectorIVFFlatTypedDict(PgVectorTypedDict, IVFFlatTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectorIVFFlatTypedDict)
def PgVectorIVFFlat(
    **parameters: Unpack[PgVectorIVFFlatTypedDict],
):
    from .config import PgVectorConfig, PgVectorIVFFlatConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.PgVector,
        db_config=PgVectorConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            load_balance=parameters["load_balance"],
            load_balance_strategy=parameters["load_balance_strategy"],
            topology_keys=parameters["topology_keys"],
        ),
        db_case_config=PgVectorIVFFlatConfig(
            metric_type=None,
            create_index_before_load=parameters["create_index_before_load"],
            create_index_after_load=parameters["create_index_after_load"],
            lists=parameters["lists"],
            probes=parameters["probes"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            quantization_type=parameters["quantization_type"],
            table_quantization_type=parameters["table_quantization_type"],
            reranking=parameters["reranking"],
            reranking_metric=parameters["reranking_metric"],
            quantized_fetch_limit=parameters["quantized_fetch_limit"],
        ),
        **parameters,
    )


class PgVectorHNSWTypedDict(PgVectorTypedDict, HNSWFlavor1): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(PgVectorHNSWTypedDict)
def PgVectorHNSW(
    **parameters: Unpack[PgVectorHNSWTypedDict],
):
    from .config import PgVectorConfig, PgVectorHNSWConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.PgVector,
        db_config=PgVectorConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            load_balance=parameters["load_balance"],
            load_balance_strategy=parameters["load_balance_strategy"],
            topology_keys=parameters["topology_keys"],
        ),
        db_case_config=PgVectorHNSWConfig(
            create_index_before_load=parameters["create_index_before_load"],
            create_index_after_load=parameters["create_index_after_load"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            maintenance_work_mem=parameters["maintenance_work_mem"],
            max_parallel_workers=parameters["max_parallel_workers"],
            quantization_type=parameters["quantization_type"],
            table_quantization_type=parameters["table_quantization_type"],
            reranking=parameters["reranking"],
            reranking_metric=parameters["reranking_metric"],
            quantized_fetch_limit=parameters["quantized_fetch_limit"],
        ),
        **parameters,
    )
