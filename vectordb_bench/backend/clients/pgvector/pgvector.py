"""Wrapper around the Pgvector vector database over VectorDB"""

import logging
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import PgVectorConfigDict, PgVectorIndexConfig

log = logging.getLogger(__name__)

# Namespace for uuid5(tenant label) tenant ids, so load and search processes agree on the ids.
TENANT_UUID_NAMESPACE = uuid.UUID("6f1c6a52-3c1e-4b8e-9d0a-5b7f2f6c1a10")


class PgVector(VectorDB):
    """Use psycopg instructions"""

    thread_safe: bool = False
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _search: sql.Composed
    _exact_search: sql.Composed | None = None

    def __init__(
        self,
        dim: int,
        db_config: PgVectorConfigDict,
        db_case_config: PgVectorIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "PgVector"
        self.case_config = db_case_config
        self.table_name = db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._index_name = "pgvector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"
        self._tenant_field = "tenant_id"

        # CloudMultiTenantSearchCase: the table is keyed by PRIMARY KEY ((tenant_id) HASH, id ASC),
        # and every search filters on one tenant, which YugabyteDB routes to the tablet holding it.
        self.multitenant = "multitenant_tenant_labels" in kwargs
        self.multitenant_tenant_labels = kwargs.get("multitenant_tenant_labels") or []
        if self.multitenant and self.case_config.tenant_column_type not in ("uuid", "text"):
            msg = f"Unsupported tenant_column_type: {self.case_config.tenant_column_type}"
            raise ValueError(msg)

        # Detected lazily on first use: True for YugabyteDB, False for vanilla
        # PostgreSQL. Some statements (e.g. ALTER TABLE ... SET STORAGE) are only
        # valid on PostgreSQL. See _server_is_yugabytedb.
        self._is_yugabytedb: bool | None = None

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        # create vector extension
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()

        log.info(f"{self.name} config values: {self.connect_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {self.connect_config}\n{self.case_config}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        # Smart-driver branch: this build installs psycopg-yugabytedb, a psycopg3
        # fork that is YugabyteDB cluster-aware (the import name stays `psycopg`).
        # When load_balance is on, set the smart-driver connection params so the
        # driver discovers cluster nodes via yb_servers() and distributes new
        # connections across them (uniform / least-connections). That way the
        # per-process search workers don't all pile onto the single configured
        # host. `host` is only the seed used for discovery -- a single endpoint
        # (e.g. master-leader) is sufficient; a comma-separated list also works
        # and just provides multiple bootstrap seeds.
        load_balance = kwargs.pop("load_balance", False)
        topology_keys = kwargs.pop("topology_keys", None)
        if load_balance:
            # libpq/YB smart-driver params (understood by psycopg-yugabytedb's
            # bundled libpq). Don't clobber an explicit caller-provided value.
            kwargs.setdefault("load_balance_hosts", "true")
            if topology_keys:
                kwargs.setdefault("topology_keys", topology_keys)

        conn = psycopg.connect(**kwargs)
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _server_is_yugabytedb(self) -> bool:
        """Return True if the connected server is YugabyteDB, False for vanilla PostgreSQL.

        YugabyteDB reports a version() string like
        'PostgreSQL 11.2-YB-2.20.1.0-b0 on x86_64-...', so the '-YB-' marker
        reliably distinguishes it from PostgreSQL. The result is cached on the
        instance so we only query once.
        """
        if self._is_yugabytedb is None:
            assert self.cursor is not None, "Cursor is not initialized"
            self.cursor.execute("SELECT version()")
            version_str = self.cursor.fetchone()[0]
            self._is_yugabytedb = "-yb-" in version_str.lower()
            log.info(
                f"{self.name} detected server: "
                f"{'YugabyteDB' if self._is_yugabytedb else 'PostgreSQL'} ({version_str})"
            )
        return self._is_yugabytedb

    def _generate_search_query(self) -> sql.Composed:
        index_param = self.case_config.index_param()
        reranking = self.case_config.search_param()["reranking"]
        column_name = (
            sql.SQL("binary_quantize({0})").format(sql.Identifier(self._vector_field))
            if index_param["quantization_type"] == "bit" and index_param["table_quantization_type"] != "bit"
            else sql.SQL(self._vector_field)
        )
        search_vector = (
            sql.SQL("binary_quantize({0})").format(sql.Placeholder())
            if index_param["quantization_type"] == "bit"
            else sql.Placeholder()
        )

        # The following sections assume that the quantization_type value matches the quantization function name
        if index_param["quantization_type"] != index_param["table_quantization_type"]:
            # Reranking makes sense only if table quantization is not "bit"
            if index_param["quantization_type"] == "bit" and reranking:
                # Embeddings needs to be passed to binary_quantize function if quantization_type is bit
                search_query = sql.Composed(
                    [
                        sql.SQL(
                            """
                            SELECT i.id
                            FROM (
                                SELECT {primary_field}, {vector_field} {reranking_metric_fun_op} %s::{table_quantization_type} AS distance
                                FROM public.{table_name} {where_clause}
                                ORDER BY {column_name}::{quantization_type}({dim})
                            """,  # noqa: E501
                        ).format(
                            table_name=sql.Identifier(self.table_name),
                            primary_field=sql.Identifier(self._primary_field),
                            vector_field=sql.Identifier(self._vector_field),
                            column_name=column_name,
                            reranking_metric_fun_op=sql.SQL(
                                self.case_config.search_param()["reranking_metric_fun_op"],
                            ),
                            search_vector=search_vector,
                            table_quantization_type=sql.SQL(index_param["table_quantization_type"]),
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                            where_clause=sql.SQL(self.where_clause),
                        ),
                        sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                        sql.SQL(
                            """
                                {search_vector}::{quantization_type}({dim})
                                LIMIT {quantized_fetch_limit}
                            ) i
                            ORDER BY i.distance
                            LIMIT %s::int
                            """,
                        ).format(
                            search_vector=search_vector,
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                            quantized_fetch_limit=sql.Literal(
                                self.case_config.search_param()["quantized_fetch_limit"],
                            ),
                        ),
                    ],
                )
            else:
                search_query = sql.Composed(
                    [
                        sql.SQL(
                            """
                            SELECT {primary_field} FROM public.{table_name}
                            {where_clause} ORDER BY {column_name}::{quantization_type}({dim})
                            """,
                        ).format(
                            table_name=sql.Identifier(self.table_name),
                            primary_field=sql.Identifier(self._primary_field),
                            column_name=column_name,
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                            where_clause=sql.SQL(self.where_clause),
                        ),
                        sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                        sql.SQL(" {search_vector}::{quantization_type}({dim}) LIMIT %s::int").format(
                            search_vector=search_vector,
                            quantization_type=sql.SQL(index_param["quantization_type"]),
                            dim=sql.Literal(self.dim),
                        ),
                    ]
                )
        else:
            search_query = sql.Composed(
                [
                    sql.SQL(
                        "SELECT {primary_field} FROM public.{table_name} {where_clause} ORDER BY {vector_field}",
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        primary_field=sql.Identifier(self._primary_field),
                        vector_field=sql.Identifier(self._vector_field),
                        where_clause=sql.SQL(self.where_clause),
                    ),
                    sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                    sql.SQL(" {search_vector}::{quantization_type}({dim}) LIMIT %s::int").format(
                        search_vector=search_vector,
                        quantization_type=sql.SQL(index_param["quantization_type"]),
                        dim=sql.Literal(self.dim),
                    ),
                ]
            )

        return search_query

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """

        self.conn, self.cursor = self._create_connection(**self.connect_config)

        # index configuration may have commands defined that we should set during each client session
        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()["session_options"]

        if len(session_options) > 0:
            for setting in session_options:
                command = sql.SQL("SET {setting_name} " + "= {val};").format(
                    setting_name=sql.Identifier(setting["parameter"]["setting_name"]),
                    val=sql.Identifier(str(setting["parameter"]["val"])),
                )
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            self.conn.commit()

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        self._post_insert()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {index_name}").format(
            index_name=sql.Identifier(self._index_name),
        )
        log.debug(drop_index_sql.as_string(self.cursor))
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _set_parallel_index_build_param(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        if index_param["maintenance_work_mem"] is not None:
            self.cursor.execute(
                sql.SQL("SET maintenance_work_mem TO {};").format(
                    index_param["maintenance_work_mem"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET maintenance_work_mem TO {};").format(
                    sql.Identifier(self.connect_config["user"]),
                    index_param["maintenance_work_mem"],
                ),
            )
            self.conn.commit()

        if index_param["max_parallel_workers"] is not None:
            self.cursor.execute(
                sql.SQL("SET max_parallel_maintenance_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_maintenance_workers TO '{}';").format(
                    sql.Identifier(self.connect_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("SET max_parallel_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_workers TO '{}';").format(
                    sql.Identifier(self.connect_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER TABLE {} SET (parallel_workers = {});").format(
                    sql.Identifier(self.table_name),
                    index_param["max_parallel_workers"],
                ),
            )
            self.conn.commit()

        results = self.cursor.execute(sql.SQL("SHOW max_parallel_maintenance_workers;")).fetchall()
        results.extend(self.cursor.execute(sql.SQL("SHOW max_parallel_workers;")).fetchall())
        results.extend(self.cursor.execute(sql.SQL("SHOW maintenance_work_mem;")).fetchall())
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param = self.case_config.index_param()
        self._set_parallel_index_build_param()
        # pgvector registers access methods in lowercase ("hnsw", "ivfflat") but
        # IndexType enum values are uppercase; also IVFFlat maps to "ivfflat" (no underscore).
        index_type_lower = index_param["index_type"].lower()
        if index_type_lower == "ivf_flat":
            index_type_lower = "ivfflat"
        log.info(f"index_type (original={index_param['index_type']}, normalized={index_type_lower})")

        options = []
        for option in index_param["index_creation_with_options"]:
            if option["val"] is not None:
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option["option_name"]),
                        val=sql.Identifier(str(option["val"])),
                    ),
                )
        with_clause = sql.SQL("WITH ({});").format(sql.SQL(", ").join(options)) if any(options) else sql.Composed(())

        if index_param["quantization_type"] != index_param["table_quantization_type"]:
            index_create_sql = sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
                USING {index_type} (({column_name}::{quantization_type}({dim})) {embedding_metric})
                """,
            ).format(
                index_name=sql.Identifier(self._index_name),
                table_name=sql.Identifier(self.table_name),
                column_name=(
                    sql.SQL("binary_quantize({0})").format(sql.Identifier("embedding"))
                    if index_param["quantization_type"] == "bit"
                    else sql.Identifier("embedding")
                ),
                index_type=sql.Identifier(index_type_lower),
                # quantization_type value matches the quantization function name
                quantization_type=sql.SQL(index_param["quantization_type"]),
                dim=self.dim,
                embedding_metric=sql.Identifier(index_param["metric"]),
            )
        else:
            index_create_sql = sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
                USING {index_type} (embedding {embedding_metric})
                """,
            ).format(
                index_name=sql.Identifier(self._index_name),
                table_name=sql.Identifier(self.table_name),
                index_type=sql.Identifier(index_type_lower),
                embedding_metric=sql.Identifier(index_param["metric"]),
            )

        index_create_sql_with_with_clause = (index_create_sql + with_clause).join(" ")
        log.debug(index_create_sql_with_with_clause.as_string(self.cursor))
        self.cursor.execute(index_create_sql_with_with_clause)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            # create table
            if self.multitenant:
                self._create_multitenant_table(dim, index_param["table_quantization_type"])
            elif self.with_scalar_labels:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding {table_quantization_type}({dim}), {label_field} VARCHAR(64)){split};
                        """,  # noqa: E501
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        table_quantization_type=sql.SQL(index_param["table_quantization_type"]),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                        label_field=sql.Identifier(self._scalar_label_field),
                        split=self._split_clause(),
                    )
                )
            else:
                self.cursor.execute(
                    sql.SQL("""
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding {table_quantization_type}({dim})){split};
                        """).format(
                        table_name=sql.Identifier(self.table_name),
                        table_quantization_type=sql.SQL(index_param["table_quantization_type"]),
                        dim=dim,
                        primary_field=sql.Identifier(self._primary_field),
                        split=self._split_clause(),
                    )
                )
            # Keep the embedding column stored inline (no TOAST out-of-line storage /
            # compression) for fair PostgreSQL vector-scan benchmarking. YugabyteDB
            # does not support ALTER TABLE ... SET STORAGE, so skip it there.
            if not self._server_is_yugabytedb():
                self.cursor.execute(
                    sql.SQL(
                        "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;",
                    ).format(table_name=sql.Identifier(self.table_name)),
                )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
            raise e from None

    def _split_clause(self) -> sql.Composable:
        num_tablets = self.case_config.num_tablets
        if not num_tablets:
            return sql.SQL("")
        if not self._server_is_yugabytedb():
            log.warning(f"{self.name} ignoring num_tablets={num_tablets}: the server is not YugabyteDB")
            return sql.SQL("")
        return sql.SQL(" SPLIT INTO {} TABLETS").format(sql.Literal(int(num_tablets)))

    def _create_multitenant_table(self, dim: int, table_quantization_type: str):
        if table_quantization_type not in ("vector", "halfvec"):
            msg = f"CloudMultiTenantSearchCase supports only vector and halfvec tables, not {table_quantization_type}"
            raise NotImplementedError(msg)
        # The tenant must be the leading hash key column for YugabyteDB to route a
        # tenant_id = $1 search to one tablet. Text keys are routed only under the C collation.
        tenant_type = sql.SQL("uuid" if self.case_config.tenant_column_type == "uuid" else 'text COLLATE "C"')
        if self._server_is_yugabytedb():
            primary_key = sql.SQL("PRIMARY KEY (({tenant}) HASH, {id} ASC)")
        else:
            primary_key = sql.SQL("PRIMARY KEY ({tenant}, {id})")
        label_column = (
            sql.SQL(", {} VARCHAR(64)").format(sql.Identifier(self._scalar_label_field))
            if self.with_scalar_labels
            else sql.SQL("")
        )
        self.cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS public.{table_name} "
                "({tenant} {tenant_type}, {id} BIGINT, {vector} {vector_type}({dim}){label_column}, {primary_key})"
                "{split}"
            ).format(
                table_name=sql.Identifier(self.table_name),
                tenant=sql.Identifier(self._tenant_field),
                tenant_type=tenant_type,
                id=sql.Identifier(self._primary_field),
                vector=sql.Identifier(self._vector_field),
                vector_type=sql.SQL(table_quantization_type),
                dim=sql.Literal(dim),
                label_column=label_column,
                primary_key=primary_key.format(
                    tenant=sql.Identifier(self._tenant_field),
                    id=sql.Identifier(self._primary_field),
                ),
                split=self._split_clause(),
            )
        )

    def _tenant_value(self, tenant: str) -> uuid.UUID | str:
        if self.case_config.tenant_column_type == "uuid":
            return uuid.uuid5(TENANT_UUID_NAMESPACE, tenant)
        return tenant

    def _insert_multitenant(
        self,
        embeddings_arr: np.ndarray,
        metadata_arr: np.ndarray,
        tenant_labels_data: list[str],
        labels_data: list[str] | None,
    ):
        table_quantization_type = self.case_config.index_param()["table_quantization_type"]
        columns = [self._tenant_field, self._primary_field, self._vector_field]
        types = [self.case_config.tenant_column_type, "bigint", table_quantization_type]
        if self.with_scalar_labels:
            columns.append(self._scalar_label_field)
            types.append("varchar")
        tenant_values = {label: self._tenant_value(label) for label in set(tenant_labels_data)}
        with self.cursor.copy(
            sql.SQL("COPY public.{table_name} ({columns}) FROM STDIN (FORMAT BINARY)").format(
                table_name=sql.Identifier(self.table_name),
                columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
            )
        ) as copy:
            copy.set_types(types)
            for i, row in enumerate(metadata_arr):
                embedding = np.float16(embeddings_arr[i]) if table_quantization_type == "halfvec" else embeddings_arr[i]
                values = [tenant_values[tenant_labels_data[i]], row, embedding]
                if self.with_scalar_labels:
                    values.append(labels_data[i])
                copy.write_row(values)

    def insert_embeddings(  # noqa: PLR0912
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        if self.with_scalar_labels:
            assert labels_data is not None, "labels_data should be provided if with_scalar_labels is set to True"

        index_param = self.case_config.index_param()

        index_param = self.case_config.index_param()

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            if self.multitenant:
                tenant_labels_data = kwargs.get("tenant_labels_data")
                assert tenant_labels_data is not None, "tenant_labels_data is required for multi-tenant cases"
                self._insert_multitenant(embeddings_arr, metadata_arr, tenant_labels_data, labels_data)
            elif index_param["table_quantization_type"] == "bit":
                with self.cursor.copy(
                    sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT TEXT)").format(
                        table_name=sql.Identifier(self.table_name)
                    )
                ) as copy:
                    # Same logic as pgvector binary_quantize
                    for i, row in enumerate(metadata_arr):
                        embeddings_bit = ""
                        for embedding in embeddings_arr[i]:
                            if embedding > 0:
                                embeddings_bit += "1"
                            else:
                                embeddings_bit += "0"
                        if self.with_scalar_labels:
                            copy.write_row((str(row), embeddings_bit, labels_data[i]))
                        else:
                            copy.write_row((str(row), embeddings_bit))
            else:
                with self.cursor.copy(
                    sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                        table_name=sql.Identifier(self.table_name)
                    )
                ) as copy:
                    if index_param["table_quantization_type"] == "halfvec":
                        for i, row in enumerate(metadata_arr):
                            if self.with_scalar_labels:
                                copy.set_types(["bigint", "halfvec", "varchar"])
                                copy.write_row((row, np.float16(embeddings_arr[i]), labels_data[i]))
                            else:
                                copy.set_types(["bigint", "halfvec"])
                                copy.write_row((row, np.float16(embeddings_arr[i])))
                    else:
                        for i, row in enumerate(metadata_arr):
                            if self.with_scalar_labels:
                                copy.set_types(["bigint", "vector", "varchar"])
                                copy.write_row((row, embeddings_arr[i], labels_data[i]))
                            else:
                                copy.set_types(["bigint", "vector"])
                                copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into pgvector table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            conditions = []
        elif filters.type == FilterOp.NumGE:
            conditions = [f"{self._primary_field} >= {filters.int_value}"]
        elif filters.type == FilterOp.StrEqual:
            conditions = [f"{self._scalar_label_field} = '{filters.label_value}'"]
        else:
            msg = f"Not support Filter for PgVector - {filters}"
            raise ValueError(msg)

        if self.multitenant:
            index_param = self.case_config.index_param()
            if index_param["quantization_type"] != index_param["table_quantization_type"]:
                msg = "CloudMultiTenantSearchCase doesn't support index quantization different from the table's"
                raise NotImplementedError(msg)
            # Bound per search; the tenant value is the first query parameter.
            conditions.insert(0, f"{self._tenant_field} = %s")
        self.where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        self._search = self._generate_search_query()
        self._exact_search = None
        if self.multitenant and self.case_config.force_vector_index:
            hint = self._index_hint(self._index_name)
            self._check_plan(hint, self._search, uses_vector_index=True)
            self._search = sql.Composed([hint, self._search])

    def _index_hint(self, index_name: str) -> sql.Composable:
        # pg_hint_plan reads the hint from a comment at the head of the statement.
        return sql.SQL("/*+ IndexScan({} {}) */ ").format(sql.Identifier(self.table_name), sql.Identifier(index_name))

    def _check_plan(self, hint: sql.Composable, query: sql.Composable, uses_vector_index: bool):
        """Fails unless the plan of hint + query uses (or, for the exact search, avoids) the
        vector index. EXPLAIN doesn't run the query, so any tenant and vector will do."""
        assert self.cursor is not None, "Cursor is not initialized"
        tenant = self._tenant_value(self.multitenant_tenant_labels[0] if self.multitenant_tenant_labels else "")
        params = (tenant, np.zeros(self.dim, dtype=np.float32), 10)
        rows = self.cursor.execute(sql.Composed([hint, sql.SQL("EXPLAIN "), query]), params).fetchall()
        self.conn.commit()
        plan = "\n".join(r[0] for r in rows)
        if (self._index_name in plan) != uses_vector_index:
            msg = (
                f"{self.name} plan {'does not use' if uses_vector_index else 'uses'} the vector index "
                f"{self._index_name}:\n{plan}"
            )
            raise RuntimeError(msg)
        log.info(f"{self.name} search plan:\n{plan}")

    def supports_multitenant(self) -> bool:
        return True

    def supports_exact_tenant_search(self) -> bool:
        return self.multitenant

    def exact_tenant_search(self, query: list[float], k: int, tenant: str) -> list[int]:
        assert self.cursor is not None, "Cursor is not initialized"
        if self._exact_search is None:
            index_param = self.case_config.index_param()
            body = sql.SQL(
                "SELECT {id} FROM public.{table_name} {where_clause} ORDER BY {vector} {op} %s::{vector_type}({dim}) "
                "LIMIT %s::int"
            ).format(
                id=sql.Identifier(self._primary_field),
                table_name=sql.Identifier(self.table_name),
                where_clause=sql.SQL(self.where_clause),
                vector=sql.Identifier(self._vector_field),
                op=sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                vector_type=sql.SQL(index_param["table_quantization_type"]),
                dim=sql.Literal(self.dim),
            )
            # Read the tenant's rows through the primary key and sort them all by distance.
            hint = self._index_hint(f"{self.table_name}_pkey")
            self._check_plan(hint, body, uses_vector_index=False)
            self._exact_search = sql.Composed([hint, body])
        result = self.cursor.execute(
            self._exact_search, (self._tenant_value(tenant), np.asarray(query), k), prepare=True, binary=True
        )
        rows = [int(r[0]) for r in result.fetchall()]
        self.conn.commit()
        return rows

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()
        search_param = self.case_config.search_param()
        q = np.asarray(query)
        if self.multitenant:
            params = (self._tenant_value(kwargs["tenant"]), q, k)
        elif index_param["quantization_type"] == "bit" and search_param["reranking"]:
            params = (q, q, k)
        else:
            params = (q, k)
        result = self.cursor.execute(self._search, params, prepare=True, binary=True)
        return [int(i[0]) for i in result.fetchall()]
