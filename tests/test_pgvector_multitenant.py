"""Unit tests for CloudMultiTenantSearchCase support in the PgVector client.

They use a fake cursor and need no database. The table and query shapes matter for
YugabyteDB: a search is routed to one tablet only when it filters on the leading hash key.
"""

from __future__ import annotations

import random
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

from vectordb_bench.backend.clients.pgvector import pgvector as pgvector_module
from vectordb_bench.backend.clients.pgvector.config import PgVectorHNSWConfig
from vectordb_bench.backend.clients.pgvector.pgvector import TENANT_UUID_NAMESPACE, PgVector
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner

DIM = 4
TENANTS = ["tenant_0000", "tenant_0001", "tenant_0002"]


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0]


class FakeCopy:
    def __init__(self, statement):
        self.statement = statement
        self.types = None
        self.rows = []

    def set_types(self, types):
        self.types = types

    def write_row(self, row):
        self.rows.append(row)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeCursor:
    # psycopg's Composable.as_string(cursor) reads cursor.connection for the encoding.
    connection = None

    def __init__(self, server_version: str, plan: str):
        self.server_version = server_version
        self.plan = plan
        self.executed = []
        self.copies = []
        # (format_type, collation) of the tenant_id column, or None if it doesn't exist.
        self.tenant_column = ("uuid", None)
        self.hint_plan_setting = "on"

    def execute(self, query, params=None, **kwargs):
        text = query if isinstance(query, str) else query.as_string(None)
        self.executed.append((text, params))
        if "version()" in text:
            self.result = FakeResult([(self.server_version,)])
        elif "EXPLAIN" in text:
            self.result = FakeResult([(line,) for line in self.plan.splitlines()])
        elif "pg_attribute" in text:
            self.result = FakeResult([self.tenant_column] if self.tenant_column else [None])
        elif "pg_hint_plan.enable_hint" in text:
            self.result = FakeResult([(self.hint_plan_setting,)])
        else:
            self.result = FakeResult([(7,), (3,)])
        return self.result

    def fetchone(self):
        return self.result.fetchone()

    def fetchall(self):
        return self.result.fetchall()

    def copy(self, statement):
        fake = FakeCopy(statement.as_string(None))
        self.copies.append(fake)
        return fake

    def close(self):
        pass


class FakeConn:
    def commit(self):
        pass

    def close(self):
        pass


VECTOR_PLAN = (
    "Limit\n  ->  Index Scan using pgvector_index on vdbbench_table_test\n        Order By: (embedding <=> $2)"
)
PKEY_PLAN = "Limit\n  ->  Sort\n        ->  Index Scan using vdbbench_table_test_pkey on vdbbench_table_test"
YB_VERSION = "PostgreSQL 15.12-YB-2.31.0.0-b0 on x86_64-pc-linux-gnu"


def make_db(monkeypatch, plan=VECTOR_PLAN, server_version=YB_VERSION, multitenant=True, **case_kwargs):
    cursor = FakeCursor(server_version, plan)
    monkeypatch.setattr(PgVector, "_create_connection", staticmethod(lambda **kwargs: (FakeConn(), cursor)))
    case_config = PgVectorHNSWConfig(
        metric_type="COSINE",
        m=16,
        ef_construction=64,
        ef_search=64,
        create_index_before_load=True,
        create_index_after_load=False,
        **case_kwargs,
    )
    kwargs = {"multitenant_tenant_labels": TENANTS} if multitenant else {}
    db = PgVector(
        dim=DIM,
        db_config={"connect_config": {}, "table_name": "vdbbench_table_test"},
        db_case_config=case_config,
        drop_old=True,
        **kwargs,
    )
    return db, cursor


@contextmanager
def opened(db, cursor):
    db.conn, db.cursor = FakeConn(), cursor
    yield


def statements(cursor, needle):
    return [text for text, _ in cursor.executed if needle in text]


def test_multitenant_table_is_hash_keyed_on_tenant_and_split(monkeypatch):
    _, cursor = make_db(monkeypatch, num_tablets=12)
    (create,) = statements(cursor, "CREATE TABLE")
    assert '"tenant_id" uuid' in create
    assert 'PRIMARY KEY (("tenant_id") HASH, "id" ASC)' in create
    assert "SPLIT INTO 12 TABLETS" in create


def test_text_tenant_uses_c_collation(monkeypatch):
    _, cursor = make_db(monkeypatch, tenant_column_type="text")
    (create,) = statements(cursor, "CREATE TABLE")
    assert '"tenant_id" text COLLATE "C"' in create


def test_split_is_skipped_on_postgres(monkeypatch):
    _, cursor = make_db(monkeypatch, server_version="PostgreSQL 17.2 on x86_64", num_tablets=12)
    (create,) = statements(cursor, "CREATE TABLE")
    assert "SPLIT" not in create
    assert 'PRIMARY KEY ("tenant_id", "id")' in create


def test_non_multitenant_table_is_unchanged_without_num_tablets(monkeypatch):
    _, cursor = make_db(monkeypatch, multitenant=False)
    (create,) = statements(cursor, "CREATE TABLE")
    assert '"id" BIGINT PRIMARY KEY' in create
    assert "tenant_id" not in create
    assert "SPLIT" not in create


def test_insert_writes_tenant_uuid(monkeypatch):
    db, cursor = make_db(monkeypatch)
    with opened(db, cursor):
        count, err = db.insert_embeddings(
            embeddings=[[0.1] * DIM, [0.2] * DIM],
            metadata=[0, 1],
            tenant_labels_data=["tenant_0000", "tenant_0001"],
        )
    assert (count, err) == (2, None)
    (copy,) = cursor.copies
    assert '("tenant_id", "id", "embedding")' in copy.statement
    assert copy.types == ["uuid", "bigint", "vector"]
    assert copy.rows[0][0] == uuid.uuid5(TENANT_UUID_NAMESPACE, "tenant_0000")
    assert copy.rows[1][0] == uuid.uuid5(TENANT_UUID_NAMESPACE, "tenant_0001")


def test_search_filters_on_tenant_with_index_hint(monkeypatch):
    db, cursor = make_db(monkeypatch)
    with opened(db, cursor):
        db.prepare_filter(non_filter)
        assert db.search_embedding([0.1] * DIM, k=10, tenant="tenant_0002") == [7, 3]
    text, params = cursor.executed[-1]
    assert text.startswith('/*+ IndexScan("vdbbench_table_test" "pgvector_index") */ SELECT "id"')
    assert 'WHERE tenant_id = %s ORDER BY "embedding"<=>' in text
    assert params[0] == uuid.uuid5(TENANT_UUID_NAMESPACE, "tenant_0002")
    assert params[2] == 10


def test_search_fails_when_plan_skips_vector_index(monkeypatch):
    db, cursor = make_db(monkeypatch, plan=PKEY_PLAN)
    with opened(db, cursor), pytest.raises(RuntimeError, match="does not use the vector index"):
        db.prepare_filter(non_filter)


def test_plan_failure_names_pg_hint_plan_when_not_loaded(monkeypatch):
    db, cursor = make_db(monkeypatch, plan=PKEY_PLAN, server_version="PostgreSQL 17.2 on x86_64")
    cursor.hint_plan_setting = None
    with opened(db, cursor), pytest.raises(RuntimeError, match="pg_hint_plan is not loaded"):
        db.prepare_filter(non_filter)


def test_tenant_column_type_mismatch_is_reported(monkeypatch):
    db, cursor = make_db(monkeypatch, tenant_column_type="text")
    with opened(db, cursor), pytest.raises(RuntimeError, match="has tenant_id uuid, but --tenant-column-type is text"):
        db.prepare_filter(non_filter)
    assert not statements(cursor, "EXPLAIN")


def test_text_tenant_column_needs_c_collation(monkeypatch):
    db, cursor = make_db(monkeypatch, tenant_column_type="text")
    cursor.tenant_column = ("text", "default")
    with opened(db, cursor), pytest.raises(RuntimeError, match="COLLATE default"):
        db.prepare_filter(non_filter)
    cursor.tenant_column = ("text", "C")
    with opened(db, cursor):
        db.prepare_filter(non_filter)


def test_missing_tenant_column_is_reported(monkeypatch):
    db, cursor = make_db(monkeypatch)
    cursor.tenant_column = None
    with opened(db, cursor), pytest.raises(RuntimeError, match="has no tenant_id column"):
        db.prepare_filter(non_filter)


def test_no_hint_or_plan_check_when_not_forced(monkeypatch):
    db, cursor = make_db(monkeypatch, plan=PKEY_PLAN, force_vector_index=False)
    with opened(db, cursor):
        db.prepare_filter(non_filter)
        db.search_embedding([0.1] * DIM, k=10, tenant="tenant_0000")
    assert not statements(cursor, "EXPLAIN")
    assert not statements(cursor, "/*+")


def test_exact_search_cannot_use_vector_index(monkeypatch):
    db, cursor = make_db(monkeypatch, plan=VECTOR_PLAN)
    with opened(db, cursor):
        db.prepare_filter(non_filter)
        cursor.plan = PKEY_PLAN
        assert db.exact_tenant_search([0.1] * DIM, 10, "tenant_0001") == [7, 3]
    text, params = cursor.executed[-1]
    # No hint: the OFFSET 0 subquery keeps the ORDER BY away from the vector index, which
    # also works on PostgreSQL without pg_hint_plan.
    assert text.startswith('SELECT "id" FROM (SELECT "id", "embedding" FROM public."vdbbench_table_test" ')
    assert "WHERE tenant_id = %s OFFSET 0) t ORDER BY" in text
    assert params[0] == uuid.uuid5(TENANT_UUID_NAMESPACE, "tenant_0001")


def test_exact_search_fails_when_plan_uses_vector_index(monkeypatch):
    db, cursor = make_db(monkeypatch, plan=VECTOR_PLAN)
    with opened(db, cursor):
        db.prepare_filter(non_filter)
        with pytest.raises(RuntimeError, match="uses the vector index"):
            db.exact_tenant_search([0.1] * DIM, 10, "tenant_0001")


class RecordingDB:
    """Stands in for a DB client; it returns the ids of the exact search as search results,
    with one id dropped for tenant_0001."""

    name = "Recording"

    def __init__(self):
        self.exact_calls = []
        self.search_calls = []

    def supports_payload_profile(self, payload_profile):
        return True

    @contextmanager
    def init(self):
        yield

    def prepare_filter(self, filters):
        pass

    def supports_exact_tenant_search(self):
        return True

    def exact_tenant_search(self, query, k, tenant):
        self.exact_calls.append(tenant)
        return [1, 2]

    def search_embedding(self, query, k, tenant=None):
        self.search_calls.append(tenant)
        return [1] if tenant == "tenant_0001" else [1, 2]


def test_serial_runner_computes_tenant_ground_truth():
    db = RecordingDB()
    test_data = [[0.1] * DIM for _ in range(20)]
    runner = SerialSearchRunner(
        db=db,
        test_data=test_data,
        ground_truth=None,
        k=2,
        tenant_labels=TENANTS,
        measure_recall=False,
    )
    recall, _, _, _ = runner.search((test_data, None))

    rng = random.Random(0)
    expected = [TENANTS[rng.randrange(len(TENANTS))] for _ in test_data]
    # Ground truth and the timed searches use the same tenant per query.
    assert db.exact_calls == expected
    assert db.search_calls == expected
    assert recall == pytest.approx(np.mean([0.5 if t == "tenant_0001" else 1.0 for t in expected]), abs=1e-4)


def test_tenant_uuid_namespace_is_stable():
    # Changing it breaks searches against tables loaded by an older client.
    assert pgvector_module.TENANT_UUID_NAMESPACE == uuid.UUID("6f1c6a52-3c1e-4b8e-9d0a-5b7f2f6c1a10")
