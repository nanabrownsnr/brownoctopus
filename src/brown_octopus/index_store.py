import json
import io
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class IndexStore(Protocol):
    """Persistence boundary for capability-index snapshots."""

    def exists(self) -> bool: ...

    def load_tools(self) -> list[dict]: ...

    def load_embeddings(self): ...

    def load_metadata(self) -> dict: ...

    def save_snapshot_atomic(
        self,
        tools: list[dict],
        embeddings,
        metadata: dict,
    ) -> None: ...

    def reset(self) -> None: ...


def _active_index_path(index_path: str | Path) -> Path:
    """Resolve the atomically published snapshot, falling back to legacy files."""
    root = Path(index_path)
    pointer = root / "current.json"
    if pointer.exists():
        with pointer.open("r", encoding="utf-8") as file:
            snapshot = json.load(file)["snapshot"]
        return root / snapshot
    return root


def save_tools(
    index_path: str | Path,
    tools: list[dict],
) -> None:
    index_path = Path(index_path)
    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    tools_path = index_path / "tools.json"

    with tools_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            tools,
            file,
            indent=2,
        )


def load_tools(
    index_path: str | Path,
) -> list[dict]:
    tools_path = _active_index_path(index_path) / "tools.json"

    with tools_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_embeddings(
    index_path: str | Path,
    embeddings,
) -> None:
    index_path = Path(index_path)
    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = index_path / "embeddings.pt"

    torch.save(
        embeddings,
        embeddings_path,
    )


def load_embeddings(
    index_path: str | Path,
):
    embeddings_path = _active_index_path(index_path) / "embeddings.pt"

    return torch.load(
        embeddings_path,
        weights_only=True,
    )


def save_metadata(
    index_path: str | Path,
    metadata: dict,
) -> None:
    index_path = Path(index_path)
    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = index_path / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def load_metadata(
    index_path: str | Path,
) -> dict:
    metadata_path = _active_index_path(index_path) / "metadata.json"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_snapshot_atomic(
    index_path: str | Path,
    tools: list[dict],
    embeddings,
    metadata: dict,
) -> None:
    """Write a complete index snapshot before replacing the live files."""
    target = Path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshots = target / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot_name = f"snapshot-{uuid.uuid4().hex}"
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=snapshots))
    published = snapshots / snapshot_name
    try:
        save_tools(temporary, tools)
        save_embeddings(temporary, embeddings)
        save_metadata(temporary, metadata)
        os.replace(temporary, published)
        pointer = target / "current.json"
        pointer_temp = target / f".current-{uuid.uuid4().hex}.json"
        with pointer_temp.open("w", encoding="utf-8") as file:
            json.dump({"snapshot": f"snapshots/{snapshot_name}"}, file)
        os.replace(pointer_temp, pointer)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class LocalIndexStore:
    """The default versioned file-backed index store."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)

    def exists(self) -> bool:
        legacy = all(
            (self.index_path / filename).exists()
            for filename in ("tools.json", "embeddings.pt", "metadata.json")
        )
        return legacy or (self.index_path / "current.json").exists()

    def load_tools(self) -> list[dict]:
        return load_tools(self.index_path)

    def load_embeddings(self):
        return load_embeddings(self.index_path)

    def load_metadata(self) -> dict:
        return load_metadata(self.index_path)

    def save_snapshot_atomic(self, tools, embeddings, metadata) -> None:
        save_snapshot_atomic(self.index_path, tools, embeddings, metadata)

    def reset(self) -> None:
        if self.index_path.exists():
            shutil.rmtree(self.index_path)


def _serialize_embeddings(embeddings) -> bytes:
    buffer = io.BytesIO()
    torch.save(embeddings, buffer)
    return buffer.getvalue()


def _deserialize_embeddings(payload: bytes):
    return torch.load(io.BytesIO(payload), weights_only=True)


def _embedding_values(embedding) -> list[float]:
    """Convert a provider result into plain values for database adapters."""
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().tolist()
    elif hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if embedding and isinstance(embedding[0], list):
        embedding = embedding[0]
    return [float(value) for value in embedding]


def _postgres_vector_literal(embedding) -> str:
    return "[" + ",".join(str(value) for value in _embedding_values(embedding)) + "]"


class PostgresIndexStore:
    """Optional Postgres-backed snapshot store.

    Complete snapshots are stored centrally. If the server exposes pgvector,
    the store also supports native cosine-distance candidate search. Install
    the optional dependency with ``pip install 'brown-octopus[postgres]'``.
    """

    def __init__(
        self,
        connection_url: str,
        *,
        index_name: str = "default",
        connection_factory=None,
    ) -> None:
        self.connection_url = connection_url
        self.index_name = index_name
        self._connection_factory = connection_factory
        self.supports_native_vector_search = False

    def _connect(self):
        if self._connection_factory is not None:
            return self._connection_factory(self.connection_url)
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgresIndexStore requires the optional dependency. "
                "Install with `pip install 'brown-octopus[postgres]'`."
            ) from exc
        return psycopg.connect(self.connection_url)

    def _ensure_schema(self, connection) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS brown_octopus_index_snapshots (
                    index_name TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    tools JSONB NOT NULL,
                    embeddings BYTEA,
                    metadata JSONB NOT NULL,
                    PRIMARY KEY (index_name, snapshot_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS brown_octopus_index_current (
                    index_name TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL
                )
                """
            )
            self._ensure_vector_schema(cursor, connection)

    def _ensure_vector_schema(self, cursor, connection) -> None:
        """Enable pgvector when the configured Postgres supports it.

        Plain Postgres remains a valid snapshot store. Native vector search is
        enabled only when the server exposes the optional ``vector`` extension.
        """
        try:
            cursor.execute("SAVEPOINT brown_octopus_vector_support")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS brown_octopus_index_vectors (
                    index_name TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    tool JSONB NOT NULL,
                    embedding vector NOT NULL,
                    PRIMARY KEY (index_name, snapshot_id, capability_id)
                )
                """
            )
            cursor.execute("RELEASE SAVEPOINT brown_octopus_vector_support")
            self.supports_native_vector_search = True
        except Exception:
            cursor.execute("ROLLBACK TO SAVEPOINT brown_octopus_vector_support")
            cursor.execute("RELEASE SAVEPOINT brown_octopus_vector_support")
            self.supports_native_vector_search = False

    def _current_snapshot(self, connection) -> str | None:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT snapshot_id FROM brown_octopus_index_current "
                "WHERE index_name = %s",
                (self.index_name,),
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def exists(self) -> bool:
        with self._connect() as connection:
            self._ensure_schema(connection)
            return self._current_snapshot(connection) is not None

    def _load(self, field: str):
        with self._connect() as connection:
            snapshot_id = self._current_snapshot(connection)
            if snapshot_id is None:
                raise FileNotFoundError(f"No index snapshot named '{self.index_name}'.")
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {field} FROM brown_octopus_index_snapshots "
                    "WHERE index_name = %s AND snapshot_id = %s",
                    (self.index_name, snapshot_id),
                )
                row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError(f"Index snapshot '{snapshot_id}' is missing.")
        value = row[0]
        return value

    def load_tools(self) -> list[dict]:
        return self._load("tools")

    def load_embeddings(self):
        return _deserialize_embeddings(self._load("embeddings"))

    def load_metadata(self) -> dict:
        return self._load("metadata")

    def save_snapshot_atomic(self, tools, embeddings, metadata) -> None:
        snapshot_id = f"snapshot-{uuid.uuid4().hex}"
        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO brown_octopus_index_snapshots "
                    "(index_name, snapshot_id, tools, embeddings, metadata) "
                    "VALUES (%s, %s, %s::jsonb, %s, %s::jsonb)",
                    (
                        self.index_name,
                        snapshot_id,
                        json.dumps(tools),
                        _serialize_embeddings(embeddings) if embeddings is not None else None,
                        json.dumps(metadata),
                    ),
                )
                if self.supports_native_vector_search and embeddings is not None:
                    for tool, embedding in zip(tools, embeddings):
                        cursor.execute(
                            "INSERT INTO brown_octopus_index_vectors "
                            "(index_name, snapshot_id, capability_id, tool, embedding) "
                            "VALUES (%s, %s, %s, %s::jsonb, %s::vector)",
                            (
                                self.index_name,
                                snapshot_id,
                                str(tool.get("capability_id") or tool.get("name")),
                                json.dumps(tool),
                                _postgres_vector_literal(embedding),
                            ),
                        )
                cursor.execute(
                    "INSERT INTO brown_octopus_index_current (index_name, snapshot_id) "
                    "VALUES (%s, %s) ON CONFLICT (index_name) DO UPDATE "
                    "SET snapshot_id = EXCLUDED.snapshot_id",
                    (self.index_name, snapshot_id),
                )
            connection.commit()

    def reset(self) -> None:
        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                if self.supports_native_vector_search:
                    cursor.execute(
                        "DELETE FROM brown_octopus_index_vectors WHERE index_name = %s",
                        (self.index_name,),
                    )
                cursor.execute(
                    "DELETE FROM brown_octopus_index_snapshots WHERE index_name = %s",
                    (self.index_name,),
                )
                cursor.execute(
                    "DELETE FROM brown_octopus_index_current WHERE index_name = %s",
                    (self.index_name,),
                )
            connection.commit()

    def search_vectors(
        self,
        query_embedding,
        *,
        limit: int | None = None,
        allowed_mcp_urls=None,
    ) -> list[dict]:
        with self._connect() as connection:
            self._ensure_schema(connection)
            if not self.supports_native_vector_search:
                raise RuntimeError(
                    "Postgres native vector search requires the pgvector extension. "
                    "Use a pgvector-enabled server or configure local retrieval."
                )
            snapshot_id = self._current_snapshot(connection)
            if snapshot_id is None:
                return []
            clauses = ["index_name = %s", "snapshot_id = %s"]
            parameters: list[Any] = [self.index_name, snapshot_id]
            if allowed_mcp_urls is not None:
                urls = list(allowed_mcp_urls)
                if not urls:
                    return []
                clauses.append("tool->>'mcp_url' = ANY(%s)")
                parameters.append(urls)
            query_vector = _postgres_vector_literal(query_embedding)
            sql = (
                "SELECT tool, 1 - (embedding <=> %s::vector) AS score "
                "FROM brown_octopus_index_vectors WHERE "
                + " AND ".join(clauses)
                + " ORDER BY embedding <=> %s::vector"
            )
            # The query vector is used in both the score and ordering clauses.
            parameters = [query_vector, *parameters, query_vector]
            if limit is not None:
                sql += " LIMIT %s"
                parameters.append(limit)
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                rows = cursor.fetchall()
        return [
            {**tool, "rank": rank, "score": float(score)}
            for rank, (tool, score) in enumerate(rows, start=1)
        ]


class MongoIndexStore:
    """Optional MongoDB-backed snapshot store.

    Install the optional dependency with
    ``pip install 'brown-octopus[mongo]'``. MongoDB stores complete snapshots;
    Atlas Vector Search can be enabled with ``vector_search_index``.
    """

    def __init__(
        self,
        connection_url: str,
        *,
        database: str = "brown_octopus",
        collection: str = "index_snapshots",
        index_name: str = "default",
        vector_search_index: str | None = None,
        client_factory=None,
    ) -> None:
        self.connection_url = connection_url
        self.database_name = database
        self.collection_name = collection
        self.index_name = index_name
        self.vector_search_index = vector_search_index
        self.supports_native_vector_search = vector_search_index is not None
        self._client_factory = client_factory

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory(self.connection_url)
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError(
                "MongoIndexStore requires the optional dependency. "
                "Install with `pip install 'brown-octopus[mongo]'`."
            ) from exc
        return MongoClient(self.connection_url)

    def _collection(self):
        client = self._client()
        return client, client[self.database_name][self.collection_name]

    def _current(self, collection):
        return collection.find_one({"_id": f"current:{self.index_name}"})

    def exists(self) -> bool:
        client, collection = self._collection()
        try:
            return self._current(collection) is not None
        finally:
            client.close()

    def _load(self) -> dict[str, Any]:
        client, collection = self._collection()
        try:
            current = self._current(collection)
            if current is None:
                raise FileNotFoundError(f"No index snapshot named '{self.index_name}'.")
            snapshot = collection.find_one(
                {
                    "_id": f"{self.index_name}:{current['snapshot_id']}"
                }
            )
            if snapshot is None:
                raise FileNotFoundError("The active MongoDB index snapshot is missing.")
            return snapshot
        finally:
            client.close()

    def load_tools(self) -> list[dict]:
        return self._load()["tools"]

    def load_embeddings(self):
        return _deserialize_embeddings(self._load()["embeddings"])

    def load_metadata(self) -> dict:
        return self._load()["metadata"]

    def save_snapshot_atomic(self, tools, embeddings, metadata) -> None:
        client, collection = self._collection()
        snapshot_id = f"snapshot-{uuid.uuid4().hex}"
        try:
            collection.replace_one(
                {"_id": f"{self.index_name}:{snapshot_id}"},
                {
                    "_id": f"{self.index_name}:{snapshot_id}",
                    "index_name": self.index_name,
                    "snapshot_id": snapshot_id,
                    "tools": tools,
                    "embeddings": _serialize_embeddings(embeddings)
                    if embeddings is not None
                    else None,
                    "metadata": metadata,
                },
                upsert=True,
            )
            if self.supports_native_vector_search and embeddings is not None:
                for tool, embedding in zip(tools, embeddings):
                    capability = str(tool.get("capability_id") or tool.get("name"))
                    collection.replace_one(
                        {
                            "document_type": "vector",
                            "index_name": self.index_name,
                            "snapshot_id": snapshot_id,
                            "capability_id": capability,
                        },
                        {
                            "document_type": "vector",
                            "index_name": self.index_name,
                            "snapshot_id": snapshot_id,
                            "capability_id": capability,
                            "mcp_url": tool.get("mcp_url"),
                            "tool": tool,
                            "embedding": _embedding_values(embedding),
                        },
                        upsert=True,
                    )
            collection.replace_one(
                {"_id": f"current:{self.index_name}"},
                {
                    "_id": f"current:{self.index_name}",
                    "index_name": self.index_name,
                    "snapshot_id": snapshot_id,
                },
                upsert=True,
            )
        finally:
            client.close()

    def search_vectors(
        self,
        query_embedding,
        *,
        limit: int | None = None,
        allowed_mcp_urls=None,
    ) -> list[dict]:
        if not self.supports_native_vector_search:
            raise RuntimeError(
                "Mongo native vector search requires vector_search_index=. "
                "Configure a MongoDB Atlas Vector Search index or use local retrieval."
            )
        client, collection = self._collection()
        try:
            current = self._current(collection)
            if current is None:
                return []
            requested_limit = limit or 100
            vector_filter = {
                "document_type": "vector",
                "index_name": self.index_name,
                "snapshot_id": current["snapshot_id"],
            }
            if allowed_mcp_urls is not None:
                urls = list(allowed_mcp_urls)
                if not urls:
                    return []
                vector_filter["mcp_url"] = {"$in": urls}
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_search_index,
                        "path": "embedding",
                        "queryVector": _embedding_values(query_embedding),
                        "numCandidates": max(requested_limit * 10, 100),
                        "limit": requested_limit,
                        "filter": vector_filter,
                    }
                },
                {
                    "$project": {
                        "tool": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            rows = list(collection.aggregate(pipeline))
        finally:
            client.close()
        return [
            {**row["tool"], "rank": rank, "score": float(row["score"])}
            for rank, row in enumerate(rows, start=1)
        ]

    def reset(self) -> None:
        client, collection = self._collection()
        try:
            collection.delete_many({"index_name": self.index_name})
            collection.delete_one({"_id": f"current:{self.index_name}"})
        finally:
            client.close()
