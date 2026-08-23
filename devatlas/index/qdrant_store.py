"""
devatlas.index.qdrant_store
===========================
One collection, three vector spaces, rich payload.

DESIGN DECISIONS (each is an interview answer):
- ONE collection, NOT collection-per-version. Qdrant applies payload
  filters DURING HNSW traversal, so version filtering stays fast, and
  cross-version questions ("what changed between 0.1 and 1.0?") need all
  versions in one searchable index.
- NAMED vectors 'dense_prose' and 'dense_code': the Gemini and Voyage
  embedding spaces are incompatible; each point carries the vector for its
  own space and queries are routed to the matching space.
- Sparse 'bm25' vector on every point: exact-identifier recall.
- Hybrid fusion with native RRF via the Query API (prefetch dense +
  prefetch sparse -> fusion). RRF fuses on RANK, not score, because cosine
  ([-1,1]) and BM25 (unbounded) scores are on incomparable scales.

VERIFIED: everything in this module runs against QdrantClient(':memory:')
in the test suite — collection creation, upsert, filtered hybrid queries.
The same code talks to a Docker Qdrant by changing the client URL.
"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from qdrant_client import QdrantClient, models

from devatlas.index.embeddings import BM25Encoder, DenseEncoder, EMBED_DIM
from devatlas.schema import Chunk

COLLECTION = "devatlas_langchain"

PROSE_VEC = "dense_prose"
CODE_VEC = "dense_code"
SPARSE_VEC = "bm25"


class QdrantStore:
    def __init__(self, client: Optional[QdrantClient] = None, url: str = "http://localhost:6333"):
        # ':memory:' for tests, Docker URL for real runs — same code path.
        self.client = client or QdrantClient(url=url)

    # -- schema ---------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        if recreate and self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        if self.client.collection_exists(COLLECTION):
            return
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                PROSE_VEC: models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
                CODE_VEC: models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VEC: models.SparseVectorParams(
                    # IDF handled by our encoder; Qdrant's modifier stays off.
                ),
            },
        )
        # Payload indexes make version/package/source_type filters cheap.
        for field, ftype in (
            ("package", models.PayloadSchemaType.KEYWORD),
            ("version", models.PayloadSchemaType.KEYWORD),
            ("source_type", models.PayloadSchemaType.KEYWORD),
            ("symbol", models.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(
                collection_name=COLLECTION, field_name=field, field_schema=ftype,
            )

    # -- writes ----------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        prose_encoder: DenseEncoder,
        code_encoder: DenseEncoder,
        bm25: BM25Encoder,
        batch_size: int = 64,
    ) -> int:
        """Routes each chunk to ONE dense space (by is_code) + sparse.

        A point only carries the named vector of its own space; Qdrant
        allows sparse points per named vector, so prose points simply have
        no 'dense_code' vector and vice versa.
        """
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            code_items = [c for c in batch if c.is_code]
            prose_items = [c for c in batch if not c.is_code]
            code_vecs = code_encoder.encode([c.embed_text for c in code_items]) if code_items else []
            prose_vecs = prose_encoder.encode([c.embed_text for c in prose_items]) if prose_items else []
            vec_by_id: dict[str, tuple[str, list[float]]] = {}
            for c, v in zip(code_items, code_vecs):
                vec_by_id[c.chunk_id] = (CODE_VEC, v)
            for c, v in zip(prose_items, prose_vecs):
                vec_by_id[c.chunk_id] = (PROSE_VEC, v)

            points = []
            for c in batch:
                space, dense = vec_by_id[c.chunk_id]
                idx, vals = bm25.encode_document(c.embed_text)
                points.append(models.PointStruct(
                    # Qdrant point IDs must be uuid/int; chunk_id already is.
                    id=c.chunk_id,
                    vector={
                        space: dense,
                        SPARSE_VEC: models.SparseVector(indices=idx, values=vals),
                    },
                    payload=c.model_dump(mode="json"),
                ))
            self.client.upsert(collection_name=COLLECTION, points=points)
            total += len(points)
        return total

    # -- reads -----------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        prose_encoder: DenseEncoder,
        code_encoder: DenseEncoder,
        bm25: BM25Encoder,
        *,
        version: Optional[str] = None,
        package: Optional[str] = None,
        source_types: Optional[list[str]] = None,
        limit: int = 12,
        prefetch_limit: int = 50,
    ) -> list[Chunk]:
        """Dense(prose) + dense(code) + sparse, fused with RRF, filtered.

        We prefetch from BOTH dense spaces: a question like "how do I make
        an agent" should surface doc prose AND the create_agent source.
        RRF then interleaves by rank across all three lists.
        """
        must: list[models.FieldCondition] = []
        if version:
            must.append(models.FieldCondition(key="version", match=models.MatchValue(value=version)))
        if package:
            must.append(models.FieldCondition(key="package", match=models.MatchValue(value=package)))
        if source_types:
            must.append(models.FieldCondition(key="source_type", match=models.MatchAny(any=source_types)))
        qfilter = models.Filter(must=must) if must else None

        sq_idx, sq_vals = bm25.encode_query(query)
        prefetch = [
            models.Prefetch(
                query=prose_encoder.encode_query(query), using=PROSE_VEC,
                filter=qfilter, limit=prefetch_limit,
            ),
            models.Prefetch(
                query=code_encoder.encode_query(query), using=CODE_VEC,
                filter=qfilter, limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sq_idx, values=sq_vals),
                using=SPARSE_VEC, filter=qfilter, limit=prefetch_limit,
            ),
        ]
        result = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [Chunk.model_validate(p.payload) for p in result.points]
