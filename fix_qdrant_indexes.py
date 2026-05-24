"""
fix_qdrant_indexes.py

One-time script to create payload indexes on the Qdrant collection.

Qdrant Cloud requires explicit payload indexes on any field used in filters.
During ingestion, payloads were stored but indexes were not created,
causing 400 errors on every filtered vector search.

Fields indexed:
  source_type  (keyword) — used by source_filter in metadata_filter.py
  chapter_num  (integer) — used by chapter_filter in metadata_filter.py
  chunk_type   (keyword) — used by equation query type filter

Run once:
  python fix_qdrant_indexes.py

Safe to re-run — Qdrant skips creation if index already exists.
"""

from dotenv import load_dotenv
load_dotenv()

from app.config.settings import get_settings
from app.utils.logger import get_logger

log      = get_logger(__name__)
settings = get_settings()


def create_indexes() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PayloadSchemaType

    log.info(
        "connecting_to_qdrant",
        url=settings.qdrant_url_with_port,
        collection=settings.qdrant_collection_name,
    )

    client = QdrantClient(
        url=settings.qdrant_url_with_port,
        api_key=settings.qdrant_api_key,
        timeout=30,
    )

    # Verify collection exists
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name not in collections:
        log.error(
            "collection_not_found",
            collection=settings.qdrant_collection_name,
            available=collections,
        )
        return

    info = client.get_collection(settings.qdrant_collection_name)
    log.info("collection_found", vectors=info.vectors_count)

    indexes_to_create = [
        # (field_name, schema_type, description)
        ("source_type",  PayloadSchemaType.KEYWORD, "thesis vs paper filter"),
        ("chapter_num",  PayloadSchemaType.INTEGER,  "chapter-specific queries"),
        ("chunk_type",   PayloadSchemaType.KEYWORD, "equation chunk filter"),
        ("paper_year",   PayloadSchemaType.INTEGER,  "future: year-range filtering"),
    ]

    for field_name, schema_type, description in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=settings.qdrant_collection_name,
                field_name=field_name,
                field_schema=schema_type,
                wait=True,
            )
            log.info(
                "index_created",
                field=field_name,
                type=schema_type.value,
                note=description,
            )
        except Exception as exc:
            # Qdrant returns an error if index already exists — safe to ignore
            if "already exists" in str(exc).lower() or "conflict" in str(exc).lower():
                log.info("index_already_exists", field=field_name)
            else:
                log.error("index_creation_failed", field=field_name, error=str(exc))

    log.info("all_indexes_done")

    # Verify by checking collection info
    info = client.get_collection(settings.qdrant_collection_name)
    log.info(
        "collection_status",
        vectors=info.vectors_count,
        status=str(info.status),
    )
    print("\n✓ Payload indexes created. Vector search with filters will now work.")
    print("  Restart uvicorn and test again.")


if __name__ == "__main__":
    create_indexes()