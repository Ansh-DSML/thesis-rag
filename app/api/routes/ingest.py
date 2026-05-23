# app/api/routes/ingest.py
from fastapi import APIRouter, HTTPException
from app.utils.logger import get_logger

ingest_router = APIRouter(prefix="/ingest", tags=["ingest"])
log = get_logger("routes.ingest")

@ingest_router.post(
    "/trigger",
    summary="Trigger ingestion pipeline (protected)",
    description=(
        "Re-runs the ingestion pipeline. "
        "Protected by X-API-Key header (see APIKeyMiddleware). "
        "For production use only — run manually via script for dev."
    ),
)
async def trigger_ingestion() -> dict:
    """
    POST /ingest/trigger

    In production this would launch run_ingestion.py as a subprocess.
    During development, run the script directly instead:
        python scripts/run_ingestion.py
    """
    log.info("ingestion_trigger_received")
    return {
        "status": "accepted",
        "message": (
            "Ingestion trigger received. "
            "For development, run: python scripts/run_ingestion.py"
        ),
    }