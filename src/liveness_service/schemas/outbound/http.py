"""JSON bodies of plain HTTP responses."""
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ErrorResponse(BaseModel):
    """Body of every 4xx/5xx this service returns (FastAPI's HTTPException format)."""
    detail: str


def error_responses(*codes):
    """OpenAPI `responses=` entry documenting ErrorResponse for each status code."""
    return {code: {"model": ErrorResponse} for code in codes}
