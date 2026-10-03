from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["pre", "post"]


class SearchRequest(BaseModel):
    owner_id: int
    vector: list[float] = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=1000)
    ef_search: int = Field(default=64, ge=1, le=10000)
    language: str | None = None
    since: datetime | None = None
    strategy: Strategy = "pre"
    post_widening: int = Field(default=10, ge=1, le=1000)


class SearchHit(BaseModel):
    vector_id: int
    distance: float


class SearchResponse(BaseModel):
    strategy: Strategy
    hits: list[SearchHit]
    matched_count: int
    latency_ms: float
