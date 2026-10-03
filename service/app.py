from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from psycopg_pool import ConnectionPool

import khoj

from service.models import SearchRequest, SearchResponse
from service.search import InvalidQuery, UnknownOwner, run_search


def create_app(index: khoj.HnswIndex, pool: ConnectionPool, **fastapi_options: object) -> FastAPI:
    app = FastAPI(title="khoj", **fastapi_options)

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
            with pool.connection() as connection:
                return run_search(connection, index, request)
        except UnknownOwner as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except InvalidQuery as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def create_app_from_environment() -> FastAPI:
    index = khoj.HnswIndex.load(os.environ["KHOJ_INDEX_PATH"])
    pool = ConnectionPool(os.environ["KHOJ_DATABASE_URL"], open=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        pool.open(wait=True)
        try:
            yield
        finally:
            pool.close()

    return create_app(index, pool, lifespan=lifespan)
