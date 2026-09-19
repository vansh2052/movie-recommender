"""FastAPI demo app. Loads the recommender pipeline once at startup (not
per request) and exposes:
  GET /recommend/{user_id}?k=10  -> ranked titles/genres/scores
  GET /user/{user_id}/history?k=20 -> movies the user has liked

Run with: make api   (uvicorn src.api.main:app --reload --port 8000)
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.schemas import HistoryResponse, MovieInfo, MovieRecommendation, RecommendResponse
from src.pipeline import RecommenderPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movie_recommender_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models and data...")
    t0 = time.perf_counter()
    app.state.pipeline = RecommenderPipeline()
    logger.info(f"Pipeline ready in {time.perf_counter() - t0:.2f}s")
    yield


app = FastAPI(title="Movie Recommender API", lifespan=lifespan)


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, request: Request, k: int = 10):
    pipeline: RecommenderPipeline = request.app.state.pipeline
    t0 = time.perf_counter()
    is_known = pipeline.is_known_user(user_id)
    recs = pipeline.recommend(user_id, k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"GET /recommend/{user_id}?k={k} known_user={is_known} took {elapsed_ms:.1f}ms")
    return RecommendResponse(
        user_id=user_id,
        is_cold_start=not is_known,
        recommendations=[MovieRecommendation(**r) for r in recs],
    )


@app.get("/user/{user_id}/history", response_model=HistoryResponse)
def history(user_id: int, request: Request, k: int = 20):
    pipeline: RecommenderPipeline = request.app.state.pipeline
    t0 = time.perf_counter()
    is_known = pipeline.is_known_user(user_id)
    hist = pipeline.get_history(user_id, k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"GET /user/{user_id}/history?k={k} known_user={is_known} took {elapsed_ms:.1f}ms")
    return HistoryResponse(
        user_id=user_id,
        is_known_user=is_known,
        history=[MovieInfo(**h) for h in hist],
    )
