"""Pydantic response models for the FastAPI demo (src/api/main.py)."""

from typing import List, Optional

from pydantic import BaseModel


class MovieInfo(BaseModel):
    movie_id: int
    title: str
    genres: str


class MovieRecommendation(MovieInfo):
    score: Optional[float] = None


class RecommendResponse(BaseModel):
    user_id: int
    is_cold_start: bool
    recommendations: List[MovieRecommendation]


class HistoryResponse(BaseModel):
    user_id: int
    is_known_user: bool
    history: List[MovieInfo]
