"""Parse the raw MovieLens 1M .dat files into pandas DataFrames."""

import re
from pathlib import Path

import pandas as pd

from src.config import CONFIG

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _raw_dir() -> Path:
    return Path(CONFIG["paths"]["raw_dir"]) / "ml-1m"


def load_ratings(raw_dir: Path = None) -> pd.DataFrame:
    """UserID::MovieID::Rating::Timestamp"""
    raw_dir = raw_dir or _raw_dir()
    df = pd.read_csv(
        raw_dir / "ratings.dat",
        sep=CONFIG["data"]["separator"],
        engine="python",
        encoding=CONFIG["data"]["encoding"],
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return df


def load_movies(raw_dir: Path = None) -> pd.DataFrame:
    """MovieID::Title::Genres|Genres|...

    Parses the release year out of the title (e.g. "Toy Story (1995)")
    and expands genres into a multi-hot matrix.
    """
    raw_dir = raw_dir or _raw_dir()
    df = pd.read_csv(
        raw_dir / "movies.dat",
        sep=CONFIG["data"]["separator"],
        engine="python",
        encoding=CONFIG["data"]["encoding"],
        names=["movie_id", "title", "genres"],
    )

    def parse_year(title: str):
        m = _YEAR_RE.search(title.strip())
        return int(m.group(1)) if m else None

    df["year"] = df["title"].apply(parse_year)
    # Median-impute the handful of titles without a parseable year rather
    # than dropping them, so every movie still gets a valid year feature.
    if df["year"].isna().any():
        df["year"] = df["year"].fillna(df["year"].median()).astype(int)

    genre_lists = df["genres"].str.split("|")
    for g in GENRES:
        df[f"genre_{g}"] = genre_lists.apply(lambda gs, g=g: int(g in gs))

    return df


def load_users(raw_dir: Path = None) -> pd.DataFrame:
    """UserID::Gender::Age::Occupation::Zip-code"""
    raw_dir = raw_dir or _raw_dir()
    df = pd.read_csv(
        raw_dir / "users.dat",
        sep=CONFIG["data"]["separator"],
        engine="python",
        encoding=CONFIG["data"]["encoding"],
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )
    return df


def load_all(raw_dir: Path = None):
    raw_dir = raw_dir or _raw_dir()
    return load_ratings(raw_dir), load_movies(raw_dir), load_users(raw_dir)
