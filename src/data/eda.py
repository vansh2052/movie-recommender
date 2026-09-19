"""Exploratory data analysis plots, saved to reports/figures/.

Run after preprocess.py has produced data/processed/interactions.parquet
and movies.parquet.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CONFIG
from src.data.loader import GENRES, load_ratings

sns.set_theme(style="whitegrid")


def _figures_dir() -> Path:
    d = Path(CONFIG["paths"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_rating_distribution(ratings: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="rating", data=ratings, ax=ax, color="#4C72B0")
    ax.set_title("Rating distribution (all ratings)")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "rating_distribution.png", dpi=150)
    plt.close(fig)


def plot_ratings_per_user(ratings: pd.DataFrame, out_dir: Path):
    counts = ratings.groupby("user_id").size().sort_values(ascending=False).values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(counts) + 1), counts, color="#DD8452")
    ax.set_yscale("log")
    ax.set_xlabel("User rank (most active first)")
    ax.set_ylabel("Number of ratings (log scale)")
    ax.set_title("Ratings per user (long tail)")
    fig.tight_layout()
    fig.savefig(out_dir / "ratings_per_user.png", dpi=150)
    plt.close(fig)


def plot_ratings_per_movie(ratings: pd.DataFrame, out_dir: Path):
    counts = ratings.groupby("movie_id").size().sort_values(ascending=False).values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(counts) + 1), counts, color="#55A868")
    ax.set_yscale("log")
    ax.set_xlabel("Movie rank (most rated first)")
    ax.set_ylabel("Number of ratings (log scale)")
    ax.set_title("Ratings per movie (long tail)")
    fig.tight_layout()
    fig.savefig(out_dir / "ratings_per_movie.png", dpi=150)
    plt.close(fig)


def plot_genre_popularity(movies: pd.DataFrame, ratings: pd.DataFrame, out_dir: Path):
    merged = ratings.merge(movies[["movie_id"] + [f"genre_{g}" for g in GENRES]], on="movie_id")
    genre_counts = {g: int(merged[f"genre_{g}"].sum()) for g in GENRES}
    series = pd.Series(genre_counts).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=series.values, y=series.index, ax=ax, color="#C44E52")
    ax.set_xlabel("Number of ratings")
    ax.set_title("Genre popularity (by rating count)")
    fig.tight_layout()
    fig.savefig(out_dir / "genre_popularity.png", dpi=150)
    plt.close(fig)


def run():
    processed_dir = Path(CONFIG["paths"]["processed_dir"])
    movies = pd.read_parquet(processed_dir / "movies.parquet")
    ratings = load_ratings()

    out_dir = _figures_dir()
    plot_rating_distribution(ratings, out_dir)
    plot_ratings_per_user(ratings, out_dir)
    plot_ratings_per_movie(ratings, out_dir)
    plot_genre_popularity(movies, ratings, out_dir)
    print(f"Saved 4 figures to {out_dir}")


if __name__ == "__main__":
    run()
