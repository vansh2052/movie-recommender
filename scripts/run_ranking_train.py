"""Train the LightGBM LambdaRank ranker.

Usage: python -m scripts.run_ranking_train
"""

from src.ranking.train import train


if __name__ == "__main__":
    train()
