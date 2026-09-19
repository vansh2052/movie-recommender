"""Evaluate the full retrieval+ranking pipeline on test.

Usage: python -m scripts.run_ranking_eval
"""

from src.ranking import evaluate


if __name__ == "__main__":
    evaluate.main()
