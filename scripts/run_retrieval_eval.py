"""Evaluate the trained two-tower + FAISS retrieval pipeline.

Usage: python -m scripts.run_retrieval_eval
"""

from src.retrieval import evaluate


if __name__ == "__main__":
    evaluate.main()
