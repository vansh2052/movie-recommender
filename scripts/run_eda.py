"""Generate EDA figures into reports/figures/.

Usage: python -m scripts.run_eda
"""

from src.data import eda


if __name__ == "__main__":
    eda.run()
