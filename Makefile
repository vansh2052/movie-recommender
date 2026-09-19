.PHONY: setup download-data eda test baselines train-retrieval eval-retrieval train-ranking eval-ranking api demo all

PYTHON := .venv/bin/python

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download-data:
	$(PYTHON) -m scripts.run_download

eda:
	$(PYTHON) -m scripts.run_eda

test:
	$(PYTHON) -m pytest tests/ -v

baselines:
	$(PYTHON) -m scripts.run_baselines

train-retrieval:
	$(PYTHON) -m scripts.run_retrieval_train

eval-retrieval:
	$(PYTHON) -m scripts.run_retrieval_eval

train-ranking:
	$(PYTHON) -m scripts.run_ranking_build
	$(PYTHON) -m scripts.run_ranking_train

eval-ranking:
	$(PYTHON) -m scripts.run_ranking_eval

api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8000

demo:
	$(PYTHON) -m streamlit run app_streamlit.py

all: download-data test eda baselines train-retrieval eval-retrieval train-ranking eval-ranking
