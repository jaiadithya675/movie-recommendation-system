.PHONY: install data tune train evaluate report notebook test lint smoke app docker all
install:
	pip install -r requirements-dev.txt
data:
	python scripts/download_data.py
tune:
	python -m movierec.cli tune
train:
	python -m movierec.cli train
evaluate:
	python -m movierec.cli evaluate
report:
	python scripts/make_report.py
notebook:
	python scripts/build_notebook.py
test:
	python -m pytest -q
lint:
	ruff check .
smoke:
	bash scripts/smoke_test.sh
app:
	streamlit run app.py
docker:
	docker build -t movie-recommender . && docker run -p 8501:8501 movie-recommender
all:
	bash scripts/run_all.sh
