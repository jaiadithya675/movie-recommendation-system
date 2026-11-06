.PHONY: install data test lint evaluate report all app
install:
	pip install -r requirements-dev.txt
data:
	python scripts/download_data.py
test:
	python -m pytest -q
lint:
	ruff check .
evaluate:
	python -m movierec.cli evaluate --k 10
app:
	streamlit run app.py

report:
	python scripts/make_report.py

all:
	bash scripts/run_all.sh
