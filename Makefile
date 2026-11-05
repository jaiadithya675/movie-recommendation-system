.PHONY: install data test lint evaluate app
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
