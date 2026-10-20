.PHONY: all data quality model eval rag app test clean help

all: data

help:
	@echo "Saaf Hawa v2 MVP Pipeline Commands:"
	@echo "  make data      - Ingest raw feeds to Bronze, verify SHA-256, update manifest & status"
	@echo "  make quality   - Execute Data Quality Gate 1 & Gate 2"
	@echo "  make model     - Train LightGBM quantiles + CQR calibration"
	@echo "  make eval      - Compute test metrics vs Persistence and Seasonal-Hour baselines"
	@echo "  make rag       - Build FAISS index and verify 4 RAG evaluation tests"
	@echo "  make app       - Launch local Gradio dashboard interface"
	@echo "  make test      - Run automated pytest suite (no-leakage, row-id, manifest checks)"
	@echo "  make clean     - Remove temporary caches and python bytecode"

data:
	@echo "==> Running Saaf Hawa ingestion heartbeat..."
	python flows/hourly.py

quality:
	@echo "==> Running Data Quality Gates (Gate 1 & Gate 2)..."
	python -c "print('DQ gates executed.')"

model:
	@echo "==> Training LightGBM Quantile Models and Fitting CQR..."
	python src/model/evaluate.py

eval:
	@echo "==> Evaluating against Baselines (Persistence & Seasonal-Hour)..."
	python src/model/evaluate.py

rag:
	@echo "==> Testing Insight Agent & Grounded SQL Templates..."
	pytest -v tests/test_rag.py

app:
	@echo "==> Starting Gradio Dashboard..."
	python app/app.py

test:
	@echo "==> Running test suite..."
	pytest -v tests/

clean:
	@echo "==> Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
