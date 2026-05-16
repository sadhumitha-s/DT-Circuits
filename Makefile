.PHONY: setup train dashboard test clean

# Setup environment
setup:
	pip install -r requirements.txt

# Run full pipeline: Harvesting -> DT Training -> SAE Training
train:
	python3 scripts/train_dt.py
	python3 scripts/train_sae.py

# Launch the explorer dashboard
dashboard:
	streamlit run src/dashboard/app.py

# Run unit tests
test:
	PYTHONPATH=. pytest tests/

# Remove artifacts and cached files
clean:
	rm -rf data/*.pt models/*.pt artifacts/saes/*.pt
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
