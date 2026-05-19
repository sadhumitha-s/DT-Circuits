.PHONY: setup train dashboard test clean deploy

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

# Package and deploy to Hugging Face Spaces
deploy:
	@echo "1. Slicing trajectories to data/trajectories_demo.pt (with zero-hardcoded guardrail)..."
	@echo "=========================================================="
	@python3 -c ' \
		import torch, os; \
		data = torch.load("data/trajectories.pt", map_location="cpu", weights_only=False); \
		count = len(data); \
		torch.save(data[:count], "data/trajectories_demo.pt"); \
		size_mb = os.path.getsize("data/trajectories_demo.pt") / (1024*1024); \
		if size_mb >= 9.5: \
			avg_size = size_mb / count; \
			count = int(9.0 / avg_size); \
			while count > 0: \
				demo_data = data[:count]; \
				torch.save(demo_data, "data/trajectories_demo.pt"); \
				size_mb = os.path.getsize("data/trajectories_demo.pt") / (1024*1024); \
				if size_mb < 9.5: \
					break; \
				count -= 1; \
		print(f"Successfully packaged {count}/{len(data)} trajectories (Size: {size_mb:.2f} MB). Safely under 10MB limit."); \
	'
	@echo "Done."
	@echo ""
	@echo "=========================================================="
	@echo "2. Staging and committing deployment assets..."
	@echo "=========================================================="
	@git add data/trajectories_demo.pt models/mini_dt.pt artifacts/saes/ .gitignore Dockerfile docker-compose.yml Makefile scripts/deploy.sh src/dashboard/app.py .github/workflows/hf_sync.yml README.md .gitattributes
	@git commit -m "feat: redeploy fresh model weights and demo trajectories" || echo "No new changes to commit."
	@echo ""
	@echo "=========================================================="
	@echo "3. Pushing changes to Hugging Face Spaces ('hf' remote)..."
	@echo "=========================================================="
	@git push hf main || echo "Failed to push to 'hf' remote automatically. Please verify your Space git remote is named 'hf', or manually push to your target remote (e.g. 'git push origin main')."

# Remove artifacts and cached files
clean:
	rm -rf data/*.pt models/*.pt artifacts/saes/*.pt
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

