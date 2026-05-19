#!/bin/bash
set -e

# DT-Explorer Automated Deployment Script
# This script handles the raw bash workflow for solo researchers to update their hosted web app.

echo "=========================================================="
# 1. Slice heavy local trajectories into a lightweight demo set (zero-hardcoded dynamic scaling)
echo "1. Slicing local trajectories down to data/trajectories_demo.pt..."
echo "=========================================================="
python3 -c '
import torch, os
data = torch.load("data/trajectories.pt", map_location="cpu", weights_only=False)
count = len(data)
# Try full dataset first
torch.save(data[:count], "data/trajectories_demo.pt")
size_mb = os.path.getsize("data/trajectories_demo.pt") / (1024*1024)

if size_mb >= 9.5:
    # Calculate average size per trajectory and estimate safe capacity
    avg_size = size_mb / count
    count = int(9.0 / avg_size) # Aim for ~9.0 MB to be safe
    
    # Verify and make minor adjustments if needed
    while count > 0:
        demo_data = data[:count]
        torch.save(demo_data, "data/trajectories_demo.pt")
        size_mb = os.path.getsize("data/trajectories_demo.pt") / (1024*1024)
        if size_mb < 9.5:
            break
        count -= 1

print(f"Successfully packaged {count}/{len(data)} trajectories (Size: {size_mb:.2f} MB). Safely under 10MB limit.")
'
echo "Done."
echo ""

echo "=========================================================="
# 2. Stage model weights, SAE checkpoints, and configuration files
echo "2. Staging deployment files in Git..."
echo "=========================================================="
git add data/trajectories_demo.pt models/mini_dt.pt artifacts/saes/ .gitignore Dockerfile docker-compose.yml Makefile scripts/deploy.sh src/dashboard/app.py .github/workflows/hf_sync.yml
echo "Staged."
echo ""

echo "=========================================================="
# 3. Commit changes locally
echo "3. Committing staged changes..."
echo "=========================================================="
git commit -m "feat: redeploy fresh model weights and demo trajectories" || echo "No new changes to commit."
echo ""

echo "=========================================================="
# 4. Push to GitHub (to trigger auto-sync to Hugging Face Space)
echo "4. Pushing to GitHub (origin main)..."
echo "=========================================================="
git push origin main
echo ""
echo "Deployment successful! Check your Hugging Face Space or GitHub repository actions for the build status."
