FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for compilation and Gymnasium rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first to cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models data artifacts/saes

# Copy trained model weights and SAE features
COPY models/mini_dt.pt ./models/mini_dt.pt
COPY artifacts/saes/ ./artifacts/saes/

# Bake in the lightweight demo trajectories as the default dataset
COPY data/trajectories_demo.pt ./data/trajectories.pt

# Copy codebase
COPY src/ ./src/

# Expose default Streamlit port
EXPOSE 8501

# Streamlit configurations for production/cloud environments
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start the dashboard application
CMD ["streamlit", "run", "src/dashboard/app.py"]
