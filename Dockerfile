# Use a CUDA-enabled base image with cuDNN support (Ubuntu)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y python3 python3-pip libgl1 libgtk2.0-dev pkg-config

# Install torch, torchvision, and torchaudio with CUDA support directly
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --break-system-packages

# Install remaining Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt --break-system-packages

# Uninstall opencv-python and install the headless version (for docker only)
RUN pip install opencv-python==4.11.0.86 --break-system-packages

# Copy the rest of project files
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Inference.py", "--server.port=8501", "--server.enableCORS=false"]
