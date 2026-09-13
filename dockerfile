FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first for efficient caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy reference passes and worker script into container
COPY subdiv1_2.jpg .
COPY CitationLongitudeAO.jpg .
COPY handler.py .

CMD ["python", "-u", "handler.py"]