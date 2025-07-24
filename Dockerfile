# Use official Python image with slim variant
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set memory optimization for Python
ENV PYTHONMALLOC=malloc
ENV PYTHONMALLOCSTATS=1
ENV PYTHONHASHSEED=0

# Set work directory
WORKDIR /app

# Install system dependencies (removed Redis)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    poppler-utils \
    libreoffice && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with memory optimization
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "20"]