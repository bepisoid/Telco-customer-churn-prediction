FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Startup command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]