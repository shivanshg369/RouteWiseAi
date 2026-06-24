# Use official Python lightweight image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Add the /app directory to the PYTHONPATH to ensure imports work correctly
ENV PYTHONPATH=/app

# Install build dependencies required for compiling Python packages (like pandas, scikit-learn, bcrypt, psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY new_requirement.txt .

# Install Python requirements (timeout increased for ML packages)
RUN pip install --no-cache-dir --default-timeout=100 -r new_requirement.txt

# Copy the rest of the application
COPY . .

# Expose port (default FastAPI/uvicorn port is 8000)
EXPOSE 8000

# Command to run the application, listening on 0.0.0.0
# If deploying to a platform that passes $PORT, we can use that, otherwise default to 8000.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
