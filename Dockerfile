FROM python:3.10-slim

# Install system dependencies for GDAL, Cartopy and other geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-bin \
    libgeos-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV GDAL_VERSION=3.6.2
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt requirements.txt

# Install packages with increased timeout, retries, and installing scipy separately with extended timeout
RUN pip install --no-cache-dir --timeout=300 --retries=10 scipy && \
    pip install --no-cache-dir --timeout=180 --retries=6 -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose ports for API and frontend
EXPOSE 8000
EXPOSE 8501

# Create entry point script
RUN echo '#!/bin/bash \n\
if [ "$1" = "api" ]; then \n\
    cd /app/api \n\
    uvicorn main:app --host 0.0.0.0 --port 8000 \n\
elif [ "$1" = "frontend" ]; then \n\
    cd /app/frontend \n\
    streamlit run app.py \n\
else \n\
    echo "Usage: docker run [options] IMAGE [api|frontend]" \n\
    exit 1 \n\
fi' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"] 