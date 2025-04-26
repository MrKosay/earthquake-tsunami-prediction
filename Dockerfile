FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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