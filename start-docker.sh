#!/bin/bash

# Script to start the Earthquake & Tsunami Risk Prediction Platform with Docker

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker Desktop and try again."
  exit 1
fi

# Read API key from .env file
if [ -f .env ]; then
  source .env
  echo "Environment variables loaded from .env file."
else
  echo "Error: .env file not found."
  echo "Please create a .env file with your OpenAI API key:"
  echo "OPENAI_API_KEY=your_openai_api_key_here"
  exit 1
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY is not set in .env file."
  echo "Please set your OpenAI API key in the .env file."
  exit 1
fi

# Export the API key for docker-compose
export OPENAI_API_KEY

# Build and start the services
echo "Building and starting Docker containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Show status
echo ""
echo "Earthquake & Tsunami Risk Prediction Platform is starting..."
echo ""
echo "Services:"
echo "- API: http://localhost:8010"
echo "- Frontend: http://localhost:8501"
echo ""
echo "You can view the logs with: docker-compose logs -f"
echo "To stop the services: docker-compose down" 