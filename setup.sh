#!/bin/bash
# Setup script for Earthquake & Tsunami Risk Prediction Platform

# Exit on error
set -e

echo "Setting up Earthquake & Tsunami Risk Prediction Platform..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# Environment Variables" > .env
    echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
    echo "API_URL=http://localhost:8000" >> .env
    echo "Please update the .env file with your OpenAI API key."
fi

# Ensure all directories exist
echo "Creating directory structure..."
mkdir -p data models api frontend chatbot notebooks

echo "Setup complete!"
echo "-------------------------------------"
echo "Next steps:"
echo "1. Update the OpenAI API key in .env file"
echo "2. Train models: python notebooks/train_models.py"
echo "3. Start API: cd api && uvicorn main:app --reload"
echo "4. Start frontend: cd frontend && streamlit run app.py"
echo "   OR use Docker: docker-compose up"
echo "-------------------------------------" 