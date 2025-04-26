#!/bin/bash

# Tsunami AI Project Pipeline Runner
# This script runs the complete tsunami prediction pipeline:
# 1. Data processing
# 2. Model training
# 3. Evaluation
# 4. Visualization

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    echo -e "${BLUE}Loading environment variables from .env${NC}"
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create an .env file with required environment variables"
    exit 1
fi

# Function to run a command and check its status
run_step() {
    echo -e "${BLUE}Running: $1${NC}"
    if eval $1; then
        echo -e "${GREEN}✓ Success: $2${NC}"
    else
        echo -e "${RED}✗ Failed: $2${NC}"
        exit 1
    fi
    echo ""
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating virtual environment${NC}"
    source venv/bin/activate
fi

# 1. Data Processing
run_step "python scripts/process_data.py" "Data processing completed"

# 2. Model Training
echo -e "${BLUE}Starting model training...${NC}"
# Train tsunami classifier
run_step "python src/models/train.py --model classifier" "Tsunami classifier trained"
# Train magnitude predictor
run_step "python src/models/train.py --model magnitude" "Magnitude predictor trained"
# Train economic impact predictor
run_step "python src/models/train.py --model economic" "Economic impact predictor trained"

# 3. Model Evaluation
echo -e "${BLUE}Evaluating models...${NC}"
run_step "python src/models/evaluate.py --all" "Model evaluation completed"

# 4. Run Visualizations
echo -e "${BLUE}Generating visualizations...${NC}"
run_step "python visualize_all.py" "Visualizations generated"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${GREEN}Check the 'visualizations/' directory for results${NC}"
echo -e "${GREEN}================================================${NC}"

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi 