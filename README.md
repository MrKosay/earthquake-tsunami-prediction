# Earthquake & Tsunami Risk Prediction Platform

A comprehensive platform for predicting and visualizing earthquake and tsunami risks using advanced machine learning models and interactive visualizations. This system analyzes historical seismic data to predict magnitude, depth, tsunami likelihood, and potential economic impact of earthquake events.

## Features

- **Predictive Analytics**: Machine learning models for earthquake magnitude, depth, tsunami likelihood, and economic impact prediction
- **Interactive Visualizations**: Dynamic maps and charts showing global seismic activity
- **Historical Comparison**: Compare model predictions with actual historical events
- **Emergency Response Chatbot**: AI-powered assistant for emergency guidance
- **Economic Impact Analysis**: Estimated financial damage and recovery needs

## Data Resources

All required data files and visualizations can be downloaded from Google Drive:
[Download Data & Visualizations](https://drive.google.com/drive/folders/1hM9HqLkblsWh6ltcoMC234QRStADeMqV?usp=sharing)

After downloading, extract the contents into the project's root directory to populate the `/data` and `/visualizations` folders.

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Docker (optional, for containerized deployment)
- Git

### Dependencies

The platform requires several Python libraries for data processing, machine learning, and visualization:
- pandas, numpy, scikit-learn (data processing and ML)
- fastapi, uvicorn (API server)
- streamlit (frontend interface)
- matplotlib, seaborn, plotly (data visualization)
- openai (for chatbot functionality)

## Running the Application

### Method 1: Direct Installation (Without Docker)

#### On macOS/Linux:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/earthquake-tsunami-prediction.git
   cd earthquake-tsunami-prediction
   ```

2. **Set up environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "API_URL=http://localhost:8000" >> .env
   ```

4. **Process data**
   ```bash
   python process_data.py
   ```

5. **Start API server** (in a new terminal)
   ```bash
   cd api
   source ../.venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. **Start frontend** (in another terminal)
   ```bash
   cd frontend
   source ../.venv/bin/activate
   streamlit run app.py
   ```

#### On Windows:

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/earthquake-tsunami-prediction.git
   cd earthquake-tsunami-prediction
   ```

2. **Set up environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```powershell
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "API_URL=http://localhost:8000" >> .env
   ```

4. **Process data**
   ```powershell
   python process_data.py
   ```

5. **Start API server** (in a new terminal)
   ```powershell
   cd api
   .\.venv\Scripts\activate
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. **Start frontend** (in another terminal)
   ```powershell
   cd frontend
   .\.venv\Scripts\activate
   streamlit run app.py
   ```

### Method 2: Using Docker (Recommended for Production)

#### On macOS/Linux:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/earthquake-tsunami-prediction.git
   cd earthquake-tsunami-prediction
   ```

2. **Configure API key**
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Build and start containers**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - API: http://localhost:8000
   - Frontend: http://localhost:8501

#### On Windows:

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/earthquake-tsunami-prediction.git
   cd earthquake-tsunami-prediction
   ```

2. **Configure API key**
   ```powershell
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Build and start containers**
   ```powershell
   docker-compose up -d
   ```

4. **Access the application**
   - API: http://localhost:8000
   - Frontend: http://localhost:8501

### One-line Startup (macOS/Linux only)
For quick development startup:
```bash
./start-docker.sh
```

## Application Components

### API Server
The FastAPI backend provides prediction endpoints:
- `/predict`: Primary prediction endpoint
- `/health`: Health check endpoint
- `/docs`: API documentation (Swagger UI)

### Frontend
The Streamlit interface offers:
- Map visualization of earthquake risks
- Input controls for location and parameters
- Historical event comparisons
- Prediction results and confidence metrics

### Models
- **Earthquake Magnitude Prediction**: 98% accuracy
- **Earthquake Depth Prediction**: 85.7% accuracy
- **Tsunami Likelihood Prediction**: 47.8% accuracy (varies by region)
- **Economic Impact Estimation**: Based on damage function models

## Testing
Run the included test script to verify functionality:
```bash
python test_api.py
```

## 📖 Developer Notes

- The `.env` file contains API keys and should not be committed to version control
- Large data files are stored separately (see Google Drive link) and not in the repository
- The application uses CPU-based ML models; no GPU is required for inference

## Troubleshooting

- **API Connection Issues**: Ensure API server is running and accessible
- **Missing Data Files**: Download data from the provided Google Drive link
- **Docker Network Issues**: Check Docker network settings and port forwarding
- **OpenAI API Errors**: Verify your API key is correctly set in the .env file

## Project Structure

```
earthquake-tsunami-prediction/
├── api/                    # FastAPI backend
├── frontend/               # Streamlit frontend
├── chatbot/                # Emergency response chatbot
├── data/                   # Data directory
│   ├── processed/          # Processed datasets
├── models/                 # Trained ML models
├── visualizations/         # Generated visualizations
├── .env                    # Environment variables (create this)
├── docker-compose.yml      # Docker configuration
├── Dockerfile              # Docker build instructions
├── requirements.txt        # Python dependencies
├── start-docker.sh         # Startup script
└── README.md               # This documentation
```
