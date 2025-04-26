# Earthquake & Tsunami Risk Prediction Platform

A comprehensive platform for predicting and visualizing earthquake and tsunami risks using machine learning models and interactive visualizations.

## Features

- Real-time earthquake and tsunami risk prediction
- Interactive map visualization
- Historical event analysis and comparison
- Emergency response chatbot
- Detailed risk assessment and economic impact analysis

## Technical Stack

- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python FastAPI for prediction endpoints
- **ML Models**: Predictive models for earthquake magnitude, depth, and tsunami likelihood
- **Visualization**: Folium for maps, Plotly for interactive charts

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start the API server: `cd api && python main.py`
4. Run the frontend: `cd frontend && streamlit run app.py`

## Emergency Response Chatbot

The platform includes an offline-capable emergency response chatbot that provides immediate guidance for earthquake and tsunami situations, ensuring critical information is accessible even without internet connectivity.

## Project Overview

This project aims to develop an accurate and reliable tsunami prediction system using deep learning techniques. By analyzing various oceanic and atmospheric parameters, the system can predict potential tsunami events and their characteristics with improved lead time.

## Project Structure

```
tsunami-prediction/
│
├── config/                 # Configuration files
├── data/                   # Data directory
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   └── external/           # External data sources
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── features/           # Feature engineering
│   ├── models/             # Model definition and training
│   └── visualization/      # Visualization tools
├── tests/                  # Test cases
├── visualizations/         # Generated visualizations
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tsunami-prediction.git
cd tsunami-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
API_URL=http://localhost:8000
```

## Usage

1. Data preprocessing:
```bash
python scripts/preprocess_data.py
```

2. Model training:
```bash
python src/models/train_model.py
```

3. Run predictions:
```bash
python src/models/predict.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing API access
- Various oceanic and atmospheric data providers
- The scientific community for tsunami research 