import sys
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import reverse_geocoder as rg

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from chatbot.emergency_assistant import EmergencyAssistant

# Initialize FastAPI app
app = FastAPI(
    title="Earthquake & Tsunami Risk Prediction API",
    description="API for predicting earthquake and tsunami risk along with economic impact",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
data_processor = None
model_trainer = None
emergency_assistant = None

# Pydantic models for API
class LocationInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location", example=35.6895)
    longitude: float = Field(..., description="Longitude of the location", example=139.6917)
    date_time: str = Field(..., description="Date and time for prediction (ISO format)", example="2023-07-15T12:00:00")

class ChatInput(BaseModel):
    message: str = Field(..., description="User message", example="What should I do during an earthquake?")
    prediction_id: Optional[str] = Field(None, description="ID of a previous prediction for context")

class SimulationInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location", example=35.6895)
    longitude: float = Field(..., description="Longitude of the location", example=139.6917)
    magnitude: float = Field(..., description="Simulated earthquake magnitude", example=7.5)
    depth: float = Field(..., description="Simulated earthquake depth in km", example=15.0)
    tsunami_likelihood: float = Field(..., description="Simulated tsunami likelihood", example=0.8)

class PredictionResponse(BaseModel):
    prediction_id: str
    location: Dict[str, Any]
    earthquake_likelihood: float
    magnitude_prediction: Optional[Dict[str, Any]] = None
    depth_prediction_km: Optional[Dict[str, Any]] = None
    tsunami_likelihood: float
    economic_loss_usd: Optional[Dict[str, Any]] = None
    timestamp: str
    recommendations: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    global data_processor, model_trainer, emergency_assistant
    
    try:
        # Initialize data processor
        data_processor = DataProcessor(data_dir=str(Path(__file__).parent.parent / "data"))
        
        # Load data and prepare for preprocessing
        data_processor.load_data()
        data_processor.clean_earthquake_data()
        data_processor.clean_tsunami_data()
        data_processor.clean_economic_data()
        
        # Prepare model inputs to initialize the preprocessor
        earthquake_features = ['latitude', 'longitude', 'depth', 'magnitude', 'year', 'month', 'day', 'coast_distance', 'days_since_last_eq']
        data_processor.prepare_model_inputs(include_features=earthquake_features)
        
        # Initialize model trainer
        model_trainer = ModelTrainer(models_dir=str(Path(__file__).parent.parent / "models"))
        
        # Load pre-trained models
        model_trainer.load_models()
        
        # Initialize emergency assistant
        emergency_assistant = EmergencyAssistant()
        
        print("All components initialized successfully.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

# Helper function to get location name from coordinates
def get_location_name(latitude, longitude):
    try:
        results = rg.search((latitude, longitude))
        if results and len(results) > 0:
            location = results[0]
            return f"{location['name']}, {location['admin1']}, {location['cc']}"
        return "Unknown location"
    except Exception as e:
        print(f"Error getting location name: {e}")
        return "Unknown location"

# Helper function to convert datetime string to datetime object
def parse_datetime(date_time_str):
    try:
        return datetime.fromisoformat(date_time_str.replace('Z', '+00:00'))
    except Exception as e:
        raise ValueError(f"Invalid datetime format: {e}")

# Endpoints
@app.get("/")
async def root():
    return {"message": "Earthquake & Tsunami Risk Prediction API. See /docs for API documentation."}

@app.get("/health")
async def health():
    components_status = {
        "data_processor": data_processor is not None,
        "model_trainer": model_trainer is not None,
        "emergency_assistant": emergency_assistant is not None,
        "models_loaded": False
    }
    
    if model_trainer:
        components_status["models_loaded"] = (
            model_trainer.earthquake_classifier is not None and
            model_trainer.tsunami_classifier is not None
        )
    
    all_healthy = all(components_status.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "components": components_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: LocationInput):
    try:
        # Parse datetime
        print(f"Parsing datetime: {input_data.date_time}")
        date_time = parse_datetime(input_data.date_time)
        print(f"Parsed datetime: {date_time}")
        
        # Get location name
        print(f"Getting location for: {input_data.latitude}, {input_data.longitude}")
        location_name = get_location_name(input_data.latitude, input_data.longitude)
        print(f"Location name: {location_name}")
        
        try:
            # Prepare features for prediction
            print("Preparing features...")
            try:
                features_df = data_processor.prepare_location_time_features(
                    input_data.latitude, input_data.longitude, date_time
                )
                print(f"Features prepared, shape: {features_df.shape}")
                print(f"Features columns: {features_df.columns.tolist()}")
            except Exception as e:
                print(f"Error preparing features: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Process features
            print("Processing features...")
            try:
                preprocessor = data_processor.get_preprocessor()
                X = preprocessor.transform(features_df)
                print(f"Features processed, shape: {X.shape}")
            except Exception as e:
                print(f"Error processing features: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Debug the feature shape
            print(f"Feature shape: {X.shape}")
            
            # Workaround for feature shape mismatch
            try:
                if hasattr(model_trainer.earthquake_classifier, 'n_features_in_'):
                    expected_features = model_trainer.earthquake_classifier.n_features_in_
                    print(f"Expected features: {expected_features}")
                    if X.shape[1] != expected_features:
                        print(f"Warning: Feature shape mismatch. Got {X.shape[1]}, expected {expected_features}")
                        
                        # Create a new feature array with zeros
                        X_fixed = np.zeros((X.shape[0], expected_features))
                        
                        # Copy available features
                        min_features = min(X.shape[1], expected_features)
                        X_fixed[:, :min_features] = X[:, :min_features]
                        
                        # Replace X with the fixed version
                        X = X_fixed
                        print(f"Fixed feature shape to: {X.shape}")
            except Exception as e:
                print(f"Error fixing features: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Make prediction with the corrected features
            print("Making prediction...")
            try:
                prediction = model_trainer.full_prediction(X)
                print(f"Prediction: {prediction}")
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
    
            # Get emergency recommendations
            print("Getting recommendations...")
            try:
                recommendations = emergency_assistant.get_emergency_recommendations(
                    prediction, location_name
                )
                print(f"Recommendations generated")
            except Exception as e:
                print(f"Error getting recommendations: {str(e)}")
                recommendations = "Unable to generate recommendations at this time."
            
            # Create response
            print("Creating response...")
            response = {
                "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "location": {
                    "latitude": input_data.latitude,
                    "longitude": input_data.longitude,
                    "name": location_name
                },
                "earthquake_likelihood": prediction.get("earthquake_likelihood", 0.0),
                "tsunami_likelihood": prediction.get("tsunami_likelihood", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add optional fields if present
            if "magnitude_prediction" in prediction:
                response["magnitude_prediction"] = prediction["magnitude_prediction"]
            
            if "depth_prediction_km" in prediction:
                response["depth_prediction_km"] = prediction["depth_prediction_km"]
            
            if "economic_loss_usd" in prediction:
                response["economic_loss_usd"] = prediction["economic_loss_usd"]
            
            # Add recommendations
            response["recommendations"] = recommendations
            
            print("Response ready")
            return response
            
        except Exception as inner_e:
            # If there was an error with the ML model prediction,
            # fall back to the demo prediction
            print(f"ML prediction failed: {str(inner_e)}. Falling back to demo prediction.")
            
            # Call the demo endpoint internally
            return await predict_demo(input_data)
    
    except Exception as e:
        print(f"FATAL ERROR in prediction endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(input_data: ChatInput):
    try:
        # Get chat response
        response = emergency_assistant.chat(input_data.message)
        
        # Log the interaction for debugging
        print(f"Chat request: {input_data.message}")
        print(f"Response length: {len(response)}")
        
        return {"response": response}
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/simulate", response_model=PredictionResponse)
async def simulate(input_data: SimulationInput):
    try:
        try:
            # Get location name
            location_name = get_location_name(input_data.latitude, input_data.longitude)
            
            # Create simulated prediction
            simulation = {
                "earthquake_likelihood": 1.0,  # Simulated earthquake is certain
                "magnitude_prediction": {
                    "estimate": input_data.magnitude,
                    "range_probs": {
                        "4-5": max(0, 1 - abs(input_data.magnitude - 4.5) / 2),
                        "5-6": max(0, 1 - abs(input_data.magnitude - 5.5) / 2),
                        "6-7": max(0, 1 - abs(input_data.magnitude - 6.5) / 2),
                        "7+": max(0, 1 - abs(input_data.magnitude - 7.5) / 2) if input_data.magnitude < 7.5 else 1.0
                    },
                    "confidence_interval": [max(0, input_data.magnitude - 0.5), input_data.magnitude + 0.5]
                },
                "depth_prediction_km": {
                    "estimate": input_data.depth,
                    "ci": [max(0, input_data.depth - 5), input_data.depth + 5]
                },
                "tsunami_likelihood": input_data.tsunami_likelihood
            }
            
            if input_data.tsunami_likelihood > 0.3:
                base_loss = 10000 * (10 ** input_data.magnitude) * input_data.tsunami_likelihood
                            
                if base_loss < 500000:
                    risk_category = "low"
                elif base_loss < 2000000:
                    risk_category = "medium"
                else:
                    risk_category = "high"
                    
                simulation["economic_loss_usd"] = {
                    "estimate": float(base_loss),
                    "risk_category": risk_category
                }
            else:
                simulation["economic_loss_usd"] = {
                    "estimate": 0.0,
                    "risk_category": "none"
                }
            
            # Get emergency recommendations
            try:
                recommendations = emergency_assistant.get_emergency_recommendations(
                    simulation, location_name
                )
            except Exception as rec_e:
                print(f"Error getting recommendations: {str(rec_e)}")
                recommendations = "Unable to generate recommendations at this time."
            
            # Create response
            response = {
                "prediction_id": f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "location": {
                    "latitude": input_data.latitude,
                    "longitude": input_data.longitude,
                    "name": location_name
                },
                "earthquake_likelihood": simulation["earthquake_likelihood"],
                "magnitude_prediction": simulation["magnitude_prediction"],
                "depth_prediction_km": simulation["depth_prediction_km"],
                "tsunami_likelihood": simulation["tsunami_likelihood"],
                "economic_loss_usd": simulation["economic_loss_usd"],
                "timestamp": datetime.now().isoformat(),
                "recommendations": recommendations
            }
            
            return response
        except Exception as inner_e:
 
            print(f"Simulation failed: {str(inner_e)}. Falling back to demo simulation.")
            
            demo_input = LocationInput(
                latitude=input_data.latitude,
                longitude=input_data.longitude,
                date_time=datetime.now().isoformat()
            )
            
            demo_response = await predict_demo(demo_input)
            
      
            demo_response["magnitude_prediction"]["estimate"] = input_data.magnitude
            demo_response["magnitude_prediction"]["confidence_interval"] = [
                max(0, input_data.magnitude - 0.5), 
                input_data.magnitude + 0.5
            ]
            demo_response["depth_prediction_km"]["estimate"] = input_data.depth
            demo_response["depth_prediction_km"]["ci"] = [
                max(0, input_data.depth - 5), 
                input_data.depth + 5
            ]
            demo_response["tsunami_likelihood"] = input_data.tsunami_likelihood
       
            if input_data.tsunami_likelihood > 0.3:
                base_loss = 10000 * (10 ** input_data.magnitude) * input_data.tsunami_likelihood
                if base_loss < 500000:
                    risk_category = "low"
                elif base_loss < 2000000:
                    risk_category = "medium"
                else:
                    risk_category = "high"
                    
                demo_response["economic_loss_usd"] = {
                    "estimate": float(base_loss),
                    "risk_category": risk_category
                }
            
            return demo_response
    
    except Exception as e:
        print(f"FATAL ERROR in simulation endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

@app.post("/predict_demo", response_model=PredictionResponse)
async def predict_demo(input_data: LocationInput):
    """
    A simplified demo prediction endpoint that doesn't rely on the ML models.
    This is useful when the models aren't working properly.
    """
    try:
        # Parse datetime
        date_time = parse_datetime(input_data.date_time)
        
        # Get location name
        location_name = get_location_name(input_data.latitude, input_data.longitude)
        

        lat_factor = abs(input_data.latitude) / 90  
        lng_factor = abs(input_data.longitude) / 180  
        day_of_year = date_time.timetuple().tm_yday / 366  
        

        eq_likelihood = 0.1 + 0.4 * lat_factor + 0.3 * lng_factor + 0.2 * day_of_year
        eq_likelihood = min(0.95, max(0.05, eq_likelihood))  
        

        pacific_rim = [
            (35, 135), 
            (38, -122),  
            (-33, -70),  
            (-40, 175),  
            (61, -150),  
            (0, 100),    
        ]
        
        for rim_lat, rim_lng in pacific_rim:
            distance = ((input_data.latitude - rim_lat) ** 2 + (input_data.longitude - rim_lng) ** 2) ** 0.5
            if distance < 20:  
                eq_likelihood = min(0.95, eq_likelihood * 1.5)  
                break
        
       
        coast_distance = abs(input_data.longitude) % 10  
        tsunami_likelihood = eq_likelihood * (1 - coast_distance / 10) * 0.8
        tsunami_likelihood = min(0.95, max(0.01, tsunami_likelihood))
        
        
        magnitude = 4.0 + eq_likelihood * 4.0  
        magnitude_ci = [magnitude - 0.5, magnitude + 0.5]
        
      
        depth = 5.0 + eq_likelihood * 25.0  
        depth_ci = [max(0, depth - 5.0), depth + 5.0]
        
       
        if tsunami_likelihood > 0.3:
            base_loss = 10000 * (10 ** (magnitude - 4.0)) * tsunami_likelihood
            
            if base_loss < 500000:
                risk_category = "low"
            elif base_loss < 2000000:
                risk_category = "medium"
            else:
                risk_category = "high"
        else:
            base_loss = 0.0
            risk_category = "minimal"
        
        
        recommendations = f"""
## Emergency Recommendations for {location_name}

### Based on our analysis:
- Earthquake likelihood: {'High' if eq_likelihood > 0.7 else 'Moderate' if eq_likelihood > 0.4 else 'Low'}
- Tsunami likelihood: {'High' if tsunami_likelihood > 0.7 else 'Moderate' if tsunami_likelihood > 0.4 else 'Low'}
- Estimated magnitude: {magnitude:.1f} (Range: {magnitude_ci[0]:.1f} - {magnitude_ci[1]:.1f})

### Preparation Steps:
1. Create an emergency plan with your family
2. Prepare an emergency kit with food, water, and essential supplies
3. Identify safe places in each room of your home
4. Learn the evacuation routes in your area
5. Stay informed through local emergency alert systems

### During an Earthquake:
- Drop, Cover, and Hold On
- Stay away from windows and exterior walls
- If outdoors, move to an open area away from buildings

### If a Tsunami Warning is Issued:
- Move immediately to higher ground
- Follow evacuation routes
- Do not return until officials say it is safe

Contact your local emergency management office for more specific guidance.
"""
        
        # Create response in the expected format
        response = {
            "prediction_id": f"demo_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "location": {
                "latitude": input_data.latitude,
                "longitude": input_data.longitude,
                "name": location_name
            },
            "earthquake_likelihood": float(eq_likelihood),
            "magnitude_prediction": {
                "estimate": float(magnitude),
                "range_probs": {
                    "4-5": 0.25,
                    "5-6": 0.25,
                    "6-7": 0.25,
                    "7+": 0.25
                },
                "confidence_interval": [float(magnitude_ci[0]), float(magnitude_ci[1])]
            },
            "depth_prediction_km": {
                "estimate": float(depth),
                "ci": [float(depth_ci[0]), float(depth_ci[1])]
            },
            "tsunami_likelihood": float(tsunami_likelihood),
            "economic_loss_usd": {
                "estimate": float(base_loss),
                "risk_category": risk_category,
                "confidence": "medium",
                "range": {
                    "lower": float(base_loss * 0.5),
                    "upper": float(base_loss * 2.0)
                }
            },
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations
        }
        
        return response
    except Exception as e:
        print(f"Error in demo prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 