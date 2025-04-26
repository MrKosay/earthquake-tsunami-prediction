import requests
import json
from datetime import datetime

def test_chat_endpoint():
    """Test the /chat endpoint of the API"""
    print("Testing chat endpoint...")
    
    # API URL - make sure this matches the URL where your API is running
    API_URL = "http://localhost:8010"
    
    # Message to send to the chatbot
    message = "What should I do during an earthquake?"
    
    try:
        # Make the request to the chat endpoint
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "prediction_id": None
            },
            timeout=30
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("Chat response received successfully:")
            print(result["response"])
            return True
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        return False

def test_predict_demo():
    """Test the /predict_demo endpoint which doesn't rely on ML models"""
    print("\nTesting predict_demo endpoint...")
    
    # API URL
    API_URL = "http://localhost:8010"
    
    # Location data for Tokyo
    location_data = {
        "latitude": 35.6895,
        "longitude": 139.6917,
        "date_time": datetime.now().isoformat()
    }
    
    try:
        # Make the request to the predict_demo endpoint
        response = requests.post(
            f"{API_URL}/predict_demo",
            json=location_data,
            timeout=30
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("Prediction received successfully:")
            print(f"Location: {result['location']['name']}")
            print(f"Earthquake likelihood: {result['earthquake_likelihood']}")
            print(f"Tsunami likelihood: {result['tsunami_likelihood']}")
            return True
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        return False

if __name__ == "__main__":
    # Test both endpoints
    chat_success = test_chat_endpoint()
    predict_success = test_predict_demo()
    
    if chat_success and predict_success:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed.") 