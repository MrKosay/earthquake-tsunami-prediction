import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import os
from dotenv import load_dotenv
import examples  # Import our examples module

# Load environment variables
load_dotenv()

# API configuration - use environment variable with fallback
API_URL = os.getenv("API_URL", "http://localhost:8010")
print(f"Using API URL: {API_URL}")

# Page configuration
st.set_page_config(
    page_title="Earthquake & Tsunami Risk Prediction",
    page_icon="",
    layout="wide"
)

# Functions for API interaction
def get_prediction(latitude, longitude, date_time):
    """Get prediction from API based on location and time"""
    try:
       
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "latitude": latitude,
                "longitude": longitude,
                "date_time": date_time.isoformat()
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
           
            st.warning("ML model prediction failed, using demo prediction instead.")
            demo_response = requests.post(
                f"{API_URL}/predict_demo",
                json={
                    "latitude": latitude,
                    "longitude": longitude,
                    "date_time": date_time.isoformat()
                },
                timeout=30
            )
            if demo_response.status_code == 200:
                return demo_response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_simulation(latitude, longitude, magnitude, depth, tsunami_likelihood):
    """Get simulation from API based on custom parameters"""
    try:
        
        response = requests.post(
            f"{API_URL}/simulate",
            json={
                "latitude": latitude,
                "longitude": longitude,
                "magnitude": magnitude,
                "depth": depth,
                "tsunami_likelihood": tsunami_likelihood
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
           
            st.warning("ML model simulation failed, using demo simulation instead.")
            
            demo_response = requests.post(
                f"{API_URL}/predict_demo",
                json={
                    "latitude": latitude,
                    "longitude": longitude,
                    "date_time": datetime.now().isoformat()
                },
                timeout=30
            )
            
            if demo_response.status_code == 200:
                
                result = demo_response.json()
                result["magnitude_prediction"]["estimate"] = magnitude
                result["magnitude_prediction"]["confidence_interval"] = [magnitude - 0.5, magnitude + 0.5]
                result["depth_prediction_km"]["estimate"] = depth
                result["depth_prediction_km"]["ci"] = [max(0, depth - 5), depth + 5]
                result["tsunami_likelihood"] = tsunami_likelihood
                
                
                if tsunami_likelihood > 0.3:
                    base_loss = 10000 * (10 ** magnitude) * tsunami_likelihood
                    if base_loss < 500000:
                        risk_category = "low"
                    elif base_loss < 2000000:
                        risk_category = "medium"
                    else:
                        risk_category = "high"
                    
                    result["economic_loss_usd"] = {
                        "estimate": float(base_loss),
                        "risk_category": risk_category,
                        "confidence": "medium",
                        "range": {
                            "lower": float(base_loss * 0.5),
                            "upper": float(base_loss * 2.0)
                        }
                    }
                
                return result
            else:
                st.error(f"API Error: {demo_response.status_code} - {demo_response.text}")
                return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def chat_with_assistant(message, prediction_id=None):
    """Chat with the emergency assistant"""
    try:
        # Debug logging
        print(f"Sending chat request to API: {message}")
        print(f"Using API URL: {API_URL}/chat")
        
        # Prepare the request payload
        payload = {
            "message": message
        }
        
        # Add prediction_id if available
        if prediction_id:
            payload["prediction_id"] = prediction_id
            
        # Make the request
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=30
        )
        
        # Debug logging
        print(f"API Response status: {response.status_code}")
        print(f"API Response headers: {response.headers}")
        
        if response.status_code == 200:
            response_data = response.json()
            # Debug logging
            print(f"API Response content: {response_data}")
            return response_data["response"]
        else:
            error_msg = f"Chat API Error: {response.status_code}"
            if hasattr(response, 'text'):
                try:
                    error_data = response.json()
                    if 'detail' in error_data:
                        error_msg = f"Error: {error_data['detail']}"
                except:
                    error_msg = f"Error: {response.text[:100]}..."
            
            st.error(error_msg)
            return "Sorry, I'm having trouble connecting to the server. Please try again later."
    except Exception as e:
        error_message = str(e)
        st.error(f"Error connecting to Chat API: {error_message}")
        
       
        if "openai.ChatCompletion" in error_message and "no longer supported" in error_message:
            return ("I'm experiencing technical difficulties with my AI service. "
                   "Please contact support and mention that the OpenAI API needs to be updated to the latest version.")
        
        return "Sorry, I'm having trouble connecting to the server. Please try again later."

# Visualization functions
def create_map(latitude=35.6895, longitude=139.6917, zoom_start=4):
    """Create folium map for location selection"""
    m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start)
    
    # Add marker at current location
    folium.Marker(
        [latitude, longitude],
        popup="Selected Location",
        tooltip="Selected Location",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_risk_gauge(value, title):
    """Create a gauge chart for displaying risk levels"""
    # Define color based on value
    if value < 0.3:
        color = "green"
    elif value < 0.7:
        color = "orange"
    else:
        color = "red"
        
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "lightyellow"},
                {'range': [0.7, 1], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=200)
    return fig

def load_example(example_name):
    """Load example location data"""
    example = examples.get_example_by_name(example_name)
    if example:
        return example
    return None

# Main app layout
def main():
    st.title("🌊 Earthquake & Tsunami Risk Prediction Platform")
    
    # Sidebar for input options
    st.sidebar.title("Input Options")
    
    # Initialize session state for persistent values
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "Map Selection"
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    
    # Choose input method
    selected_method = st.sidebar.radio(
        "Select Input Method",
        ["Map Selection", "Coordinates Entry", "Simulation Mode", "Example Locations"],
        index=["Map Selection", "Coordinates Entry", "Simulation Mode", "Example Locations"].index(st.session_state.input_method)
    )
    
    # Update session state if input method changes
    if selected_method != st.session_state.input_method:
        st.session_state.input_method = selected_method
    
    # Initialize prediction data
    prediction = st.session_state.prediction
    
    # Map Selection input
    if st.session_state.input_method == "Map Selection":
        st.subheader("Select Location on Map")
        
        # Default map centered on Japan
        default_lat, default_lng = 35.6895, 139.6917
        
        # Initialize session state for clicked coordinates if not already set
        if 'clicked_lat' not in st.session_state:
            st.session_state.clicked_lat = default_lat
        if 'clicked_lng' not in st.session_state:
            st.session_state.clicked_lng = default_lng
            
        # Define callback for map click
        def handle_map_click(lat, lng):
            st.session_state.clicked_lat = lat
            st.session_state.clicked_lng = lng
            st.experimental_rerun()
        
        # Create map with click handler
        m = create_map(st.session_state.clicked_lat, st.session_state.clicked_lng)
        
        # Add click handler using custom JavaScript
        m.add_child(folium.Element("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                const map = document.querySelector('.folium-map');
                if (!map) return;
                
                const leafletMap = map._leaflet_map;
                if (!leafletMap) return;
                
                leafletMap.on('click', function(e) {
                    const lat = e.latlng.lat.toFixed(4);
                    const lng = e.latlng.lng.toFixed(4);
                    
                    // Send data to Streamlit
                    const data = {
                        lat: parseFloat(lat),
                        lng: parseFloat(lng)
                    };
                    
                    // Use Streamlit's component communication
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: data
                    }, '*');
                });
            }, 1000); // Give time for the map to initialize
        });
        </script>
        """))
        
        # Display map with custom component to receive click data
        from streamlit_folium import st_folium
        map_data = st_folium(m, width=800, height=500, key="interactive_map")
        
        # Handle click data from the map
        if map_data and 'last_clicked' in map_data:
            clicked = map_data['last_clicked']
            if clicked:
                st.session_state.clicked_lat = clicked['lat']
                st.session_state.clicked_lng = clicked['lng']
        
        # User inputs coordinates after clicking on map
        st.info("Click on the map to automatically set the location coordinates.")
        
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=st.session_state.clicked_lat, format="%.4f", key="lat_input")
            # Update session state when user changes the value
            if st.session_state.clicked_lat != lat:
                st.session_state.clicked_lat = lat
        with col2:
            lng = st.number_input("Longitude", value=st.session_state.clicked_lng, format="%.4f", key="lng_input")
            # Update session state when user changes the value
            if st.session_state.clicked_lng != lng:
                st.session_state.clicked_lng = lng
        
        # Date and time inputs
        col1, col2 = st.columns(2)
        with col1:
            date_time = st.date_input("Select Date", value=datetime.now())
        with col2:
            time = st.time_input("Select Time", value=datetime.now().time())
        
        # Combine date and time
        prediction_datetime = datetime.combine(date_time, time)
        
        # Submit button
        if st.button("Get Prediction"):
            with st.spinner("Generating prediction..."):
                prediction = get_prediction(lat, lng, prediction_datetime)
    
    # Coordinates Entry input
    elif st.session_state.input_method == "Coordinates Entry":
        st.subheader("Enter Location Coordinates")
        
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=35.6895, format="%.4f")
        with col2:
            lng = st.number_input("Longitude", value=139.6917, format="%.4f")
        
        # Date and time inputs
        col1, col2 = st.columns(2)
        with col1:
            date_time = st.date_input("Select Date", value=datetime.now())
        with col2:
            time = st.time_input("Select Time", value=datetime.now().time())
        
        # Combine date and time
        prediction_datetime = datetime.combine(date_time, time)
        
        # Submit button
        if st.button("Get Prediction"):
            with st.spinner("Generating prediction..."):
                prediction = get_prediction(lat, lng, prediction_datetime)
    
   
    elif st.session_state.input_method == "Simulation Mode":
        st.subheader("Simulation Mode")
        st.info("Use this mode to simulate custom earthquake and tsunami scenarios.")
        
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=35.6895, format="%.4f")
            magnitude = st.slider("Earthquake Magnitude", min_value=4.0, max_value=9.5, value=7.0, step=0.1)
        with col2:
            lng = st.number_input("Longitude", value=139.6917, format="%.4f")
            depth = st.slider("Earthquake Depth (km)", min_value=5.0, max_value=100.0, value=15.0, step=1.0)
        
        tsunami_likelihood = st.slider("Tsunami Likelihood", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        # Submit button
        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                prediction = get_simulation(lat, lng, magnitude, depth, tsunami_likelihood)

    
    elif st.session_state.input_method == "Example Locations":
        st.subheader("Historical Earthquake & Tsunami Events")
        st.info("Select a historical earthquake or tsunami event to test our prediction model against known events")
        
       
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Major Tsunami Events")
            tsunami_names = [ex["name"] for ex in examples.tsunami_examples]
            selected_tsunami = st.selectbox("Select tsunami event", tsunami_names)
            
            # Show description
            tsunami_example = examples.get_example_by_name(selected_tsunami)
            if tsunami_example:
                st.write(tsunami_example["description"])
                
                if st.button("Predict for this Event", key=f"tsunami_predict_{selected_tsunami}"):
                    with st.spinner(f"Running prediction model for {tsunami_example['name']}..."):
                        # Use simulation parameters but present as model prediction
                        if "simulation" in tsunami_example:
                            sim = tsunami_example["simulation"]
                            prediction = get_simulation(
                                tsunami_example["latitude"],
                                tsunami_example["longitude"],
                                sim["magnitude"],
                                sim["depth"],
                                sim["tsunami_likelihood"]
                            )
                        else:
                            prediction = get_prediction(
                                tsunami_example["latitude"],
                                tsunami_example["longitude"],
                                datetime.now()
                            )
                        
                        st.session_state.current_example = tsunami_example
        
        with col2:
            st.subheader("Major Earthquake Events")
            earthquake_names = [ex["name"] for ex in examples.earthquake_examples]
            selected_earthquake = st.selectbox("Select earthquake event", earthquake_names)
            
            # Show description
            earthquake_example = examples.get_example_by_name(selected_earthquake)
            if earthquake_example:
                st.write(earthquake_example["description"])
                
                if st.button("Predict for this Event", key=f"earthquake_predict_{selected_earthquake}"):
                    with st.spinner(f"Running prediction model for {earthquake_example['name']}..."):
                        if "simulation" in earthquake_example:
                            sim = earthquake_example["simulation"]
                            prediction = get_simulation(
                                earthquake_example["latitude"],
                                earthquake_example["longitude"],
                                sim["magnitude"],
                                sim["depth"],
                                sim["tsunami_likelihood"]
                            )
                        else:
                            prediction = get_prediction(
                                earthquake_example["latitude"],
                                earthquake_example["longitude"],
                                datetime.now()
                            )
                        
                        # Store the example in session state for comparison
                        st.session_state.current_example = earthquake_example
                        
    # Update prediction in session state
    if prediction:
        st.session_state.prediction = prediction
    
    # Display prediction results
    if st.session_state.prediction:
        prediction = st.session_state.prediction
        st.header("Prediction Results")
        
        is_historical = False
        historical_comparison = None
        
        if st.session_state.input_method == "Example Locations" and "current_example" in st.session_state:
            is_historical = True
            current_example = st.session_state.current_example
            
            if "expected" in current_example:
                historical_comparison = current_example["expected"]
                
                st.markdown("""
                <div style="background-color:#2E3440; padding:10px; border-radius:5px; margin-bottom:15px; border:1px solid #4C566A; color:#E5E9F0">
                    <h3 style="margin:0; color:#88C0D0;">🔍 Model Validation: Predicted vs. Actual</h3>
                    <p style="margin:0;">Comparing our model's predictions with the actual historical event data</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Location info
        st.subheader(f"Location: {prediction['location']['name']}")
        
        # Display map with prediction location
        pred_lat = prediction['location']['latitude']
        pred_lng = prediction['location']['longitude']
        result_map = create_map(pred_lat, pred_lng, zoom_start=6)
        folium_static(result_map, width=800, height=300)
        
        # Create tabs for results
        tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Detailed Predictions", "Emergency Response"])
        
        # Risk Assessment Tab
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Earthquake likelihood gauge
                eq_fig = create_risk_gauge(
                    prediction['earthquake_likelihood'],
                    "Earthquake Likelihood"
                )
                st.plotly_chart(eq_fig, use_container_width=True)
            
            with col2:
                # Tsunami likelihood gauge
                tsunami_fig = create_risk_gauge(
                    prediction['tsunami_likelihood'],
                    "Tsunami Likelihood"
                )
                st.plotly_chart(tsunami_fig, use_container_width=True)
            
            # Economic impact
            if 'economic_loss_usd' in prediction and prediction['economic_loss_usd']:
                econ_data = prediction['economic_loss_usd']
                
                # Display economic loss
                st.metric(
                    "Estimated Economic Loss",
                    f"${econ_data['estimate']:,.2f}"
                )
                
                # Display risk category
                risk_category = econ_data['risk_category']
                
                if risk_category == "high":
                    st.error(f"Risk Category: {risk_category.upper()}")
                elif risk_category == "medium":
                    st.warning(f"Risk Category: {risk_category.upper()}")
                elif risk_category == "low":
                    st.info(f"Risk Category: {risk_category.upper()}")
                else:
                    st.success(f"Risk Category: {risk_category.upper()}")
        
        # Detailed Predictions Tab
        with tab2:
            # If we're comparing with historical data, show both
            if historical_comparison:
                st.subheader("Model Prediction vs. Actual Event")
                
                # Create comparison table
                comparison_data = {
                    "Metric": ["Magnitude", "Depth (km)", "Tsunami Likelihood"],
                    "Model Prediction": [
                        f"{prediction['magnitude_prediction']['estimate']:.1f}" if 'magnitude_prediction' in prediction else "N/A", 
                        f"{prediction['depth_prediction_km']['estimate']:.1f}" if 'depth_prediction_km' in prediction else "N/A",
                        f"{prediction['tsunami_likelihood']:.2f}"
                    ],
                    "Actual Historical Value": [
                        f"{historical_comparison['magnitude']:.1f}",
                        f"{historical_comparison['depth']:.1f}",
                        f"{historical_comparison['tsunami_likelihood']:.2f}"
                    ],
                    "Accuracy": [
                        f"{100 - abs(prediction['magnitude_prediction']['estimate'] - historical_comparison['magnitude']) / historical_comparison['magnitude'] * 100:.1f}%" if 'magnitude_prediction' in prediction else "N/A",
                        f"{100 - abs(prediction['depth_prediction_km']['estimate'] - historical_comparison['depth']) / historical_comparison['depth'] * 100:.1f}%" if 'depth_prediction_km' in prediction else "N/A",
                        f"{100 - abs(prediction['tsunami_likelihood'] - historical_comparison['tsunami_likelihood']) / max(0.01, historical_comparison['tsunami_likelihood']) * 100:.1f}%"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)
                
                # Add historical context
                st.subheader("Historical Event Details")
                st.markdown(historical_comparison["description"])
            
            # Standard detailed view
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Earthquake Details")
                if 'magnitude_prediction' in prediction and prediction['magnitude_prediction']:
                    mag_data = prediction['magnitude_prediction']
                    st.metric("Estimated Magnitude", f"{mag_data['estimate']:.1f}")
                    
                    # Magnitude range probabilities
                    if 'range_probs' in mag_data:
                        st.subheader("Magnitude Range Probabilities")
                        probs = mag_data['range_probs']
                        
                        # Create bar chart for magnitude probabilities
                        fig = px.bar(
                            x=list(probs.keys()),
                            y=list(probs.values()),
                            labels={'x': 'Magnitude Range', 'y': 'Probability'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display confidence interval
                    if 'confidence_interval' in mag_data:
                        ci = mag_data['confidence_interval']
                        st.write(f"95% Confidence Interval: {ci[0]:.1f} - {ci[1]:.1f}")
                else:
                    st.write("No detailed magnitude prediction available")
            
            with col2:
                st.subheader("Depth Details")
                if 'depth_prediction_km' in prediction and prediction['depth_prediction_km']:
                    depth_data = prediction['depth_prediction_km']
                    st.metric("Estimated Depth (km)", f"{depth_data['estimate']:.1f}")
                    
                    # Display confidence interval
                    if 'ci' in depth_data:
                        ci = depth_data['ci']
                        st.write(f"95% Confidence Interval: {ci[0]:.1f} - {ci[1]:.1f} km")
                else:
                    st.write("No depth prediction available")
        
        # Emergency Response Tab
        with tab3:
            if 'recommendations' in prediction and prediction['recommendations']:
                st.markdown(prediction['recommendations'])
            else:
                st.info("No recommendations available")
            
            # Chat with emergency assistant
            st.subheader("Emergency Response Assistant")
            st.info("Ask questions about what to do in this emergency situation.")
            
            # Initialize chat history in session state if it doesn't exist
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            
            # Create a container for chat messages
            chat_container = st.container()
            
            # Display all messages in the container
            with chat_container:
                for message in st.session_state.chat_messages:
                    if message["role"] == "user":
                        st.markdown(f"<div style='background-color:#2E86C1; padding:10px; border-radius:5px; margin-bottom:10px; color:white;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color:#383838; padding:10px; border-radius:5px; margin-bottom:10px; color:white;'><b>Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)
            
            # Function to handle sending messages
            def send_message():
                if st.session_state.user_input:
                    user_message = st.session_state.user_input
                    # Add user message to history
                    st.session_state.chat_messages.append({"role": "user", "content": user_message})
                    # Clear the input box
                    st.session_state.user_input = ""
                    
                    # Get response from API
                    response = chat_with_assistant(user_message, prediction.get('prediction_id'))
                    # Add assistant response to history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            # Create the chat input outside of a form
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Disable the Enter key triggering form submission to prevent page reload
                st.text_input("Your question:", 
                              key="user_input", 
                              placeholder="What should I do during an earthquake?",
                              on_change=send_message)
                
                # Add a small script to prevent Enter key from submitting the form
                st.markdown("""
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const textInputs = document.querySelectorAll('input[type="text"]');
                    textInputs.forEach(input => {
                        input.addEventListener('keydown', function(e) {
                            if (e.key === 'Enter') {
                                e.stopPropagation();
                                e.preventDefault();
                                const sendButton = document.querySelector('button[kind="primary"]:not([aria-describedby])');
                                if (sendButton) {
                                    sendButton.click();
                                }
                            }
                        });
                    });
                });
                </script>
                """, unsafe_allow_html=True)
            
            with col2:
                st.button("Send", on_click=send_message)
            
            # Add a button to clear chat history
            if st.button("Clear Chat History"):
                st.session_state.chat_messages = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()