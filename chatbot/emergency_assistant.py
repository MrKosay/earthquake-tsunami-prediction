import os
import sys
from dotenv import load_dotenv
import json
import requests
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

class EmergencyAssistant:
    """
    AI-powered chatbot for providing emergency response recommendations based on
    earthquake and tsunami predictions.
    """
    
    def __init__(self):
        """
        Initialize the EmergencyAssistant.
        """
        # Check if API key is available
        self.api_key = os.getenv("OPENAI_API_KEY")
        print(f"API key loaded, length: {len(self.api_key) if self.api_key else 0}")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
            
        self.conversation_history = []
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
        print(f"EmergencyAssistant initialized with model: {self.model}")
    
    def _format_prediction_for_prompt(self, prediction):
        """
        Format the prediction data for inclusion in the prompt.
        
        Args:
            prediction (dict): Dictionary containing prediction results
            
        Returns:
            str: Formatted prediction for prompt
        """
        # Check if prediction is empty
        if not prediction:
            return "No prediction data available."
        
        # Format earthquake information
        eq_info = []
        eq_likelihood = prediction.get('earthquake_likelihood', 0)
        eq_info.append(f"Earthquake likelihood: {eq_likelihood:.2f}")
        
        # Add magnitude and depth info if available
        if 'magnitude_prediction' in prediction:
            mag_data = prediction['magnitude_prediction']
            mag_estimate = mag_data.get('estimate', 'Unknown')
            mag_ci = mag_data.get('confidence_interval', [])
            
            eq_info.append(f"Estimated magnitude: {mag_estimate:.1f}")
            if mag_ci:
                eq_info.append(f"Magnitude confidence interval: {mag_ci[0]:.1f} - {mag_ci[1]:.1f}")
        
        if 'depth_prediction_km' in prediction:
            depth_data = prediction['depth_prediction_km']
            depth_estimate = depth_data.get('estimate', 'Unknown')
            depth_ci = depth_data.get('ci', [])
            
            eq_info.append(f"Estimated depth: {depth_estimate:.1f} km")
            if depth_ci:
                eq_info.append(f"Depth confidence interval: {depth_ci[0]:.1f} - {depth_ci[1]:.1f} km")
        
        # Format tsunami information
        tsunami_info = []
        tsunami_likelihood = prediction.get('tsunami_likelihood', 0)
        tsunami_info.append(f"Tsunami likelihood: {tsunami_likelihood:.2f}")
        
        # Format economic loss information
        economic_info = []
        if 'economic_loss_usd' in prediction:
            econ_data = prediction['economic_loss_usd']
            econ_estimate = econ_data.get('estimate', 0)
            risk_category = econ_data.get('risk_category', 'unknown')
            
            economic_info.append(f"Estimated economic loss: ${econ_estimate:,.2f}")
            economic_info.append(f"Risk category: {risk_category}")
        
        # Combine all sections
        sections = [
            "## Earthquake Information", 
            "\n".join(eq_info),
            "## Tsunami Information", 
            "\n".join(tsunami_info),
            "## Economic Impact", 
            "\n".join(economic_info)
        ]
        
        return "\n".join(sections)
    
    def _call_openai_api(self, messages, max_tokens=500):
        """
        Call the OpenAI API.
        
        Args:
            messages (list): List of message dictionaries
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Generated response
        """
        print(f"Calling OpenAI API with {len(messages)} messages")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            print(f"Sending request to {self.api_base}")
            print(f"Using model: {self.model}")
            
            response = requests.post(self.api_base, headers=headers, json=data)
            status_code = response.status_code
            print(f"API response status code: {status_code}")
            
            if status_code != 200:
                print(f"API error response: {response.text}")
                return self._get_fallback_response() + f"\n\n(API Error: Status code {status_code})"
                
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses
            response_json = response.json()
            print(f"API response received, content length: {len(str(response_json))}")
            
            result = response_json["choices"][0]["message"]["content"]
            print(f"Extracted text response, length: {len(result)}")
            return result
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}", file=sys.stderr)
            # Print traceback for debugging
            import traceback
            traceback.print_exc()
            return self._get_fallback_response() + f"\n\n(API Error: {str(e)})"
    
    def _get_fallback_response(self):
        """
        Get a fallback response for when the API call fails.
        
        Returns:
            str: Fallback response
        """
        # Get the user's last message to provide a more relevant response
        last_user_message = ""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        
        base_response = (
            "I can provide you with emergency guidance for earthquake and tsunami situations:\n\n"
            "### Earthquake Safety:\n"
            "- Drop to the ground, take cover under sturdy furniture, and hold on until shaking stops\n"
            "- Stay away from windows, exterior walls, and anything that could fall\n"
            "- If outdoors, move to an open area away from buildings, trees, and power lines\n\n"
            "### Tsunami Safety:\n"
            "- If you feel strong shaking near the coast, move immediately to higher ground\n"
            "- Follow evacuation routes and official instructions\n"
            "- Wait for official all-clear before returning to low-lying areas\n\n"
            "### General Emergency Preparedness:\n"
            "- Create an emergency plan and discuss it with your household\n"
            "- Prepare an emergency kit with water, food, medications, and essentials\n"
            "- Know evacuation routes and emergency contact numbers"
        )
        
        
        prefix = ""
        if "earthquake" in last_user_message.lower():
            prefix = "Regarding earthquakes: "
        elif "tsunami" in last_user_message.lower():
            prefix = "About tsunami safety: "
        elif "prepare" in last_user_message.lower() or "preparation" in last_user_message.lower():
            prefix = "For emergency preparation: "
        elif "kit" in last_user_message.lower() or "supplies" in last_user_message.lower():
            prefix = "For emergency supplies: "
        elif "evacuat" in last_user_message.lower():
            prefix = "About evacuation procedures: "
        elif "hello" in last_user_message.lower() or "hi" in last_user_message.lower():
            prefix = "Hello! I'm your emergency response advisor. "
        elif "help" in last_user_message.lower():
            prefix = "I'm here to help! "
        
        return prefix + base_response if prefix else base_response
    
    def get_emergency_recommendations(self, prediction, location_name=None):
        """
        Get emergency recommendations based on prediction results.
        
        Args:
            prediction (dict): Dictionary containing prediction results
            location_name (str, optional): Name of the location for context
            
        Returns:
            str: Emergency recommendations
        """
        # Format prediction data
        formatted_prediction = self._format_prediction_for_prompt(prediction)
        
        # Prepare location context
        location_context = f"Location: {location_name}" if location_name else "Location: Unknown"
        
        # Create prompt
        system_prompt = {
            "role": "system", 
            "content": (
                "You are an AI emergency response advisor specializing in earthquake and tsunami disasters. "
                "Your role is to provide clear, specific, and actionable recommendations to help people "
                "stay safe during and after these natural disasters. Be concise but comprehensive, "
                "focusing on practical advice that could save lives."
            )
        }
        
        user_prompt = {
            "role": "user",
            "content": (
                f"Based on the following prediction data for an upcoming potential disaster, "
                f"please provide emergency response recommendations:\n\n"
                f"{location_context}\n\n"
                f"{formatted_prediction}\n\n"
                f"Please provide specific safety advice for this scenario, considering the risks of "
                f"earthquake, potential tsunami, and economic impact. Include recommendations for "
                f"before, during, and immediately after the event."
            )
        }
        
        # Generate response using API
        try:
            recommendations = self._call_openai_api(
                messages=[system_prompt, user_prompt],
                max_tokens=800
            )
            return recommendations
        except Exception as e:
            print(f"Error getting emergency recommendations: {str(e)}", file=sys.stderr)
            return self._get_fallback_response()
    
    def chat(self, user_message, prediction=None, location_name=None):
        """
        Chat with the user and provide emergency-related assistance.
        
        Args:
            user_message (str): User's message
            prediction (dict, optional): Dictionary containing prediction results
            location_name (str, optional): Name of the location for context
            
        Returns:
            str: Assistant's response
        """
        print(f"Chat request received: '{user_message}'")
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Prepare system prompt with context
        system_content = (
            "You are an AI emergency response advisor specializing in earthquake and tsunami disasters. "
            "Your goal is to provide helpful, accurate information to assist people before, during, "
            "and after natural disasters. Be concise, clear, and reassuring, while providing "
            "potentially life-saving advice."
        )
        
        # Prepare context from prediction if available
        if prediction:
            print(f"Including prediction data in context")
            formatted_prediction = self._format_prediction_for_prompt(prediction)
            system_content += f"\n\nCurrent disaster prediction:\n{formatted_prediction}"
            
            if location_name:
                system_content += f"\n\nLocation: {location_name}"
        
        # Create messages for API call
        messages = [
            {"role": "system", "content": system_content}
        ]
        
        # Add conversation history (last 10 messages for context)
        history_to_include = self.conversation_history[-10:]  # Include more history for better context
        messages.extend(history_to_include)
        
        print(f"Sending {len(messages)} messages to API")
        
        # Generate response using API
        try:
            assistant_response = self._call_openai_api(
                messages=messages,
                max_tokens=500
            )
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            print(f"Response received, length: {len(assistant_response)}")
            return assistant_response
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            print(error_msg, file=sys.stderr)
            
            # Print traceback for debugging
            import traceback
            traceback.print_exc()
            
            fallback = self._get_fallback_response()
            # Add fallback response to conversation history
            self.conversation_history.append({"role": "assistant", "content": fallback})
            return fallback
    
    def clear_conversation_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = [] 