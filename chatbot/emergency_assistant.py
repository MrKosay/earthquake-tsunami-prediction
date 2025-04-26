import os
import openai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Get OpenAI API key and clean it
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    # Clean up key if it contains newlines or whitespace
    api_key = api_key.replace("\n", "").replace(" ", "").strip()

# Set OpenAI client 
client = openai.OpenAI(api_key=api_key)

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
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
            
        self.conversation_history = []
        
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
        system_prompt = (
            "You are an AI emergency response advisor specializing in earthquake and tsunami disasters. "
            "Your role is to provide clear, specific, and actionable recommendations to help people "
            "stay safe during and after these natural disasters. Be concise but comprehensive, "
            "focusing on practical advice that could save lives."
        )
        
        user_prompt = (
            f"Based on the following prediction data for an upcoming potential disaster, "
            f"please provide emergency response recommendations:\n\n"
            f"{location_context}\n\n"
            f"{formatted_prediction}\n\n"
            f"Please provide specific safety advice for this scenario, considering the risks of "
            f"earthquake, potential tsunami, and economic impact. Include recommendations for "
            f"before, during, and immediately after the event."
        )
        
        # Generate response using OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            recommendations = response.choices[0].message.content.strip()
            return recommendations
        
        except Exception as e:
            error_message = str(e)
            
            # Handle specific error cases with user-friendly messages
            if "insufficient_quota" in error_message:
                return (
                    "## Emergency Recommendations\n\n"
                    "I apologize, but I'm currently unable to provide personalized recommendations "
                    "due to API usage limits. Here are some general safety guidelines:\n\n"
                    "### Earthquake Safety:\n"
                    "- Drop, Cover, and Hold On when shaking starts\n"
                    "- Stay away from windows, exterior walls, and heavy furniture\n"
                    "- If outdoors, move to an open area away from buildings\n\n"
                    "### Tsunami Safety:\n"
                    "- If near the coast and you feel strong shaking, move to higher ground immediately\n"
                    "- Follow evacuation routes and official instructions\n"
                    "- Stay away from the coast until officials say it's safe to return\n\n"
                    "Please check with your local emergency management office for more specific guidance."
                )
            elif "openai.ChatCompletion" in error_message and "no longer supported" in error_message:
                return (
                    "## Technical Difficulties\n\n"
                    "I'm experiencing technical difficulties with my AI service and cannot generate personalized "
                    "recommendations at this time. Please refer to general emergency guidelines for your area "
                    "or contact the system administrator."
                )
            else:
                return f"Sorry, I couldn't generate emergency recommendations due to an error: {error_message}"
    
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
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Check for common earthquake and tsunami related questions
        # This serves as a fallback when API is unavailable
        response = self._get_hardcoded_response(user_message, prediction, location_name)
        if response:
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Prepare system prompt
        system_prompt = (
            "You are an AI emergency response advisor specializing in earthquake and tsunami disasters. "
            "Your goal is to provide helpful, accurate information to assist people before, during, "
            "and after natural disasters. Be concise, clear, and reassuring, while providing "
            "potentially life-saving advice."
        )
        
        # Prepare context from prediction if available
        context = ""
        if prediction:
            formatted_prediction = self._format_prediction_for_prompt(prediction)
            context = f"Current disaster prediction:\n{formatted_prediction}\n\n"
            
            if location_name:
                context = f"Location: {location_name}\n\n" + context
        
        # Create messages for API call
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context as system message if available
        if context:
            messages.append({"role": "system", "content": context})
        
        # Add conversation history
        messages.extend(self.conversation_history[-5:])  # Include last 5 messages for context
        
        # Generate response using OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            error_message = str(e)
            print(f"Chat API error: {error_message}")
            
            # Fall back to hardcoded responses
            fallback_response = self._get_hardcoded_response(user_message, prediction, location_name)
            if fallback_response:
                # Add fallback response to conversation history
                self.conversation_history.append({"role": "assistant", "content": fallback_response})
                return fallback_response
                
            # Handle specific error cases with user-friendly messages
            if "insufficient_quota" in error_message:
                friendly_error = (
                    "I'm providing this information directly from my emergency knowledge base:\n\n"
                    + self._get_general_emergency_advice()
                )
            elif "openai.ChatCompletion" in error_message and "no longer supported" in error_message:
                friendly_error = (
                    "I'm providing this information directly from my emergency knowledge base:\n\n"
                    + self._get_general_emergency_advice()
                )
            else:
                friendly_error = (
                    "I'm having trouble connecting to my knowledge services, but here's what I can "
                    "tell you about earthquake and tsunami safety:\n\n"
                    + self._get_general_emergency_advice()
                )
            
            # Add error response to conversation history
            self.conversation_history.append({"role": "assistant", "content": friendly_error})
            
            return friendly_error
    
    def _get_hardcoded_response(self, user_message, prediction=None, location_name=None):
        """Get a hardcoded response for common emergency questions"""
        user_query = user_message.lower()
        
        # Check for specific query types
        if any(word in user_query for word in ["earthquake", "shake", "tremor", "quake"]):
            if any(word in user_query for word in ["prepare", "preparation", "get ready", "before"]):
                return self._get_earthquake_preparedness_advice()
            elif any(word in user_query for word in ["during", "what to do", "how to survive", "safety"]):
                return self._get_earthquake_safety_advice()
            elif any(word in user_query for word in ["after", "aftermath", "follow", "next", "then"]):
                return self._get_earthquake_aftermath_advice()
                
        if any(word in user_query for word in ["tsunami", "wave", "flood", "water"]):
            if any(word in user_query for word in ["prepare", "preparation", "get ready", "before"]):
                return self._get_tsunami_preparedness_advice()
            elif any(word in user_query for word in ["during", "what to do", "how to survive", "safety"]):
                return self._get_tsunami_safety_advice()
            elif any(word in user_query for word in ["after", "aftermath", "follow", "next", "then"]):
                return self._get_tsunami_aftermath_advice()
                
        if any(word in user_query for word in ["evacuat", "leave", "run", "escape", "flee"]):
            return self._get_evacuation_advice()
            
        if any(word in user_query for word in ["emergency", "kit", "supplies", "pack"]):
            return self._get_emergency_kit_advice()
            
        if any(word in user_query for word in ["sign", "warning", "predict", "forecast"]):
            if "tsunami" in user_query:
                return self._get_tsunami_warning_signs()
            else:
                return self._get_earthquake_warning_signs()
                
        # No matching hardcoded response
        return None
        
    def _get_general_emergency_advice(self):
        """Generic emergency advice for earthquakes and tsunamis"""
        return (
            "## General Emergency Advice\n\n"
            "### Earthquake Safety:\n"
            "- **Before:** Secure heavy furniture, prepare emergency kits, know evacuation routes\n"
            "- **During:** Drop, Cover, Hold On - get under sturdy furniture, stay away from windows\n"
            "- **After:** Check for injuries, evacuate if needed, listen for emergency broadcasts\n\n"
            "### Tsunami Safety:\n"
            "- **If you feel strong shaking near the coast:** Move immediately to higher ground\n"
            "- **If you see the water recede unusually:** This is a tsunami warning sign - evacuate immediately\n"
            "- **Follow evacuation routes:** Don't return until officials say it's safe\n\n"
            "### Emergency Kit Essentials:\n"
            "- Water (1 gallon per person per day for at least 3 days)\n"
            "- Non-perishable food for 3+ days\n"
            "- Battery or hand-crank radio\n"
            "- Flashlight and extra batteries\n"
            "- First aid kit\n"
            "- Whistle to signal for help\n"
            "- Local maps\n"
            "- Phone chargers and backup battery\n"
        )
    
    def _get_earthquake_preparedness_advice(self):
        """Advice for preparing for an earthquake"""
        return (
            "## Earthquake Preparedness\n\n"
            "### Home Preparation:\n"
            "- Secure heavy furniture and appliances to walls with straps\n"
            "- Move heavy objects to lower shelves\n"
            "- Repair deep cracks in ceilings or foundations\n"
            "- Store breakable items in low, closed cabinets with latches\n"
            "- Hang heavy items like mirrors away from beds and seating\n\n"
            "### Create a Plan:\n"
            "- Identify safe spots in each room (under sturdy furniture, against interior walls)\n"
            "- Practice 'Drop, Cover, and Hold On' drills with your household\n"
            "- Establish a meeting point and communication plan\n"
            "- Learn how to shut off utilities\n\n"
            "### Prepare Emergency Supplies:\n"
            "- Water and non-perishable food for at least 3 days\n"
            "- Flashlights, batteries, and a radio\n"
            "- First aid kit and medications\n"
            "- Copies of important documents in waterproof container\n"
            "- Cash and emergency contact information\n"
        )
    
    def _get_earthquake_safety_advice(self):
        """Advice for what to do during an earthquake"""
        return (
            "## During an Earthquake: Safety Actions\n\n"
            "### If You're Indoors:\n"
            "- DROP to the ground before the earthquake drops you\n"
            "- COVER your head and neck with your arms. If possible, crawl under a sturdy table or desk\n"
            "- HOLD ON to your shelter until the shaking stops\n"
            "- Stay away from windows, outside doors, and walls\n"
            "- Stay inside until the shaking stops\n\n"
            "### If You're Outdoors:\n"
            "- Move to a clear area away from buildings, utility wires, and trees\n"
            "- Once in the open, drop to the ground and cover until shaking stops\n"
            "- If you're driving, pull over to a clear location, stop, and stay inside\n\n"
            "### If You're in Bed:\n"
            "- Stay there and cover your head with a pillow\n"
            "- Unless you're under heavy lighting or objects that could fall\n\n"
            "### NEVER:\n"
            "- Run outside during shaking\n"
            "- Stand in a doorway (modern doorways aren't stronger than other parts)\n"
            "- Use elevators\n"
        )
    
    def _get_earthquake_aftermath_advice(self):
        """Advice for what to do after an earthquake"""
        return (
            "## After an Earthquake: Next Steps\n\n"
            "### Immediate Actions:\n"
            "- Check yourself and others for injuries\n"
            "- If someone is seriously injured, call for help if possible\n"
            "- Look for and extinguish small fires\n"
            "- Listen to radio or TV for emergency information\n\n"
            "### Home Safety Check:\n"
            "- Inspect your home for damage - leave if it looks unsafe\n"
            "- Check for gas leaks by smell (rotten eggs) - don't use open flames if you suspect a leak\n"
            "- Look for damaged electrical wiring - turn off power at main panel if you see damage\n"
            "- Check water and sewage lines - avoid using toilets if lines might be damaged\n\n"
            "### General Safety:\n"
            "- Stay away from damaged areas and fallen power lines\n"
            "- Be prepared for aftershocks\n"
            "- If you're near the coast, be alert for tsunami warnings\n"
            "- Use text messages rather than calls to communicate (less network congestion)\n"
            "- Only call 911 for life-threatening emergencies\n"
        )
    
    def _get_tsunami_preparedness_advice(self):
        """Advice for preparing for a tsunami"""
        return (
            "## Tsunami Preparedness\n\n"
            "### If You Live in a Tsunami Risk Zone:\n"
            "- Know local evacuation routes to higher ground - aim for at least 100 feet above sea level\n"
            "- Prepare a 'go bag' with essential items you can grab quickly\n"
            "- Plan meeting locations with family - one in your neighborhood and one outside the tsunami zone\n"
            "- Learn the tsunami warning system in your area\n\n"
            "### Important Preparations:\n"
            "- Store important documents in waterproof containers that you can quickly take\n"
            "- Keep emergency supplies accessible and ready to go\n"
            "- Develop a family communication plan - include an out-of-state contact\n"
            "- Plan evacuation routes for your home, work, and common locations\n\n"
            "### Tsunami Warning Signs:\n"
            "- Strong ground shaking near the coast\n"
            "- Unusual ocean behavior - water suddenly receding or surging\n"
            "- Loud roaring sound coming from the ocean\n"
            "- Official warnings via emergency alert systems\n"
        )
    
    def _get_tsunami_safety_advice(self):
        """Advice for what to do during a tsunami warning"""
        return (
            "## During a Tsunami Warning: Immediate Actions\n\n"
            "### If You Feel Strong Shaking:\n"
            "- If you're in a coastal area and feel strong shaking, evacuate IMMEDIATELY\n"
            "- Don't wait for official warnings if you're near the coast - natural signs are your first alert\n"
            "- Move quickly to higher ground - at least 100 feet above sea level if possible\n\n"
            "### During an Official Warning:\n"
            "- Follow evacuation orders immediately\n"
            "- Take only essential items that are ready to go\n"
            "- Move on foot if possible - roads may be congested or damaged\n"
            "- Stay away from the coast, including beaches, harbors, and river mouths\n\n"
            "### If You Cannot Evacuate:\n"
            "- Go to the highest floor of a sturdy building\n"
            "- Climb a sturdy tree as a last resort\n"
            "- NEVER go to the shore to watch the tsunami\n"
            "- Remember that tsunami waves can last for hours - don't return until officials say it's safe\n"
        )
    
    def _get_tsunami_aftermath_advice(self):
        """Advice for what to do after a tsunami"""
        return (
            "## After a Tsunami: Recovery Steps\n\n"
            "### Immediate Safety:\n"
            "- Stay away from flooded and damaged areas until officials say it's safe to return\n"
            "- Continue listening to emergency broadcasts - tsunamis often have multiple waves\n"
            "- Avoid downed power lines and damaged buildings\n"
            "- Be alert for objects carried by tsunami water - debris can be dangerous\n\n"
            "### When Returning Home:\n"
            "- Check for structural damage before entering buildings\n"
            "- Use extreme caution as tsunami water often damages building foundations\n"
            "- Wear protective clothing, including boots and gloves\n"
            "- Watch for wild animals, including snakes, that may have entered with the water\n\n"
            "### Health Precautions:\n"
            "- Avoid wading in floodwater - it may be contaminated\n"
            "- Don't use food that came in contact with flood water\n"
            "- Use bottled or treated water until water supply is declared safe\n"
            "- Clean and disinfect everything that got wet\n"
            "- Get medical care for wounds that contact flood water\n"
        )
    
    def _get_evacuation_advice(self):
        """Advice for evacuation during a disaster"""
        return (
            "## Evacuation Guidelines\n\n"
            "### Before Evacuating:\n"
            "- Follow official evacuation orders - don't delay\n"
            "- Grab your emergency go-bag if readily available\n"
            "- Wear sturdy shoes and appropriate clothing\n"
            "- Shut off utilities if instructed to do so and time permits\n"
            "- Lock your home if time permits\n\n"
            "### During Evacuation:\n"
            "- Follow recommended evacuation routes - shortcuts may be blocked\n"
            "- Stay away from downed power lines\n"
            "- Don't walk or drive through flood waters\n"
            "- If evacuating by car, keep windows closed, ventilation system off\n"
            "- Stay tuned to emergency broadcasts\n\n"
            "### After Arriving at Safe Location:\n"
            "- Register with official evacuation centers so family can find you\n"
            "- Contact your emergency point of contact\n"
            "- Do not return home until authorities say it's safe\n"
            "- Follow official guidance for next steps\n"
        )
    
    def _get_emergency_kit_advice(self):
        """Advice for creating an emergency kit"""
        return (
            "## Emergency Kit Essentials\n\n"
            "### Basic Emergency Supply Kit:\n"
            "- Water: One gallon per person per day for at least 3 days\n"
            "- Food: Non-perishable food for at least 3 days\n"
            "- Battery-powered or hand-crank radio\n"
            "- Flashlight and extra batteries\n"
            "- First aid kit\n"
            "- Whistle to signal for help\n"
            "- Dust mask to filter contaminated air\n"
            "- Plastic sheeting and duct tape for shelter\n"
            "- Moist towelettes, garbage bags, and plastic ties for sanitation\n"
            "- Wrench or pliers to turn off utilities\n"
            "- Manual can opener\n"
            "- Local maps\n"
            "- Cell phone with chargers and backup battery\n\n"
            "### Additional Items to Consider:\n"
            "- Prescription medications and glasses\n"
            "- Infant formula and diapers\n"
            "- Pet food and extra water for pets\n"
            "- Important family documents in waterproof container\n"
            "- Cash or traveler's checks\n"
            "- Emergency reference materials\n"
            "- Sleeping bags or warm blankets\n"
            "- Complete change of clothing and sturdy shoes\n"
            "- Fire extinguisher\n"
            "- Matches in a waterproof container\n"
            "- Feminine supplies and personal hygiene items\n"
            "- Mess kits, paper cups, plates, towels, and utensils\n"
            "- Paper and pencil\n"
            "- Books, games, puzzles for children\n"
        )
    
    def _get_tsunami_warning_signs(self):
        """Information about tsunami warning signs"""
        return (
            "## Tsunami Warning Signs\n\n"
            "### Natural Warning Signs:\n"
            "- Strong earthquake that makes it hard to stand\n"
            "- Unusual ocean behavior - water suddenly receding far from shore\n"
            "- Water moving far inland when it shouldn't\n"
            "- Loud roaring sound similar to a train or aircraft\n"
            "- Abnormal ocean activity, a wall of water, or rapid rise in water level\n\n"
            "### Official Warning Systems:\n"
            "- Emergency broadcasts on radio and TV\n"
            "- Outdoor warning sirens\n"
            "- Text alerts on mobile phones\n"
            "- Warnings from emergency officials\n\n"
            "### Important Notes:\n"
            "- If you notice any natural warning signs, evacuate IMMEDIATELY - don't wait for official warnings\n"
            "- A tsunami may arrive within minutes after a nearby earthquake\n"
            "- The first wave may not be the largest or most dangerous\n"
            "- Never go to the shore to watch for a tsunami\n"
        )
    
    def _get_earthquake_warning_signs(self):
        """Information about earthquake warning signs"""
        return (
            "## Earthquake Warning Signs\n\n"
            "### Potential Precursors:\n"
            "- It's important to understand that, unlike tsunamis, there are typically NO reliable warning signs before most earthquakes\n"
            "- Scientists cannot yet predict exactly when earthquakes will occur\n"
            "- Some areas may have earthquake early warning systems that provide seconds to tens of seconds of warning\n\n"
            "### What to Know:\n"
            "- Small foreshocks may precede a major earthquake, but they cannot be distinguished from regular seismic activity until after the main event\n"
            "- Animals may sometimes behave unusually before earthquakes, but this is not a reliable warning system\n"
            "- Reports of strange lights, ground water changes, or unusual gas releases are being studied but are not proven warning signs\n\n"
            "### Best Approach:\n"
            "- Rather than looking for warning signs, focus on preparation\n"
            "- Develop an emergency plan and practice it regularly\n"
            "- Prepare your home by securing heavy furniture and objects\n"
            "- Create emergency kits and know evacuation routes\n"
            "- Stay informed about your area's seismic risk\n"
        )
    
    def clear_conversation_history(self):
        """
        Clear the conversation history.
        
        Returns:
            bool: True if successful
        """
        self.conversation_history = []
        return True 