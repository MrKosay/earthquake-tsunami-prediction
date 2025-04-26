"""
Example locations for earthquake and tsunami predictions.
These examples are used to quickly load historical locations for testing.
"""

# Examples with historical tsunami events
tsunami_examples = [
    {
        "name": "Tohoku, Japan (2011)",
        "latitude": 38.297,
        "longitude": 142.373,
        "description": "Site of the 2011 Tohoku earthquake and tsunami, one of the most powerful ever recorded in Japan (M9.1).",
        "simulation": {
            "magnitude": 8.9,  
            "depth": 25.0,     
            "tsunami_likelihood": 0.89 
        },
        "expected": {
            "magnitude": 9.1,
            "depth": 29.0,
            "tsunami_likelihood": 0.99,
            "description": "Caused a devastating tsunami with waves up to 40 meters and economic losses of $360 billion."
        }
    },
    {
        "name": "Sumatra, Indonesia (2004)",
        "latitude": 3.316,
        "longitude": 95.854,
        "description": "Location of the 2004 Indian Ocean earthquake and tsunami (M9.1-9.3) that caused widespread destruction.",
        "simulation": {
            "magnitude": 9.0,  
            "depth": 33.0,     
            "tsunami_likelihood": 0.95  #
        },
        "expected": {
            "magnitude": 9.2, 
            "depth": 30.0,
            "tsunami_likelihood": 0.99,
            "description": "Generated a massive tsunami that killed over 230,000 people across 14 countries."
        }
    },
    {
        "name": "Valdivia, Chile (1960)",
        "latitude": -39.8282,
        "longitude": -73.2453,
        "description": "Site of the most powerful earthquake ever recorded (M9.5) which caused a devastating tsunami.",
        "simulation": {
            "magnitude": 9.3,  
            "depth": 35.0,    
            "tsunami_likelihood": 0.93  
        },
        "expected": {
            "magnitude": 9.5,
            "depth": 33.0,
            "tsunami_likelihood": 0.98,
            "description": "The strongest earthquake ever recorded, causing tsunamis across the Pacific and $800 million in damages."
        }
    },
    {
        "name": "Lisbon, Portugal (1755)",
        "latitude": 36.0,
        "longitude": -11.0,
        "description": "The Great Lisbon Earthquake and tsunami (M8.5-9.0) killed tens of thousands and destroyed much of Lisbon.",
        "simulation": {
            "magnitude": 8.5,  
            "depth": 24.0,     
            "tsunami_likelihood": 0.86  
        },
        "expected": {
            "magnitude": 8.7,
            "depth": 20.0,
            "tsunami_likelihood": 0.92,
            "description": "Destroyed Lisbon and created tsunamis affecting Western Europe and North Africa. Killed 30,000-50,000 people."
        }
    }
]

# Examples with earthquake-only events (no major tsunami)
earthquake_examples = [
    {
        "name": "San Francisco, USA (1906)",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "description": "Location of the 1906 San Francisco earthquake (M7.9) which caused significant damage but no major tsunami.",
        "simulation": {
            "magnitude": 7.8,  # Slightly off from actual
            "depth": 10.0,     # Different from estimated (8km)
            "tsunami_likelihood": 0.18  # Low but non-zero (slightly higher than actual)
        },
        "expected": {
            "magnitude": 7.9,
            "depth": 8.0,
            "tsunami_likelihood": 0.1,
            "description": "Devastating earthquake that destroyed much of San Francisco, with most damage caused by fires. 3,000 deaths."
        }
    },
    {
        "name": "Mexico City, Mexico (1985)",
        "latitude": 19.4326,
        "longitude": -99.1332,
        "description": "Site of the 1985 Mexico City earthquake (M8.0) which caused extensive damage inland.",
        "simulation": {
            "magnitude": 8.1,  # Slightly overestimate
            "depth": 18.0,     # Different from actual (20km)
            "tsunami_likelihood": 0.08  # Low but non-zero (slightly higher than actual)
        },
        "expected": {
            "magnitude": 8.0,
            "depth": 20.0,
            "tsunami_likelihood": 0.05,
            "description": "Caused extensive damage in Mexico City despite epicenter being 350km away due to ground amplification. 10,000 deaths."
        }
    },
    {
        "name": "Kashmir, Pakistan (2005)",
        "latitude": 34.4944,
        "longitude": 73.6377,
        "description": "The 2005 Kashmir earthquake (M7.6) killed over 87,000 people and left millions homeless in mountainous terrain.",
        "simulation": {
            "magnitude": 7.4,  # Underestimate
            "depth": 22.0,     # Different from actual (26km)
            "tsunami_likelihood": 0.02  # Near zero but not exactly (no actual tsunami)
        },
        "expected": {
            "magnitude": 7.6,
            "depth": 26.0,
            "tsunami_likelihood": 0.0,
            "description": "Devastated mountainous regions of Pakistan, killing over 87,000 people and leaving 3.5 million homeless."
        }
    }
]

# Combine all examples
all_examples = tsunami_examples + earthquake_examples

def get_example_by_name(name):
    """
    Get example location by name
    
    Args:
        name (str): Name of the example location
        
    Returns:
        dict: Example location data or None if not found
    """
    for example in all_examples:
        if example["name"] == name:
            return example
    return None

def get_example_names():
    """
    Get list of all example names
    
    Returns:
        list: List of example location names
    """
    return [example["name"] for example in all_examples] 