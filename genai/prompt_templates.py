"""
Prompt templates for generating personalized city posters with AQI information.

This module contains templates for generating prompts that will be sent to
Amazon Bedrock for creating personalized city images with air quality
information and health recommendations.
"""

# Base template for city poster generation
CITY_POSTER_BASE_TEMPLATE = """
Create a beautiful, artistic poster-style image of {city_name} that captures its unique characteristics.
The image should prominently feature {theme_of_day} theme.
Include iconic landmarks and representative architecture of {city_name} as the main focal point.
The scene should reflect an Air Quality Index (AQI) of {aqi_value} ({aqi_category}), showing appropriate atmospheric conditions.
The image should be high quality, photorealistic with artistic elements, suitable for a weather/air quality app.
"""

# Templates for different AQI categories
AQI_STYLE_TEMPLATES = {
    "Good": """
The sky should be clear and vibrant blue, with excellent visibility.
The atmosphere should look fresh and clean, with vibrant colors throughout the scene.
The lighting should be bright and cheerful, creating a positive and healthy ambiance.
""",
    "Moderate": """
The sky should have a slightly hazy appearance with a light blue or slightly whitish hue.
The atmosphere should look slightly less clear, with slightly muted colors.
The lighting should be bright but with a subtle filter effect.
""",
    "Unhealthy for Sensitive Groups": """
The sky should have a noticeable haze with a grayish-blue color.
The atmosphere should look somewhat hazy, with moderately muted colors and reduced visibility of distant objects.
The lighting should be slightly diffused through the haze.
""",
    "Unhealthy": """
The sky should have a significant haze or smog with a grayish appearance.
The atmosphere should look visibly polluted with muted colors and limited visibility.
The lighting should be noticeably diffused, creating a filtered effect on sunlight.
""",
    "Very Unhealthy": """
The sky should have a thick haze or smog with a gray or brownish appearance.
The atmosphere should look heavily polluted with significantly muted colors and poor visibility.
The lighting should be heavily diffused, creating a dim environment.
""",
    "Hazardous": """
The sky should have a very thick smog with a dark gray, brown, or reddish appearance.
The atmosphere should look extremely polluted with severely muted colors and very poor visibility.
The lighting should be heavily obscured, creating a dark and ominous environment.
"""
}

# Templates for health recommendations based on AQI category
HEALTH_ADVICE_TEMPLATES = {
    "Good": "Health Advice: Air quality is considered satisfactory, and air pollution poses little or no risk.",
    "Moderate": "Health Advice: Air quality is acceptable; however, there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution.",
    "Unhealthy for Sensitive Groups": "Health Advice: Members of sensitive groups may experience health effects. The general public is not likely to be affected.",
    "Unhealthy": "Health Advice: Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
    "Very Unhealthy": "Health Advice: Health warnings of emergency conditions. The entire population is more likely to be affected.",
    "Hazardous": "Health Advice: Health alert: everyone may experience more serious health effects."
}

def get_aqi_category(aqi_value):
    """
    Determine AQI category based on the numeric value.
    
    Args:
        aqi_value (int): The AQI value
        
    Returns:
        str: The AQI category
    """
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_health_advice(aqi_category):
    """
    Get health advice based on AQI category.
    
    Args:
        aqi_category (str): The AQI category
        
    Returns:
        str: Health advice for the given AQI category
    """
    return HEALTH_ADVICE_TEMPLATES.get(aqi_category, "No specific health advice available.")

def generate_city_poster_prompt(city_name, theme_of_day, aqi_value):
    """
    Generate a complete prompt for city poster generation.
    
    Args:
        city_name (str): The name of the city
        theme_of_day (str): The theme for the day
        aqi_value (int): The AQI value
        
    Returns:
        str: Complete prompt for image generation
    """
    aqi_category = get_aqi_category(aqi_value)
    
    # Combine the base template with the appropriate AQI style
    prompt = CITY_POSTER_BASE_TEMPLATE.format(
        city_name=city_name,
        theme_of_day=theme_of_day,
        aqi_value=aqi_value,
        aqi_category=aqi_category
    )
    
    # Add the AQI-specific style guidance
    prompt += AQI_STYLE_TEMPLATES.get(aqi_category, "")
    
    return prompt.strip()
