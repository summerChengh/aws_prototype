"""
Example usage of the Personalized City Poster Generator.

This script demonstrates how to generate personalized city posters with
AQI information and health recommendations.
"""

import os
import json
import argparse
from datetime import datetime


from generate_image import generate_city_poster, CityPosterGenerator
from prompt_templates import get_aqi_category, get_health_advice
from user_profile import UserProfile, EXAMPLE_USER_PROFILE


def generate_basic_poster(city_name, aqi_value, theme=None, output_path=None, mode="auto", local_model_url=None):
    """
    Generate a basic city poster without user personalization.
    
    Args:
        city_name (str): Name of the city
        aqi_value (int): AQI value
        theme (str, optional): Theme for the day
        output_path (str, optional): Path to save the generated image
        mode (str, optional): Generation mode - "auto", "local", "local_sd", or "bedrock"
        local_model_url (str, optional): URL for the local Stable Diffusion API endpoint
        
    Returns:
        dict: Result of the poster generation
    """
    print(f"Generating basic poster for {city_name} with AQI {aqi_value}...")
    
    if not output_path:
        # Create a default output path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_city_name = city_name.lower().replace(" ", "_")
        output_path = f"{safe_city_name}_{timestamp}.jpg"
    
    # Generate the poster
    result = generate_city_poster(
        city_name=city_name,
        theme_of_day=theme,
        aqi_value=aqi_value,
        output_path=output_path,
        mode=mode,
        local_model_url=local_model_url
    )
    
    print(f"Generated poster for {result['city_name']}")
    print(f"Theme: {result['theme_of_day']}")
    print(f"AQI: {result['aqi_value']} ({result['aqi_category']})")
    print(f"Health Advice: {result['health_advice']}")
    print(f"Generation Mode: {result['generation_mode']}")
    if 'model_id' in result:
        print(f"Model: {result['model_id']}")
    print(f"Saved to: {result['local_path']}")
    
    return result

def generate_personalized_poster(city_name, aqi_value, user_profile, theme=None, output_path=None, mode="auto", local_model_url=None):
    """
    Generate a personalized city poster based on user profile.
    
    Args:
        city_name (str): Name of the city
        aqi_value (int): AQI value
        user_profile (UserProfile): User profile for personalization
        theme (str, optional): Theme for the day
        output_path (str, optional): Path to save the generated image
        mode (str, optional): Generation mode - "auto", "local", "local_sd", or "bedrock"
        local_model_url (str, optional): URL for the local Stable Diffusion API endpoint
        
    Returns:
        dict: Result of the poster generation
    """
    print(f"Generating personalized poster for {city_name} with AQI {aqi_value}...")
    
    if not output_path:
        # Create a default output path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_city_name = city_name.lower().replace(" ", "_")
        output_path = f"{safe_city_name}_personalized_{timestamp}.jpg"
    
    # Get theme from user profile if not provided
    if not theme:
        theme = user_profile.get_daily_theme_suggestion(city_name)
        print(f"Using theme from user profile: {theme}")
    
    # Create a poster generator
    generator = CityPosterGenerator(mode=mode, local_model_url=local_model_url)
    
    # Generate the base prompt
    from prompt_templates import generate_city_poster_prompt
    base_prompt = generate_city_poster_prompt(city_name, theme, aqi_value)
    
    # Enhance the prompt with user preferences
    enhanced_prompt = user_profile.enhance_poster_prompt(base_prompt, city_name)
    print("Enhanced prompt with user preferences")
    
    # Generate the poster with the enhanced prompt
    result = generator.generate_poster(
        city_name=city_name,
        theme_of_day=theme,
        aqi_value=aqi_value,
        output_path=output_path
    )
    
    # Update user history
    user_profile.add_history_entry('poster_generate', {
        'city': city_name,
        'theme': theme,
        'aqi': aqi_value,
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"Generated personalized poster for {result['city_name']}")
    print(f"Theme: {result['theme_of_day']}")
    print(f"AQI: {result['aqi_value']} ({result['aqi_category']})")
    print(f"Health Advice: {result['health_advice']}")
    print(f"Generation Mode: {result['generation_mode']}")
    if 'model_id' in result:
        print(f"Model: {result['model_id']}")
    print(f"Saved to: {result['local_path']}")
    
    return result

def main():
    """Main function to demonstrate the poster generator."""
    parser = argparse.ArgumentParser(description='Generate personalized city posters with AQI information.')
    parser.add_argument('--city', type=str, default='San Francisco', help='City name')
    parser.add_argument('--aqi', type=int, default=75, help='AQI value')
    parser.add_argument('--theme', type=str, help='Theme for the day')
    parser.add_argument('--personalized', action='store_true', help='Generate a personalized poster')
    parser.add_argument('--output', type=str, help='Output path for the generated image')
    parser.add_argument('--mode', type=str, choices=['auto', 'local', 'local_sd', 'bedrock'], default='auto', 
                        help='Image generation mode: auto (default), local, local_sd, or bedrock')
    parser.add_argument('--sd-url', type=str, default='http://localhost:7860', 
                        help='URL for local Stable Diffusion API (default: http://localhost:7860)')
    
    args = parser.parse_args()
    
    if args.personalized:
        # Load example user profile
        user_profile = UserProfile().load_from_json(EXAMPLE_USER_PROFILE)
        generate_personalized_poster(args.city, args.aqi, user_profile, args.theme, args.output, 
                                    args.mode, args.sd_url)
    else:
        generate_basic_poster(args.city, args.aqi, args.theme, args.output, 
                             args.mode, args.sd_url)

if __name__ == '__main__':
    main() 