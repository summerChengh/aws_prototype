"""
AWS Air Quality Predictor - GenAI Component

This package contains modules for generating personalized city posters
with AQI information and health recommendations using generative AI.
"""

__version__ = '0.1.0'

from .generate_image import generate_city_poster, CityPosterGenerator
from .prompt_templates import generate_city_poster_prompt, get_aqi_category, get_health_advice
from .user_profile import UserProfile, create_user_profile, load_user_profile

__all__ = [
    'generate_city_poster',
    'CityPosterGenerator',
    'generate_city_poster_prompt',
    'get_aqi_category',
    'get_health_advice',
    'UserProfile',
    'create_user_profile',
    'load_user_profile',
] 