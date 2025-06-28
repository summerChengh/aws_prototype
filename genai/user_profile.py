"""
User Profile Manager for Personalized Content Generation

This module handles user profiles and preferences to generate personalized
city posters that match the user's interests and characteristics.
"""

import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserProfile:
    """
    A class to manage user profiles and generate personalized content based on user characteristics.
    """
    
    def __init__(self, user_id=None, preferences=None):
        """
        Initialize a user profile.
        
        Args:
            user_id (str, optional): The user's unique identifier
            preferences (dict, optional): User preferences and characteristics
        """
        self.user_id = user_id
        self.preferences = preferences or {}
        self.history = []
    
    def load_from_json(self, json_data):
        """
        Load user profile from JSON data.
        
        Args:
            json_data (str or dict): JSON string or dictionary containing user profile data
            
        Returns:
            UserProfile: Self for method chaining
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        self.user_id = data.get('user_id', self.user_id)
        self.preferences = data.get('preferences', {})
        self.history = data.get('history', [])
        
        logger.info(f"Loaded profile for user {self.user_id}")
        return self
    
    def to_json(self):
        """
        Convert user profile to JSON.
        
        Returns:
            str: JSON representation of the user profile
        """
        data = {
            'user_id': self.user_id,
            'preferences': self.preferences,
            'history': self.history
        }
        return json.dumps(data)
    
    def update_preferences(self, new_preferences):
        """
        Update user preferences.
        
        Args:
            new_preferences (dict): New preferences to add or update
            
        Returns:
            UserProfile: Self for method chaining
        """
        self.preferences.update(new_preferences)
        logger.info(f"Updated preferences for user {self.user_id}")
        return self
    
    def add_history_entry(self, entry_type, data):
        """
        Add an entry to the user's history.
        
        Args:
            entry_type (str): Type of history entry (e.g., 'poster_view', 'search')
            data (dict): Data associated with the history entry
            
        Returns:
            UserProfile: Self for method chaining
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': entry_type,
            'data': data
        }
        self.history.append(entry)
        return self
    
    def get_city_preferences(self, city_name):
        """
        Get user preferences specific to a city.
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            dict: User preferences for the specified city
        """
        city_prefs = self.preferences.get('cities', {}).get(city_name, {})
        general_prefs = {k: v for k, v in self.preferences.items() if k != 'cities'}
        
        # Combine general preferences with city-specific ones
        combined = {**general_prefs, **city_prefs}
        return combined
    
    def enhance_poster_prompt(self, base_prompt, city_name):
        """
        Enhance a poster generation prompt based on user preferences.
        
        Args:
            base_prompt (str): Base prompt for poster generation
            city_name (str): Name of the city
            
        Returns:
            str: Enhanced prompt based on user preferences
        """
        prefs = self.get_city_preferences(city_name)
        
        # Add user preferences to the prompt
        enhancements = []
        
        # Add preferred art style
        if 'preferred_art_style' in prefs:
            enhancements.append(f"The image should be in {prefs['preferred_art_style']} style.")
        
        # Add preferred color scheme
        if 'preferred_colors' in prefs:
            enhancements.append(f"Use a color palette featuring {prefs['preferred_colors']}.")
        
        # Add preferred time of day
        if 'preferred_time_of_day' in prefs:
            enhancements.append(f"The scene should depict {prefs['preferred_time_of_day']}.")
        
        # Add preferred weather
        if 'preferred_weather' in prefs and 'rainy' not in base_prompt.lower():
            enhancements.append(f"Include {prefs['preferred_weather']} weather elements if appropriate for the AQI.")
        
        # Add preferred landmarks
        if 'preferred_landmarks' in prefs:
            landmarks = prefs['preferred_landmarks']
            if isinstance(landmarks, list):
                landmarks = ', '.join(landmarks)
            enhancements.append(f"Include recognizable landmarks such as {landmarks} if appropriate.")
        
        # Combine enhancements with base prompt
        enhanced_prompt = base_prompt
        if enhancements:
            enhanced_prompt += "\n" + " ".join(enhancements)
        
        return enhanced_prompt
    
    def get_daily_theme_suggestion(self, city_name, date=None):
        """
        Generate a theme suggestion based on user preferences and date.
        
        Args:
            city_name (str): Name of the city
            date (datetime, optional): Date for theme suggestion, defaults to today
            
        Returns:
            str: Suggested theme for the day
        """
        if date is None:
            date = datetime.now()
            
        prefs = self.get_city_preferences(city_name)
        
        # Default themes by season
        month = date.month
        if 3 <= month <= 5:  # Spring
            default_theme = "Spring Bloom"
        elif 6 <= month <= 8:  # Summer
            default_theme = "Summer Day"
        elif 9 <= month <= 11:  # Fall
            default_theme = "Autumn Colors"
        else:  # Winter
            default_theme = "Winter Scene"
        
        # Use user's preferred theme if available
        if 'preferred_themes' in prefs:
            themes = prefs['preferred_themes']
            if isinstance(themes, list) and themes:
                # Use the first theme in the list or rotate based on day of month
                theme_index = date.day % len(themes)
                return themes[theme_index]
            elif isinstance(themes, str):
                return themes
        
        return default_theme

def create_user_profile(user_data):
    """
    Create a user profile from user data.
    
    Args:
        user_data (dict): User data including preferences
        
    Returns:
        UserProfile: A new user profile
    """
    user_id = user_data.get('user_id', str(datetime.now().timestamp()))
    preferences = user_data.get('preferences', {})
    
    profile = UserProfile(user_id=user_id, preferences=preferences)
    logger.info(f"Created new user profile with ID {user_id}")
    
    return profile

def load_user_profile(user_id, profile_store=None):
    """
    Load a user profile from storage.
    
    Args:
        user_id (str): User ID
        profile_store (callable, optional): Function to retrieve profile data
        
    Returns:
        UserProfile: The loaded user profile or a new one if not found
    """
    if profile_store:
        try:
            profile_data = profile_store(user_id)
            if profile_data:
                return UserProfile().load_from_json(profile_data)
        except Exception as e:
            logger.error(f"Error loading user profile: {str(e)}")
    
    # Return a new profile if loading failed
    return UserProfile(user_id=user_id)

# Example user profile data
EXAMPLE_USER_PROFILE = {
    "user_id": "user123",
    "preferences": {
        "preferred_art_style": "watercolor painting",
        "preferred_colors": "blue and green tones",
        "preferred_time_of_day": "sunset",
        "preferred_weather": "partly cloudy",
        "cities": {
            "San Francisco": {
                "preferred_landmarks": ["Golden Gate Bridge", "Alcatraz", "Painted Ladies"],
                "preferred_themes": ["Foggy Morning", "Tech Innovation", "Coastal Views"]
            },
            "New York": {
                "preferred_landmarks": ["Empire State Building", "Central Park", "Brooklyn Bridge"],
                "preferred_themes": ["Urban Jungle", "City Lights", "Autumn in the Park"]
            }
        }
    }
} 