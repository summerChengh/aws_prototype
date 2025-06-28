"""
Tests for the prompt_templates module.
"""

import unittest
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
try:
    from aws_air_quality_predictor.genai.prompt_templates import (
        get_aqi_category,
        get_health_advice,
        generate_city_poster_prompt,
        AQI_STYLE_TEMPLATES,
        HEALTH_ADVICE_TEMPLATES
    )
except ImportError:
    # Alternative import if package structure is not set up
    from prompt_templates import (
        get_aqi_category,
        get_health_advice,
        generate_city_poster_prompt,
        AQI_STYLE_TEMPLATES,
        HEALTH_ADVICE_TEMPLATES
    )

class TestPromptTemplates(unittest.TestCase):
    """Test cases for the prompt_templates module."""

    def test_get_aqi_category(self):
        """Test the get_aqi_category function."""
        self.assertEqual(get_aqi_category(25), "Good")
        self.assertEqual(get_aqi_category(50), "Good")
        self.assertEqual(get_aqi_category(51), "Moderate")
        self.assertEqual(get_aqi_category(100), "Moderate")
        self.assertEqual(get_aqi_category(101), "Unhealthy for Sensitive Groups")
        self.assertEqual(get_aqi_category(150), "Unhealthy for Sensitive Groups")
        self.assertEqual(get_aqi_category(151), "Unhealthy")
        self.assertEqual(get_aqi_category(200), "Unhealthy")
        self.assertEqual(get_aqi_category(201), "Very Unhealthy")
        self.assertEqual(get_aqi_category(300), "Very Unhealthy")
        self.assertEqual(get_aqi_category(301), "Hazardous")
        self.assertEqual(get_aqi_category(500), "Hazardous")

    def test_get_health_advice(self):
        """Test the get_health_advice function."""
        for category in HEALTH_ADVICE_TEMPLATES:
            self.assertEqual(get_health_advice(category), HEALTH_ADVICE_TEMPLATES[category])
        
        # Test with an unknown category
        self.assertEqual(get_health_advice("Unknown"), "No specific health advice available.")

    def test_generate_city_poster_prompt(self):
        """Test the generate_city_poster_prompt function."""
        city_name = "San Francisco"
        theme_of_day = "Foggy Morning"
        aqi_value = 75
        
        prompt = generate_city_poster_prompt(city_name, theme_of_day, aqi_value)
        
        # Check that the prompt contains the city name, theme, and AQI value
        self.assertIn(city_name, prompt)
        self.assertIn(theme_of_day, prompt)
        self.assertIn(str(aqi_value), prompt)
        self.assertIn("Moderate", prompt)  # AQI category for value 75
        
        # Check that the prompt contains the appropriate AQI style
        self.assertIn("slightly hazy", prompt.lower())
        
        # Test with different AQI values
        for aqi_value, expected_category in [
            (25, "Good"),
            (75, "Moderate"),
            (125, "Unhealthy for Sensitive Groups"),
            (175, "Unhealthy"),
            (250, "Very Unhealthy"),
            (350, "Hazardous")
        ]:
            prompt = generate_city_poster_prompt(city_name, theme_of_day, aqi_value)
            self.assertIn(expected_category, prompt)
            
            # Check that the prompt contains text from the appropriate AQI style template
            style_text = AQI_STYLE_TEMPLATES[expected_category].strip()
            style_snippet = style_text.split('\n')[0].strip()
            self.assertIn(style_snippet.lower(), prompt.lower())

if __name__ == '__main__':
    unittest.main() 