"""
Personalized City Poster Generator API

This Flask application provides an API for generating personalized city posters
with AQI information and health recommendations based on user profiles.
"""

import os
import json
import logging
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import base64
from datetime import datetime

# Import custom modules
from aws_air_quality_predictor.genai.generate_image import generate_city_poster, CityPosterGenerator
from aws_air_quality_predictor.genai.prompt_templates import get_aqi_category, get_health_advice
from aws_air_quality_predictor.genai.user_profile import UserProfile, create_user_profile, load_user_profile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'stability.stable-diffusion-xl')
ENABLE_USER_PROFILES = os.environ.get('ENABLE_USER_PROFILES', 'true').lower() == 'true'

# Initialize poster generator
poster_generator = CityPosterGenerator(model_id=BEDROCK_MODEL_ID, s3_bucket=S3_BUCKET)

# Simple in-memory user profile store (for demo purposes)
# In production, use a database like DynamoDB
user_profiles = {}

def get_user_profile(user_id):
    """Get user profile from store or create a new one"""
    if not ENABLE_USER_PROFILES:
        return None
        
    if user_id in user_profiles:
        return user_profiles[user_id]
    
    # Create a default profile
    profile = UserProfile(user_id=user_id)
    user_profiles[user_id] = profile
    return profile

def save_user_profile(profile):
    """Save user profile to store"""
    if not ENABLE_USER_PROFILES:
        return
        
    user_profiles[profile.user_id] = profile

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'city-poster-generator'
    })

@app.route('/api/v1/posters/generate', methods=['POST'])
def generate_poster():
    """Generate a personalized city poster"""
    try:
        # Parse request data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Required parameters
        city_name = data.get('city')
        if not city_name:
            return jsonify({'error': 'City name is required'}), 400
            
        aqi_value = data.get('aqi')
        if not isinstance(aqi_value, (int, float)):
            return jsonify({'error': 'Valid AQI value is required'}), 400
            
        # Optional parameters
        theme_of_day = data.get('theme')
        user_id = data.get('user_id')
        output_format = data.get('format', 'json')  # 'json' or 'image'
        
        # Get user profile if available
        user_profile = None
        if user_id and ENABLE_USER_PROFILES:
            user_profile = get_user_profile(user_id)
            
            # Update profile with request data if provided
            if 'profile_data' in data:
                user_profile.update_preferences(data['profile_data'])
                save_user_profile(user_profile)
        
        # Determine theme if not provided
        if not theme_of_day:
            if user_profile:
                theme_of_day = user_profile.get_daily_theme_suggestion(city_name)
            else:
                # Default theme based on current month
                month = datetime.now().month
                if 3 <= month <= 5:  # Spring
                    theme_of_day = "Spring Day"
                elif 6 <= month <= 8:  # Summer
                    theme_of_day = "Summer Day"
                elif 9 <= month <= 11:  # Fall
                    theme_of_day = "Autumn Colors"
                else:  # Winter
                    theme_of_day = "Winter Scene"
        
        logger.info(f"Generating poster for {city_name} with theme '{theme_of_day}' and AQI {aqi_value}")
        
        # Generate the poster
        result = generate_city_poster(
            city_name=city_name,
            theme_of_day=theme_of_day,
            aqi_value=aqi_value,
            s3_bucket=S3_BUCKET
        )
        
        # Update user history if profile exists
        if user_profile:
            user_profile.add_history_entry('poster_generate', {
                'city': city_name,
                'theme': theme_of_day,
                'aqi': aqi_value,
                'timestamp': datetime.now().isoformat()
            })
            save_user_profile(user_profile)
        
        # Return appropriate response based on format
        if output_format == 'image':
            # Decode image from base64 and return as file
            image_data = base64.b64decode(result['image_data'])
            return send_file(
                BytesIO(image_data),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f"{city_name.lower().replace(' ', '_')}_poster.jpg"
            )
        else:
            # Return JSON response
            return jsonify({
                'city_name': result['city_name'],
                'theme_of_day': result['theme_of_day'],
                'aqi_value': result['aqi_value'],
                'aqi_category': result['aqi_category'],
                'health_advice': result['health_advice'],
                's3_url': result['s3_url'],
                'image_data': result['image_data']  # Base64 encoded image
            })
            
    except Exception as e:
        logger.error(f"Error generating poster: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/users/<user_id>/profile', methods=['GET', 'PUT'])
def manage_user_profile(user_id):
    """Get or update user profile"""
    if not ENABLE_USER_PROFILES:
        return jsonify({'error': 'User profiles are disabled'}), 403
        
    if request.method == 'GET':
        # Get user profile
        profile = get_user_profile(user_id)
        if not profile:
            return jsonify({'error': 'User profile not found'}), 404
            
        return jsonify({
            'user_id': profile.user_id,
            'preferences': profile.preferences,
            'history': profile.history
        })
    
    elif request.method == 'PUT':
        # Update user profile
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        profile = get_user_profile(user_id)
        if 'preferences' in data:
            profile.update_preferences(data['preferences'])
            
        save_user_profile(profile)
        
        return jsonify({
            'user_id': profile.user_id,
            'preferences': profile.preferences,
            'message': 'Profile updated successfully'
        })

@app.route('/api/v1/aqi/categories', methods=['GET'])
def get_aqi_categories():
    """Get AQI categories and health advice"""
    categories = [
        {'range': '0-50', 'category': 'Good', 'advice': get_health_advice('Good')},
        {'range': '51-100', 'category': 'Moderate', 'advice': get_health_advice('Moderate')},
        {'range': '101-150', 'category': 'Unhealthy for Sensitive Groups', 'advice': get_health_advice('Unhealthy for Sensitive Groups')},
        {'range': '151-200', 'category': 'Unhealthy', 'advice': get_health_advice('Unhealthy')},
        {'range': '201-300', 'category': 'Very Unhealthy', 'advice': get_health_advice('Very Unhealthy')},
        {'range': '301+', 'category': 'Hazardous', 'advice': get_health_advice('Hazardous')}
    ]
    
    return jsonify(categories)

@app.route('/api/v1/aqi/category/<int:aqi_value>', methods=['GET'])
def get_aqi_info(aqi_value):
    """Get AQI category and health advice for a specific AQI value"""
    category = get_aqi_category(aqi_value)
    advice = get_health_advice(category)
    
    return jsonify({
        'aqi': aqi_value,
        'category': category,
        'health_advice': advice
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False) 