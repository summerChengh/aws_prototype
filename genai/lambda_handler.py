"""
AWS Lambda Handler for the Personalized City Poster Generator

This module adapts the FastAPI application to run as an AWS Lambda function,
enabling serverless deployment of the city poster generator.
"""

import json
import logging
import base64
from io import BytesIO

# Import AWS Lambda Powertools for better logging and tracing
try:
    from aws_lambda_powertools import Logger, Tracer
    from aws_lambda_powertools.event_handler import APIGatewayRestResolver
    from aws_lambda_powertools.utilities.typing import LambdaContext
    POWERTOOLS_AVAILABLE = True
except ImportError:
    POWERTOOLS_AVAILABLE = False
    
# Import the poster generator
from aws_air_quality_predictor.genai.generate_image import generate_city_poster, CityPosterGenerator
from aws_air_quality_predictor.genai.prompt_templates import get_aqi_category, get_health_advice
from aws_air_quality_predictor.genai.user_profile import UserProfile, create_user_profile, load_user_profile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Powertools if available
if POWERTOOLS_AVAILABLE:
    tracer = Tracer(service="city-poster-generator")
    logger = Logger(service="city-poster-generator")
    app = APIGatewayRestResolver()
else:
    tracer = lambda x: x  # No-op decorator when Powertools not available
    app = None

# Simple in-memory user profile store (for demo purposes)
# In production, use a database like DynamoDB
user_profiles = {}

def get_user_profile(user_id):
    """Get user profile from store or create a new one"""
    if user_id in user_profiles:
        return user_profiles[user_id]
    
    # Create a default profile
    profile = UserProfile(user_id=user_id)
    user_profiles[user_id] = profile
    return profile

def save_user_profile(profile):
    """Save user profile to store"""
    user_profiles[profile.user_id] = profile

@tracer
def generate_poster_handler(event):
    """Handle poster generation requests"""
    try:
        # Parse request data
        body = json.loads(event.get('body', '{}'))
        
        # Required parameters
        city_name = body.get('city')
        if not city_name:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'City name is required'})
            }
            
        aqi_value = body.get('aqi')
        if not isinstance(aqi_value, (int, float)):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Valid AQI value is required'})
            }
            
        # Optional parameters
        theme_of_day = body.get('theme')
        user_id = body.get('user_id')
        output_format = body.get('format', 'json')  # 'json' or 'image'
        
        # Get user profile if available
        user_profile = None
        if user_id:
            user_profile = get_user_profile(user_id)
            
            # Update profile with request data if provided
            if 'profile_data' in body:
                user_profile.update_preferences(body['profile_data'])
                save_user_profile(user_profile)
        
        # Determine theme if not provided
        if not theme_of_day and user_profile:
            theme_of_day = user_profile.get_daily_theme_suggestion(city_name)
        
        # Generate the poster
        result = generate_city_poster(
            city_name=city_name,
            theme_of_day=theme_of_day,
            aqi_value=aqi_value
        )
        
        # Update user history if profile exists
        if user_profile:
            user_profile.add_history_entry('poster_generate', {
                'city': city_name,
                'theme': theme_of_day,
                'aqi': aqi_value
            })
            save_user_profile(user_profile)
        
        # Return appropriate response based on format
        if output_format == 'image':
            # Return image as binary response
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'image/jpeg',
                    'Content-Disposition': f'attachment; filename="{city_name.lower().replace(" ", "_")}_poster.jpg"'
                },
                'body': result['image_data'],
                'isBase64Encoded': True
            }
        else:
            # Return JSON response
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'city_name': result['city_name'],
                    'theme_of_day': result['theme_of_day'],
                    'aqi_value': result['aqi_value'],
                    'aqi_category': result['aqi_category'],
                    'health_advice': result['health_advice'],
                    's3_url': result['s3_url'],
                    'image_data': result['image_data']  # Base64 encoded image
                })
            }
            
    except Exception as e:
        logger.error(f"Error generating poster: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

@tracer
def user_profile_handler(event):
    """Handle user profile requests"""
    try:
        # Get user ID from path parameters
        path_parameters = event.get('pathParameters', {})
        user_id = path_parameters.get('user_id')
        
        if not user_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'User ID is required'})
            }
        
        method = event.get('httpMethod')
        
        if method == 'GET':
            # Get user profile
            profile = get_user_profile(user_id)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'user_id': profile.user_id,
                    'preferences': profile.preferences,
                    'history': profile.history
                })
            }
        
        elif method == 'PUT':
            # Update user profile
            body = json.loads(event.get('body', '{}'))
            
            profile = get_user_profile(user_id)
            if 'preferences' in body:
                profile.update_preferences(body['preferences'])
                
            save_user_profile(profile)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'user_id': profile.user_id,
                    'preferences': profile.preferences,
                    'message': 'Profile updated successfully'
                })
            }
        
        else:
            return {
                'statusCode': 405,
                'body': json.dumps({'error': 'Method not allowed'})
            }
            
    except Exception as e:
        logger.error(f"Error handling user profile: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

@tracer
def aqi_info_handler(event):
    """Handle AQI information requests"""
    try:
        path = event.get('path', '')
        
        # Get all AQI categories
        if path.endswith('/categories'):
            categories = [
                {'range': '0-50', 'category': 'Good', 'advice': get_health_advice('Good')},
                {'range': '51-100', 'category': 'Moderate', 'advice': get_health_advice('Moderate')},
                {'range': '101-150', 'category': 'Unhealthy for Sensitive Groups', 'advice': get_health_advice('Unhealthy for Sensitive Groups')},
                {'range': '151-200', 'category': 'Unhealthy', 'advice': get_health_advice('Unhealthy')},
                {'range': '201-300', 'category': 'Very Unhealthy', 'advice': get_health_advice('Very Unhealthy')},
                {'range': '301+', 'category': 'Hazardous', 'advice': get_health_advice('Hazardous')}
            ]
            
            return {
                'statusCode': 200,
                'body': json.dumps(categories)
            }
        
        # Get specific AQI category
        elif '/category/' in path:
            # Extract AQI value from path
            path_parameters = event.get('pathParameters', {})
            aqi_value = path_parameters.get('aqi_value')
            
            if not aqi_value:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'AQI value is required'})
                }
            
            try:
                aqi_value = int(aqi_value)
            except ValueError:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'AQI value must be a number'})
                }
            
            category = get_aqi_category(aqi_value)
            advice = get_health_advice(category)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'aqi': aqi_value,
                    'category': category,
                    'health_advice': advice
                })
            }
        
        else:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Resource not found'})
            }
            
    except Exception as e:
        logger.error(f"Error handling AQI info: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

@tracer
def health_check_handler(event):
    """Handle health check requests"""
    import datetime
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'healthy',
            'timestamp': datetime.datetime.now().isoformat(),
            'service': 'city-poster-generator'
        })
    }

@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event (dict): Lambda event data
        context (LambdaContext): Lambda context object
        
    Returns:
        dict: API Gateway response
    """
    try:
        # Parse the request path
        path = event.get('path', '')
        method = event.get('httpMethod', '')
        
        # Route the request to the appropriate handler
        if path == '/health' and method == 'GET':
            return health_check_handler(event)
        elif path.startswith('/api/v1/posters') and method == 'POST':
            return generate_poster_handler(event)
        elif path.startswith('/api/v1/users') and (method == 'GET' or method == 'PUT'):
            return user_profile_handler(event)
        elif path.startswith('/api/v1/aqi') and method == 'GET':
            return aqi_info_handler(event)
        else:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Resource not found'})
            }
            
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error'})
        } 