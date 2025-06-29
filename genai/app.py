"""
Personalized City Poster Generator API

This FastAPI application provides an API for generating personalized city posters
with AQI information and health recommendations based on user profiles.
"""

import os
import json
import logging
from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import base64
from datetime import datetime
from io import BytesIO

# Import custom modules
from aws_air_quality_predictor.genai.generate_image import generate_city_poster, CityPosterGenerator
from aws_air_quality_predictor.genai.prompt_templates import get_aqi_category, get_health_advice
from aws_air_quality_predictor.genai.user_profile import UserProfile, create_user_profile, load_user_profile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="City Poster Generator API",
    description="API for generating personalized city posters with AQI information",
    version="1.0.0"
)

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'stability.stable-diffusion-xl')
ENABLE_USER_PROFILES = os.environ.get('ENABLE_USER_PROFILES', 'true').lower() == 'true'

# Initialize poster generator
poster_generator = CityPosterGenerator(model_id=BEDROCK_MODEL_ID, s3_bucket=S3_BUCKET)

# Simple in-memory user profile store (for demo purposes)
# In production, use a database like DynamoDB
user_profiles = {}

# Pydantic models for request/response validation
class PosterRequest(BaseModel):
    city: str
    aqi: float
    theme: Optional[str] = None
    user_id: Optional[str] = None
    format: Optional[str] = "json"
    profile_data: Optional[Dict[str, Any]] = None

class ProfileData(BaseModel):
    preferences: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str

class AQICategory(BaseModel):
    range: str
    category: str
    advice: str

class AQIInfo(BaseModel):
    aqi: int
    category: str
    health_advice: str

class PosterResponse(BaseModel):
    city_name: str
    theme_of_day: str
    aqi_value: float
    aqi_category: str
    health_advice: str
    s3_url: str = ""  # 设置默认值为空字符串
    image_data: str

class ProfileResponse(BaseModel):
    user_id: str
    preferences: Dict[str, Any]
    history: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

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

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'city-poster-generator'
    }

@app.post("/api/v1/posters/generate", response_model=PosterResponse)
async def generate_poster(request: PosterRequest):
    """Generate a personalized city poster"""
    try:
        # Extract data from request
        city_name = request.city
        aqi_value = request.aqi
        theme_of_day = request.theme
        user_id = request.user_id
        output_format = request.format
        
        # Get user profile if available
        user_profile = None
        if user_id and ENABLE_USER_PROFILES:
            user_profile = get_user_profile(user_id)
            
            # Update profile with request data if provided
            if request.profile_data:
                user_profile.update_preferences(request.profile_data)
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
            return StreamingResponse(
                BytesIO(image_data),
                media_type='image/jpeg',
                headers={"Content-Disposition": f"attachment; filename={city_name.lower().replace(' ', '_')}_poster.jpg"}
            )
        else:
            # Return JSON response
            return {
                'city_name': result['city_name'],
                'theme_of_day': result['theme_of_day'],
                'aqi_value': result['aqi_value'],
                'aqi_category': result['aqi_category'],
                'health_advice': result['health_advice'],
                's3_url': result['s3_url'] or "",  # 确保s3_url为None时返回空字符串
                'image_data': result['image_data']  # Base64 encoded image
            }
            
    except Exception as e:
        logger.error(f"Error generating poster: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/users/{user_id}/profile", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """Get user profile"""
    if not ENABLE_USER_PROFILES:
        raise HTTPException(status_code=403, detail="User profiles are disabled")
        
    profile = get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
            
    return {
        'user_id': profile.user_id,
        'preferences': profile.preferences,
        'history': profile.history
    }

@app.put("/api/v1/users/{user_id}/profile", response_model=ProfileResponse)
async def update_profile(user_id: str, profile_data: ProfileData):
    """Update user profile"""
    if not ENABLE_USER_PROFILES:
        raise HTTPException(status_code=403, detail="User profiles are disabled")
            
    profile = get_user_profile(user_id)
    profile.update_preferences(profile_data.preferences)
    save_user_profile(profile)
        
    return {
        'user_id': profile.user_id,
        'preferences': profile.preferences,
        'message': 'Profile updated successfully'
    }

@app.get("/api/v1/aqi/categories", response_model=List[AQICategory])
async def get_aqi_categories():
    """Get AQI categories and health advice"""
    categories = [
        {'range': '0-50', 'category': 'Good', 'advice': get_health_advice('Good')},
        {'range': '51-100', 'category': 'Moderate', 'advice': get_health_advice('Moderate')},
        {'range': '101-150', 'category': 'Unhealthy for Sensitive Groups', 'advice': get_health_advice('Unhealthy for Sensitive Groups')},
        {'range': '151-200', 'category': 'Unhealthy', 'advice': get_health_advice('Unhealthy')},
        {'range': '201-300', 'category': 'Very Unhealthy', 'advice': get_health_advice('Very Unhealthy')},
        {'range': '301+', 'category': 'Hazardous', 'advice': get_health_advice('Hazardous')}
    ]
    
    return categories

@app.get("/api/v1/aqi/category/{aqi_value}", response_model=AQIInfo)
async def get_aqi_info(aqi_value: int):
    """Get AQI category and health advice for a specific AQI value"""
    category = get_aqi_category(aqi_value)
    advice = get_health_advice(category)
    
    return {
        'aqi': aqi_value,
        'category': category,
        'health_advice': advice
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(status_code=404, content={'error': 'Resource not found'})

@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(exc)}")
    return JSONResponse(status_code=500, content={'error': 'Internal server error'})

if __name__ == '__main__':
    # Get port from environment or default to 8000
    port = int(os.environ.get('PORT', 8000))
    
    # Run the FastAPI app using uvicorn
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port) 