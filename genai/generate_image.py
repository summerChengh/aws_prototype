"""
Personalized City Poster Generator

This module generates personalized city posters that reflect the characteristics of the city,
the theme of the day, AQI information, and health information using Amazon Bedrock's
image generation capabilities.

The generated poster combines:
1. A city-specific image showing the current air quality conditions
2. AQI information overlay
3. Health recommendations based on the AQI level
"""

import os
import json
import boto3
import base64
import logging
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from botocore.exceptions import ClientError

# Import prompt templates
from aws_air_quality_predictor.genai.prompt_templates import (
    generate_city_poster_prompt,
    get_aqi_category,
    get_health_advice
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CityPosterGenerator:
    """
    A class to generate personalized city posters with AQI information and health advice.
    """
    
    def __init__(self, model_id="stability.stable-diffusion-xl", s3_bucket=None):
        """
        Initialize the CityPosterGenerator.
        
        Args:
            model_id (str): The Bedrock model ID to use for image generation
            s3_bucket (str, optional): S3 bucket name for storing generated images
        """
        self.bedrock_client = boto3.client('bedrock-runtime')
        self.s3_client = boto3.client('s3') if s3_bucket else None
        self.s3_bucket = s3_bucket
        self.model_id = model_id
    
    def _call_bedrock_model(self, prompt):
        """
        Call Amazon Bedrock model to generate an image based on the prompt.
        
        Args:
            prompt (str): The prompt for image generation
            
        Returns:
            bytes: The generated image data
        """
        try:
            # Prepare request body based on the model
            if "stability" in self.model_id:
                request_body = {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 8,
                    "steps": 50,
                    "seed": 0,
                    "width": 1024,
                    "height": 768
                }
            else:
                # Default to Titan Image Generator format
                request_body = {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text": prompt,
                        "negativeText": "poor quality, blurry, distorted, unrealistic, low resolution",
                    },
                    "imageGenerationConfig": {
                        "numberOfImages": 1,
                        "height": 768,
                        "width": 1024,
                        "cfgScale": 8
                    }
                }
            
            # Make the API call
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse the response based on the model
            response_body = json.loads(response.get('body').read())
            
            if "stability" in self.model_id:
                image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
            else:
                # Default to Titan Image Generator format
                image_data = base64.b64decode(response_body['images'][0])
                
            return image_data
            
        except ClientError as e:
            logger.error(f"Error calling Bedrock: {str(e)}")
            raise
    
    def _add_aqi_overlay(self, image_data, city_name, aqi_value, theme_of_day, health_advice):
        """
        Add AQI information and health advice overlay to the generated image.
        
        Args:
            image_data (bytes): The raw image data
            city_name (str): Name of the city
            aqi_value (int): AQI value
            theme_of_day (str): Theme of the day
            health_advice (str): Health advice text
            
        Returns:
            BytesIO: The processed image with overlay
        """
        try:
            # Open the image
            image = Image.open(BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            
            # Determine text color based on image brightness
            # Use white text for dark backgrounds, black text for light backgrounds
            
            # Add semi-transparent overlay at the bottom
            overlay_height = 150
            overlay = Image.new('RGBA', (image.width, overlay_height), (0, 0, 0, 180))
            image.paste(overlay, (0, image.height - overlay_height), overlay)
            
            # Try to load a font, use default if not available
            try:
                title_font = ImageFont.truetype("Arial.ttf", 36)
                info_font = ImageFont.truetype("Arial.ttf", 24)
            except IOError:
                title_font = ImageFont.load_default()
                info_font = ImageFont.load_default()
            
            # Add text
            aqi_category = get_aqi_category(aqi_value)
            
            # City name and theme
            draw.text((20, image.height - overlay_height + 15), 
                     f"{city_name} - {theme_of_day}", fill=(255, 255, 255), font=title_font)
            
            # AQI information
            draw.text((20, image.height - overlay_height + 60), 
                     f"AQI: {aqi_value} - {aqi_category}", fill=(255, 255, 255), font=info_font)
            
            # Health advice (shortened if needed)
            if len(health_advice) > 100:
                health_advice = health_advice[:97] + "..."
                
            draw.text((20, image.height - overlay_height + 95), 
                     health_advice, fill=(255, 255, 255), font=info_font)
            
            # Convert back to bytes
            result = BytesIO()
            image.save(result, format='JPEG')
            result.seek(0)
            return result
            
        except Exception as e:
            logger.error(f"Error adding overlay: {str(e)}")
            raise
    
    def _save_to_s3(self, image_data, city_name, theme_of_day):
        """
        Save the generated image to S3 bucket.
        
        Args:
            image_data (BytesIO): The image data
            city_name (str): Name of the city
            theme_of_day (str): Theme of the day
            
        Returns:
            str: The S3 URL of the saved image
        """
        if not self.s3_bucket:
            logger.warning("No S3 bucket specified, image not saved to S3")
            return None
            
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            safe_city_name = city_name.lower().replace(" ", "_")
            safe_theme = theme_of_day.lower().replace(" ", "_")
            filename = f"{safe_city_name}/{safe_theme}_{timestamp}.jpg"
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                image_data,
                self.s3_bucket,
                f"city_posters/{filename}",
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # Generate the S3 URL
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/city_posters/{filename}"
            return s3_url
            
        except ClientError as e:
            logger.error(f"Error saving to S3: {str(e)}")
            return None
    
    def generate_poster(self, city_name, theme_of_day, aqi_value, save_to_s3=True, output_path=None):
        """
        Generate a personalized city poster with AQI information and health advice.
        
        Args:
            city_name (str): Name of the city
            theme_of_day (str): Theme of the day
            aqi_value (int): AQI value
            save_to_s3 (bool): Whether to save the image to S3
            output_path (str, optional): Local path to save the image
            
        Returns:
            dict: A dictionary containing the image data and metadata
        """
        logger.info(f"Generating poster for {city_name} with theme '{theme_of_day}' and AQI {aqi_value}")
        
        try:
            # Generate the prompt
            prompt = generate_city_poster_prompt(city_name, theme_of_day, aqi_value)
            logger.info(f"Generated prompt: {prompt[:100]}...")
            
            # Call Bedrock to generate the image
            image_data = self._call_bedrock_model(prompt)
            
            # Get health advice
            aqi_category = get_aqi_category(aqi_value)
            health_advice = get_health_advice(aqi_category)
            
            # Add AQI overlay
            processed_image = self._add_aqi_overlay(
                image_data, city_name, aqi_value, theme_of_day, health_advice
            )
            
            # Save to S3 if requested
            s3_url = None
            if save_to_s3 and self.s3_bucket:
                s3_url = self._save_to_s3(processed_image, city_name, theme_of_day)
                processed_image.seek(0)  # Reset the file pointer
            
            # Save locally if output_path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(processed_image.getvalue())
                logger.info(f"Image saved to {output_path}")
            
            # Return the result
            return {
                "city_name": city_name,
                "theme_of_day": theme_of_day,
                "aqi_value": aqi_value,
                "aqi_category": aqi_category,
                "health_advice": health_advice,
                "s3_url": s3_url,
                "local_path": output_path if output_path else None,
                "image_data": base64.b64encode(processed_image.getvalue()).decode('utf-8')
            }
            
        except Exception as e:
            logger.error(f"Error generating poster: {str(e)}")
            raise

def generate_city_poster(city_name, theme_of_day, aqi_value, output_path=None, s3_bucket=None):
    """
    Generate a personalized city poster with AQI information and health advice.
    
    This is a convenience function that uses the CityPosterGenerator class.
    
    Args:
        city_name (str): Name of the city
        theme_of_day (str): Theme of the day
        aqi_value (int): AQI value
        output_path (str, optional): Local path to save the image
        s3_bucket (str, optional): S3 bucket name for storing generated images
        
    Returns:
        dict: A dictionary containing the image data and metadata
    """
    generator = CityPosterGenerator(s3_bucket=s3_bucket)
    return generator.generate_poster(
        city_name=city_name,
        theme_of_day=theme_of_day,
        aqi_value=aqi_value,
        save_to_s3=bool(s3_bucket),
        output_path=output_path
    )

if __name__ == "__main__":
    # Example usage
    result = generate_city_poster(
        city_name="San Francisco",
        theme_of_day="Foggy Morning",
        aqi_value=75,
        output_path="example_poster.jpg"
    )
    
    print(f"Generated poster for {result['city_name']}")
    print(f"AQI Category: {result['aqi_category']}")
    print(f"Health Advice: {result['health_advice']}")
    if result['local_path']:
        print(f"Saved to: {result['local_path']}")
    if result['s3_url']:
        print(f"S3 URL: {result['s3_url']}")
