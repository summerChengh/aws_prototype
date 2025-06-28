# Personalized City Poster Generator

This component of the AWS Air Quality Predictor system generates personalized city posters that combine:
1. City-specific imagery reflecting the city's unique characteristics
2. Daily theme visualization
3. Air Quality Index (AQI) information
4. Health recommendations based on AQI levels

The generated images are personalized based on user preferences and provide visually engaging representations of air quality data.

## Features

- **City-Specific Imagery**: Generate images that capture the unique characteristics and landmarks of different cities
- **AQI Visualization**: Visual representation of air quality through atmospheric conditions in the image
- **Personalization**: Customize images based on user preferences (art style, color palette, landmarks)
- **Health Information**: Include relevant health advice based on current AQI levels
- **Seasonal Themes**: Automatically suggest themes based on season or use custom themes
- **Multiple Output Formats**: Get results as JSON with base64-encoded images or direct image files
- **S3 Integration**: Automatically store generated images in S3 for persistence

## Architecture

The component uses Amazon Bedrock for image generation and consists of several modules:

- `prompt_templates.py`: Contains templates for generating prompts for image creation
- `generate_image.py`: Core functionality for generating images using Amazon Bedrock
- `user_profile.py`: Manages user profiles and personalization preferences
- `app.py`: Flask application providing RESTful API endpoints

## API Endpoints

### Generate Poster

```
POST /api/v1/posters/generate
```

Request body:
```json
{
  "city": "San Francisco",
  "aqi": 75,
  "theme": "Foggy Morning",
  "user_id": "user123",
  "format": "json"
}
```

Parameters:
- `city` (required): Name of the city
- `aqi` (required): Air Quality Index value
- `theme` (optional): Theme for the image (if not provided, will be determined based on user preferences or season)
- `user_id` (optional): User ID for personalization
- `format` (optional): Output format - "json" (default) or "image"

Response (JSON format):
```json
{
  "city_name": "San Francisco",
  "theme_of_day": "Foggy Morning",
  "aqi_value": 75,
  "aqi_category": "Moderate",
  "health_advice": "Health Advice: Air quality is acceptable; however, there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution.",
  "s3_url": "https://bucket-name.s3.amazonaws.com/city_posters/san_francisco/foggy_morning_20230615123045.jpg",
  "image_data": "base64_encoded_image_data..."
}
```

### User Profile Management

```
GET /api/v1/users/{user_id}/profile
```

Retrieves a user's profile including preferences and history.

```
PUT /api/v1/users/{user_id}/profile
```

Request body:
```json
{
  "preferences": {
    "preferred_art_style": "watercolor painting",
    "preferred_colors": "blue and green tones",
    "preferred_time_of_day": "sunset",
    "preferred_weather": "partly cloudy",
    "cities": {
      "San Francisco": {
        "preferred_landmarks": ["Golden Gate Bridge", "Alcatraz"],
        "preferred_themes": ["Foggy Morning", "Tech Innovation"]
      }
    }
  }
}
```

### AQI Information

```
GET /api/v1/aqi/categories
```

Returns all AQI categories and their health advice.

```
GET /api/v1/aqi/category/{aqi_value}
```

Returns the category and health advice for a specific AQI value.

## Setup and Deployment

### Prerequisites

- AWS Account with access to Amazon Bedrock
- Python 3.8+
- AWS CLI configured with appropriate permissions

### Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```
   export S3_BUCKET=your-bucket-name
   export BEDROCK_MODEL_ID=stability.stable-diffusion-xl
   export ENABLE_USER_PROFILES=true
   ```

3. Run the application:
   ```
   python app.py
   ```

### Deployment to AWS Lambda

1. Package the application:
   ```
   zip -r lambda_function.zip . -x "*.git*" "*.pytest_cache*" "__pycache__/*"
   ```

2. Deploy to Lambda:
   ```
   aws lambda create-function \
     --function-name city-poster-generator \
     --runtime python3.9 \
     --handler app.lambda_handler \
     --zip-file fileb://lambda_function.zip \
     --role arn:aws:iam::your-account-id:role/lambda-execution-role
   ```

3. Create API Gateway integration as described in the main project documentation.

## Customization

### Adding New AQI Categories

Modify the `AQI_STYLE_TEMPLATES` and `HEALTH_ADVICE_TEMPLATES` dictionaries in `prompt_templates.py`.

### Supporting New Image Models

The `CityPosterGenerator` class in `generate_image.py` can be extended to support additional image generation models by modifying the `_call_bedrock_model` method.

### Custom Themes

Add new themes to the `get_daily_theme_suggestion` method in `user_profile.py` or provide them directly in API requests.

## Examples

### Basic Usage

```python
from aws_air_quality_predictor.genai.generate_image import generate_city_poster

result = generate_city_poster(
    city_name="New York",
    theme_of_day="Winter Scene",
    aqi_value=120,
    output_path="new_york_poster.jpg"
)

print(f"Generated poster saved to {result['local_path']}")
print(f"AQI Category: {result['aqi_category']}")
print(f"Health Advice: {result['health_advice']}")
```

### With User Profile

```python
from aws_air_quality_predictor.genai.user_profile import UserProfile
from aws_air_quality_predictor.genai.generate_image import CityPosterGenerator
from aws_air_quality_predictor.genai.prompt_templates import generate_city_poster_prompt

# Create user profile
profile = UserProfile(user_id="user123")
profile.update_preferences({
    "preferred_art_style": "watercolor painting",
    "preferred_landmarks": ["Empire State Building", "Central Park"]
})

# Generate base prompt
city_name = "New York"
theme = "Winter Scene"
aqi_value = 120
base_prompt = generate_city_poster_prompt(city_name, theme, aqi_value)

# Enhance prompt with user preferences
enhanced_prompt = profile.enhance_poster_prompt(base_prompt, city_name)

# Generate poster
generator = CityPosterGenerator()
result = generator.generate_poster(
    city_name=city_name,
    theme_of_day=theme,
    aqi_value=aqi_value,
    output_path="personalized_ny_poster.jpg"
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 