# Air Quality Prediction API

This directory contains the FastAPI application for the air quality prediction system.

## API Endpoints

### Get Cities

```
GET /api/cities
```

Returns a list of supported cities with their IDs, names, and location information.

**Response Example:**
```json
{
  "cities": [
    {
      "id": "beijing",
      "name": "北京",
      "location_id": "123456",
      "latitude": 39.9042,
      "longitude": 116.4074
    },
    ...
  ]
}
```

### Predict Air Quality

```
POST /api/predict
```

Predicts air quality for a specific city and date.

**Request Body:**
```json
{
  "city_id": "beijing",
  "date": "2025-06-25"
}
```

**Response Example:**
```json
{
  "aqi": 135,
  "level": "Unhealthy for Sensitive Groups",
  "pollutants": {
    "pm25": 45.2,
    "pm10": 88.7,
    "o3": 48.2,
    "no2": 37.1,
    "so2": 9.8,
    "co": 0.9
  },
  "image_url": "https://bucket-name.s3.region.amazonaws.com/images/beijing-20250625-12345.jpg",
  "health_advice": "Sensitive groups should reduce outdoor activity, consider wearing masks"
}
```

### Health Check

```
GET /health
```

Returns the health status of the API service.

**Response Example:**
```json
{
  "status": "healthy",
  "timestamp": "2023-06-25T12:34:56.789Z",
  "service": "air-quality-predictor"
}
```

## Running the API

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```
   uvicorn api.main:app --reload
   ```

3. Access the API documentation at:
   ```
   http://localhost:8000/docs
   ```

### Environment Variables

- `MODEL_DIR`: Directory containing trained models (default: './models/automl')
- `S3_BUCKET`: S3 bucket for storing generated images (default: 'air-quality-predictor-images')
- `CITIES_FILE`: Path to the cities JSON file (default: './data/cities.json')

## Testing

Run tests using pytest:
```
pytest api/tests/
``` 