import { defineStore } from 'pinia'
import axios from 'axios'

interface City {
  id: string
  name: string
  location_id: string
}

interface PollutantLevels {
  pm25: number
  pm10: number
  o3: number
  no2: number
  so2: number
  co: number
}

interface PredictionResult {
  aqi: number
  level: string
  pollutants: PollutantLevels
  image_url: string
  image_data?: string
  health_advice: string
}

export const useAirQualityStore = defineStore('airQuality', {
  state: () => ({
    cities: [] as City[],
    selectedCity: null as City | null,
    predictionResult: null as PredictionResult | null,
    loading: false,
    error: null as string | null
  }),
  
  actions: {
    async fetchCities() {
      this.loading = true
      this.error = null
      
      try {
        const response = await axios.get('/api/cities')
        this.cities = response.data.cities
        return this.cities
      } catch (error) {
        this.error = '获取城市列表失败'
        console.error('Error fetching cities:', error)
        throw error
      } finally {
        this.loading = false
      }
    },
    
    async predictAirQuality(cityId: string, date: string) {
      this.loading = true
      this.error = null
      
      try {
        const response = await axios.post('/api/predict', {
          city_id: cityId,
          date: date
        })
        
        this.predictionResult = response.data
        return this.predictionResult
      } catch (error) {
        this.error = '预测空气质量失败'
        console.error('Error predicting air quality:', error)
        throw error
      } finally {
        this.loading = false
      }
    }
  }
}) 