import axios from 'axios'

// 创建axios实例
const apiClient = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  config => {
    // 可以在这里添加认证信息等
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  response => {
    return response
  },
  error => {
    console.error('API请求错误:', error)
    return Promise.reject(error)
  }
)

// 获取城市列表
export const getCities = async () => {
  try {
    const response = await apiClient.get('/cities')
    return response.data.cities
  } catch (error) {
    console.error('获取城市列表失败:', error)
    throw error
  }
}

// 预测空气质量
export const predictAirQuality = async (cityId: string, date: string) => {
  try {
    const response = await apiClient.post('/predict', {
      city_id: cityId,
      date: date
    })
    return response.data
  } catch (error) {
    console.error('预测空气质量失败:', error)
    throw error
  }
} 