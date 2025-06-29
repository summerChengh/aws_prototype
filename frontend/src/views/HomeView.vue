<template>
  <div class="home-container">
    <!-- 城市和日期选择区域 -->
    <div class="selection-panel">
      <h2>空气质量预测</h2>
      <el-form :model="form" label-position="top">
        <el-form-item label="城市">
          <el-select 
            v-model="form.cityId" 
            placeholder="请选择城市" 
            :loading="citiesLoading"
            class="full-width"
          >
            <el-option
              v-for="city in cities"
              :key="city.id"
              :label="city.name"
              :value="city.id"
            />
          </el-select>
        </el-form-item>
        
        <el-form-item label="日期">
          <el-date-picker
            v-model="form.date"
            type="date"
            placeholder="选择预测日期"
            format="YYYY-MM-DD"
            value-format="YYYY-MM-DD"
            :disabled-date="disablePastDates"
            class="full-width"
          />
        </el-form-item>
        
        <el-form-item>
          <el-button 
            type="primary" 
            :loading="predictionLoading" 
            @click="submitPrediction"
            class="full-width"
          >
            预测空气质量
          </el-button>
        </el-form-item>
      </el-form>
      
      <!-- 添加图例组件 -->
      <AirQualityLegend />
    </div>
    
    <!-- 预测结果展示区域 -->
    <div v-if="predictionResult" class="results-panel">
      <!-- 生成图像展示区域 -->
      <div class="image-panel">
        <h3>空气质量可视化</h3>
        <div class="image-container">
          <img 
            v-if="imageSource" 
            :src="imageSource" 
            alt="空气质量可视化" 
            @click="showFullImage"
          />
          <div v-else class="image-placeholder">
            图像生成中...
          </div>
        </div>
      </div>
      
      <div class="info-row">
        <div class="aqi-panel" :class="aqiLevelClass">
          <h3>空气质量指数 (AQI)</h3>
          <div class="aqi-value">{{ predictionResult.aqi }}</div>
          <div class="aqi-level">{{ predictionResult.level }}</div>
        </div>
        
        <!-- 健康建议展示区域 -->
        <div class="advice-panel">
          <h3>健康建议</h3>
          <el-alert
            :title="predictionResult.health_advice"
            :type="alertType"
            :closable="false"
            show-icon
          />
          
          <div class="pollutants-info">
            <h4>主要污染物数据</h4>
            <el-descriptions :column="3" border>
              <el-descriptions-item label="PM2.5">{{ predictionResult.pollutants.pm25 }} μg/m³</el-descriptions-item>
              <el-descriptions-item label="PM10">{{ predictionResult.pollutants.pm10 }} μg/m³</el-descriptions-item>
              <el-descriptions-item label="O3">{{ predictionResult.pollutants.o3 }} μg/m³</el-descriptions-item>
              <el-descriptions-item label="NO2">{{ predictionResult.pollutants.no2 }} μg/m³</el-descriptions-item>
              <el-descriptions-item label="SO2">{{ predictionResult.pollutants.so2 }} μg/m³</el-descriptions-item>
              <el-descriptions-item label="CO">{{ predictionResult.pollutants.co }} mg/m³</el-descriptions-item>
            </el-descriptions>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 图像全屏预览对话框 -->
    <el-dialog
      v-model="dialogVisible"
      title="空气质量可视化"
      width="80%"
      center
    >
      <img 
        v-if="imageSource" 
        :src="imageSource" 
        alt="空气质量可视化" 
        style="width: 100%;"
      />
    </el-dialog>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, reactive, computed, onMounted } from 'vue'
import { useAirQualityStore } from '../stores/airQuality'
import { ElMessage } from 'element-plus'
import AirQualityLegend from '../components/AirQualityLegend.vue'

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
  health_advice: string
}

export default defineComponent({
  name: 'HomeView',
  components: {
    AirQualityLegend
  },
  setup() {
    const airQualityStore = useAirQualityStore()
    
    // 表单数据
    const form = reactive({
      cityId: '',
      date: new Date().toISOString().split('T')[0] // 默认今天
    })
    
    // 城市列表
    const cities = ref<City[]>([])
    const citiesLoading = ref(false)
    
    // 预测结果
    const predictionResult = ref<PredictionResult | null>(null)
    const predictionLoading = ref(false)
    
    // 图像预览对话框
    const dialogVisible = ref(false)
    
    // 禁用过去的日期
    const disablePastDates = (date: Date) => {
      return date.getTime() < Date.now() - 86400000 // 今天之前的日期禁用
    }
    
    // 根据AQI等级获取对应的CSS类
    const aqiLevelClass = computed(() => {
      if (!predictionResult.value) return ''
      
      const aqi = predictionResult.value.aqi
      if (aqi <= 50) return 'aqi-good'
      if (aqi <= 100) return 'aqi-moderate'
      if (aqi <= 150) return 'aqi-unhealthy-sensitive'
      if (aqi <= 200) return 'aqi-unhealthy'
      if (aqi <= 300) return 'aqi-very-unhealthy'
      return 'aqi-hazardous'
    })
    
    // 根据AQI等级获取对应的警告类型
    const alertType = computed(() => {
      if (!predictionResult.value) return 'info'
      
      const aqi = predictionResult.value.aqi
      if (aqi <= 50) return 'success'
      if (aqi <= 100) return 'info'
      if (aqi <= 150) return 'warning'
      return 'error'
    })
    
    // 图像源计算属性
    const imageSource = computed(() => {
      if (!predictionResult.value) return null
      
      // 优先使用image_url，如果为空则尝试使用image_data
      if (predictionResult.value.image_url) {
        return predictionResult.value.image_url
      } else if (predictionResult.value.image_data) {
        // 使用base64图像数据
        return `data:image/jpeg;base64,${predictionResult.value.image_data}`
      }
      
      return null
    })
    
    // 获取城市列表
    const fetchCities = async () => {
      citiesLoading.value = true
      try {
        const response = await airQualityStore.fetchCities()
        cities.value = response
      } catch (error) {
        ElMessage.error('获取城市列表失败')
        console.error('Failed to fetch cities:', error)
      } finally {
        citiesLoading.value = false
      }
    }
    
    // 提交预测请求
    const submitPrediction = async () => {
      if (!form.cityId || !form.date) {
        ElMessage.warning('请选择城市和日期')
        return
      }
      
      predictionLoading.value = true
      // 先清空之前的结果，这样会显示"图像生成中..."
      predictionResult.value = null
      
      try {
        const result = await airQualityStore.predictAirQuality(form.cityId, form.date)
        predictionResult.value = result
      } catch (error) {
        ElMessage.error('预测失败，请稍后重试')
        console.error('Prediction failed:', error)
      } finally {
        predictionLoading.value = false
      }
    }
    
    // 显示全屏图像
    const showFullImage = () => {
      dialogVisible.value = true
    }
    
    // 组件挂载时获取城市列表
    onMounted(() => {
      fetchCities()
    })
    
    return {
      form,
      cities,
      citiesLoading,
      predictionResult,
      predictionLoading,
      dialogVisible,
      disablePastDates,
      aqiLevelClass,
      alertType,
      submitPrediction,
      showFullImage,
      imageSource
    }
  }
})
</script>

<style lang="scss" scoped>
.home-container {
  max-width: 1200px;
  margin: 0 auto;
}

.selection-panel {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  
  h2 {
    margin-top: 0;
    margin-bottom: 20px;
    color: #409EFF;
  }
}

.full-width {
  width: 100%;
}

.results-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.info-row {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  
  @media (min-width: 768px) {
    grid-template-columns: 1fr 2fr;
  }
}

.aqi-panel {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  
  h3 {
    margin-top: 0;
    margin-bottom: 20px;
  }
  
  .aqi-value {
    font-size: 4rem;
    font-weight: bold;
    line-height: 1;
    margin-bottom: 10px;
  }
  
  .aqi-level {
    font-size: 1.5rem;
    font-weight: bold;
  }
  
  &.aqi-good {
    background-color: #f0f9eb;
    .aqi-value, .aqi-level { color: #67c23a; }
  }
  
  &.aqi-moderate {
    background-color: #fdf6ec;
    .aqi-value, .aqi-level { color: #e6a23c; }
  }
  
  &.aqi-unhealthy-sensitive {
    background-color: #fef0f0;
    .aqi-value, .aqi-level { color: #f56c6c; }
  }
  
  &.aqi-unhealthy {
    background-color: #f56c6c;
    .aqi-value, .aqi-level { color: #fff; }
  }
  
  &.aqi-very-unhealthy {
    background-color: #800080;
    .aqi-value, .aqi-level { color: #fff; }
  }
  
  &.aqi-hazardous {
    background-color: #7d0000;
    .aqi-value, .aqi-level { color: #fff; }
  }
}

.image-panel {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  width: 100%;
  
  h3 {
    margin-top: 0;
    margin-bottom: 20px;
  }
  
  .image-container {
    width: 100%;
    height: 450px; /* 增加高度 */
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background-color: #f5f7fa;
    border-radius: 4px;
    
    img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      cursor: pointer;
      transition: transform 0.3s;
      
      &:hover {
        transform: scale(1.05);
      }
    }
    
    .image-placeholder {
      color: #909399;
      font-size: 1.2rem;
    }
  }
}

.advice-panel {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  
  h3, h4 {
    margin-top: 0;
    margin-bottom: 20px;
  }
  
  .pollutants-info {
    margin-top: 20px;
  }
}
</style> 