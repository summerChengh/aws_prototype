<template>
  <div class="legend-container">
    <h3>空气质量指数 (AQI) 图例</h3>
    <div class="legend-items">
      <div class="legend-item" v-for="level in aqiLevels" :key="level.name">
        <div class="color-box" :style="{ backgroundColor: level.color }"></div>
        <div class="level-info">
          <div class="level-name">{{ level.name }}</div>
          <div class="level-range">{{ level.range }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'

export default defineComponent({
  name: 'AirQualityLegend',
  setup() {
    const aqiLevels = [
      { name: '优', range: '0-50', color: '#67c23a' },
      { name: '良', range: '51-100', color: '#e6a23c' },
      { name: '轻度污染', range: '101-150', color: '#f56c6c' },
      { name: '中度污染', range: '151-200', color: '#f56c6c' },
      { name: '重度污染', range: '201-300', color: '#800080' },
      { name: '严重污染', range: '301+', color: '#7d0000' }
    ]
    
    return {
      aqiLevels
    }
  }
})
</script>

<style lang="scss" scoped>
.legend-container {
  background-color: #fff;
  border-radius: 8px;
  padding: 15px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  margin-top: 20px;
  
  h3 {
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 1rem;
    color: #303133;
  }
}

.legend-items {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.legend-item {
  display: flex;
  align-items: center;
  padding: 5px;
  
  .color-box {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    margin-right: 8px;
  }
  
  .level-info {
    .level-name {
      font-weight: bold;
      font-size: 0.9rem;
    }
    
    .level-range {
      font-size: 0.8rem;
      color: #606266;
    }
  }
}
</style> 