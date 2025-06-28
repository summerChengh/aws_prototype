<template>
  <div class="app-container">
    <header class="app-header">
      <h1>空气质量预测系统</h1>
    </header>
    
    <main class="app-content">
      <router-view />
    </main>
    
    <footer class="app-footer">
      <p>© {{ currentYear }} 空气质量预测系统 - 基于OpenAQ数据</p>
    </footer>
  </div>
</template>

<script lang="ts">
import { defineComponent, computed } from 'vue'

export default defineComponent({
  name: 'App',
  setup() {
    const currentYear = computed(() => new Date().getFullYear())
    
    return {
      currentYear
    }
  }
})
</script>

<style lang="scss">
html, body {
  margin: 0;
  padding: 0;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  height: 100%;
}

#app {
  height: 100%;
}

.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  background-color: #409EFF;
  color: white;
  padding: 1rem;
  text-align: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.app-content {
  flex: 1;
  padding: 2rem;
}

.app-footer {
  background-color: #f5f5f5;
  padding: 1rem;
  text-align: center;
  font-size: 0.9rem;
  color: #666;
}
</style> 