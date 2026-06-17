<template>
  <div class="control-panel">
    <div class="control-header">
      <h3 class="panel-title">控制面板</h3>
    </div>

    <div class="control-content">
      <div class="button-group">
        <el-button
          type="success"
          size="large"
          :loading="isProcessing && !isPaused"
          :disabled="isProcessing && !isPaused"
          :class="{ 'paused': isPaused }"
          @click="handleStartProcessing"
          class="action-button"
        >
          <template #icon>
            <span v-if="!isProcessing">▶</span>
            <span v-else-if="isPaused">▶</span>
          </template>
          {{ isPaused ? '继续处理' : '开始处理' }}
        </el-button>

        <el-button
          type="warning"
          size="large"
          :disabled="!isProcessing || isPaused"
          @click="handlePauseProcessing"
          class="action-button"
        >
          <template #icon>
            <span>⏸</span>
          </template>
          暂停
        </el-button>

        <el-button
          type="danger"
          size="large"
          :disabled="!isProcessing"
          @click="handleStopProcessing"
          class="action-button"
        >
          <template #icon>
            <span>⏹</span>
          </template>
          结束处理
        </el-button>
      </div>

      <div class="status-section">
        <div class="status-item">
          <span class="status-label">处理状态:</span>
          <span class="status-value" :class="statusClass">
            {{ statusText }}
          </span>
        </div>

        <div class="status-item">
          <span class="status-label">连接状态:</span>
          <span class="status-value" :class="connectionStatusClass">
            <span class="status-dot"></span>
            {{ connectionStatusText }}
          </span>
        </div>

        <div class="status-item" v-if="isLoading">
          <span class="status-label">处理状态:</span>
          <span class="status-value status-processing">
            <span class="status-dot"></span>
            {{ loadingMessage }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, computed } from 'vue';
import { ElButton } from 'element-plus';
import { query } from '@/api/home.js';
import JSZip from 'jszip';

const props = defineProps({
  connectionStatus: {
    type: String,
    default: 'connected'
  }
});

const isProcessing = ref(false);
const isPaused = ref(false);
const currentConnectionStatus = ref(props.connectionStatus);
const isLoading = ref(false);
const loadingMessage = ref('');
const isProcessingFrame = ref(false);

const emit = defineEmits<{
  'start-processing': [];
  'stop-processing': [];
  'pause-processing': [];
  'resume-processing': [];
  'images-updated': [images: {
    originalImages?: {
      camera: string;
      radar: string;
      lidar: string;
      other: string;
      micro?: string;
    };
    boxesImages?: {
      camera: string;
      radar: string;
      lidar: string;
      other: string;
    };
    fusionImage?: string;
  }];
  'detections-updated': [detectionData: {
    boxesImages?: {
      camera: string;
      radar: string;
      lidar: string;
      other: string;
    };
    fusionImage?: string;
    detections?: any[];
  }];
  'connection-status-changed': [status: 'connected' | 'disconnected' | 'connecting'];
  'output-json-updated': [outputJson: any];
  'metrics-updated': [metrics: any];
}>();

const statusText = computed(() => {
  if (isProcessing.value && isPaused.value) {
    return '已暂停';
  } else if (isProcessing.value) {
    return '处理中...';
  }
  return '已就绪';
});

const statusClass = computed(() => {
  return {
    'status-processing': isProcessing.value && !isPaused.value,
    'status-paused': isProcessing.value && isPaused.value,
    'status-idle': !isProcessing.value
  };
});

const connectionStatusText = computed(() => {
  const statusMap = {
    connected: '已连接',
    disconnected: '已断开',
    connecting: '连接中...'
  };

  return statusMap[currentConnectionStatus.value as keyof typeof statusMap] || '未知';
});

const connectionStatusClass = computed(() => {
  return {
    'status-connected': currentConnectionStatus.value === 'connected',
    'status-disconnected': currentConnectionStatus.value === 'disconnected',
    'status-connecting': currentConnectionStatus.value === 'connecting'
  };
});

const frame = ref({ frame: 1});//517,712,820

// 使用 setTimeout 递归循环，不使用 setInterval。
// queryDelay = 0 表示：上一轮完成后，立刻请求下一轮。
const QUERY_DELAY_MS = 0;

// 变量名保留 queryInterval，是为了兼容 home.vue 里可能访问 controlPanelRef.value.queryInterval。
let queryInterval: number | null = null;

const clearQueryTimer = () => {
  if (queryInterval !== null) {
    window.clearTimeout(queryInterval);
    queryInterval = null;
  }
};

const queryFrame = async () => {
  if (isProcessingFrame.value) {
    console.log('正在处理另一个压缩包，跳过本次请求');
    return;
  }

  if (isPaused.value) {
    console.log('系统处于暂停状态，跳过本次请求');
    return;
  }

  try {
    isProcessingFrame.value = true;
    isLoading.value = true;
    loadingMessage.value = '正在获取压缩包数据...';

    console.log('请求帧号:', frame.value.frame);

    loadingMessage.value = '正在接收压缩包数据...';
    const result = await query(frame.value);

    const dataLength = result?.data?.byteLength || result?.data?.length || 0;
    console.log('后端返回数据长度:', dataLength);

    if (currentConnectionStatus.value !== 'connected') {
      currentConnectionStatus.value = 'connected';
      emit('connection-status-changed', 'connected');
    }

    loadingMessage.value = '正在解压压缩包...';

    const zip = new JSZip();
    const zipContent = await zip.loadAsync(result.data);
    const fileNames = Object.keys(zipContent.files);

    console.log('压缩包内文件:', fileNames);

    loadingMessage.value = '正在读取配置文件...';

    let outputJson: any = null;
    let metricsData: any = null;

    if (zipContent.files['output.json']) {
      const outputContent = await zipContent.files['output.json'].async('string');
      outputJson = JSON.parse(outputContent);

      console.log('output.json 内容:', outputJson);

      if (outputJson.global_metrics && outputJson.global_metrics.modes) {
        metricsData = {
          camera: outputJson.global_metrics.modes.camera || {
            avg_error_rate: 0,
            avg_recall: 0
          },
          full: outputJson.global_metrics.modes.full || {
            avg_error_rate: 0,
            avg_recall: 0
          },
          lidar: outputJson.global_metrics.modes.lidar || {
            avg_error_rate: 0,
            avg_recall: 0
          },
          radar: outputJson.global_metrics.modes.radar || {
            avg_error_rate: 0,
            avg_recall: 0
          }
        };

        console.log('旧格式条形图数据:', metricsData);
      }
    }

    const metricJsonFile = fileNames.find(name => {
      const lower = name.toLowerCase();

      return !zipContent.files[name].dir &&
        (
          lower === 'metrics' ||
          lower.endsWith('/metrics') ||
          lower.includes('metrics.json') ||
          lower.includes('metric')
        );
    });

    if (metricJsonFile) {
      try {
        const metricContent = await zipContent.files[metricJsonFile].async('string');
        metricsData = JSON.parse(metricContent);

        console.log('读取新的指标 JSON:', metricJsonFile, metricsData);
      } catch (e) {
        console.warn('新的指标 JSON 解析失败:', metricJsonFile, e);
      }
    }

    const images: any = {};

    const originalImages = {
      camera: '',
      radar: '',
      lidar: '',
      other: '',
      micro: ''
    };

    const boxesImages = {
      camera: '',
      radar: '',
      lidar: '',
      other: ''
    };

    let fusionImage = '';

    const imageFiles = fileNames.filter(fileName => {
      const lower = fileName.toLowerCase();

      return (
        (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg')) &&
        !zipContent.files[fileName].dir
      );
    });

    loadingMessage.value = `正在处理 ${imageFiles.length} 张图片...`;

    // for (let i = 0; i < imageFiles.length; i++) {
    //   const rawFileName = imageFiles[i];
    //   const fileName = rawFileName.toLowerCase();

    //   loadingMessage.value = `正在处理图片 ${i + 1}/${imageFiles.length}...`;

    //   const imageData = await zipContent.files[rawFileName].async('base64');

    //   const mimeType =
    //     fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')
    //       ? 'image/jpeg'
    //       : 'image/png';

    //   const imageSrc = `data:${mimeType};base64,${imageData}`;

    //   // 融合检测图
    //   if (
    //     fileName.includes('fusion_model_fusion_vis') ||
    //     fileName.includes('fusion_vis')
    //   ) {
    //     fusionImage = imageSrc;
    //   }

    //   // 5 张原始图
    //   else if (fileName.includes('visible_raw')) {
    //     originalImages.camera = imageSrc;
    //   } else if (fileName.includes('infrared_raw')) {
    //     originalImages.other = imageSrc;
    //   } else if (fileName.includes('lidar_raw')) {
    //     originalImages.lidar = imageSrc;
    //   } else if (fileName.includes('micro_raw')) {
    //     originalImages.micro = imageSrc;
    //   } else if (fileName.includes('mmwave_raw')) {
    //     originalImages.radar = imageSrc;
    //   }

    //   // 4 张单模态推理图
    //   else if (fileName.includes('visible_single_pred')) {
    //     boxesImages.camera = imageSrc;
    //   } else if (fileName.includes('infrared_single_pred')) {
    //     boxesImages.other = imageSrc;
    //   } else if (fileName.includes('micro_single_pred')) {
    //     boxesImages.lidar = imageSrc;
    //   } else if (fileName.includes('mmwave_single_pred')) {
    //     boxesImages.radar = imageSrc;
    //   }
    // }
    loadingMessage.value = `正在并行处理 ${imageFiles.length} 张图片...`;

    const imageResults = await Promise.all(
      imageFiles.map(async (rawFileName) => {
        const fileName = rawFileName.toLowerCase();

        const imageData = await zipContent.files[rawFileName].async('base64');

        const mimeType =
          fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')
            ? 'image/jpeg'
            : 'image/png';

        return {
          fileName,
          imageSrc: `data:${mimeType};base64,${imageData}`
        };
      })
    );

    for (const { fileName, imageSrc } of imageResults) {
      if (
        fileName.includes('fusion_model_fusion_vis') ||
        fileName.includes('fusion_vis')
      ) {
        fusionImage = imageSrc;
      }

      else if (fileName.includes('visible_raw')) {
        originalImages.camera = imageSrc;
      } else if (fileName.includes('infrared_raw')) {
        originalImages.other = imageSrc;
      } else if (fileName.includes('lidar_raw')) {
        originalImages.lidar = imageSrc;
      } else if (fileName.includes('micro_raw')) {
        originalImages.micro = imageSrc;
      } else if (fileName.includes('mmwave_raw')) {
        originalImages.radar = imageSrc;
      }

      else if (fileName.includes('visible_single_pred')) {
        boxesImages.camera = imageSrc;
      } else if (fileName.includes('infrared_single_pred')) {
        boxesImages.other = imageSrc;
      } else if (fileName.includes('micro_single_pred')) {
        boxesImages.lidar = imageSrc;
      } else if (fileName.includes('mmwave_single_pred')) {
        boxesImages.radar = imageSrc;
      }
    }

    images.originalImages = originalImages;
    images.boxesImages = boxesImages;
    images.fusionImage = fusionImage;

    loadingMessage.value = '正在更新图像显示...';
    emit('images-updated', images);

    const displayJson = metricsData || outputJson;

    if (displayJson) {
      loadingMessage.value = '正在更新数据显示...';

      emit('output-json-updated', displayJson);
      emit('metrics-updated', displayJson);

      console.log('已发送展示 JSON 到 DataDisplay:', displayJson);
    }

    frame.value.frame++;
    console.log('帧号已更新为:', frame.value.frame);

    loadingMessage.value = '处理完成';
  } catch (error) {
    console.error('查询失败:', error);

    loadingMessage.value = '处理失败';
    currentConnectionStatus.value = 'disconnected';
    emit('connection-status-changed', 'disconnected');
  } finally {
    // 这里立刻释放锁，保证下一轮可以马上请求。
    isLoading.value = false;
    loadingMessage.value = '';
    isProcessingFrame.value = false;
  }
};

/**
 * 连续查询循环：
 * 上一轮 queryFrame 完整结束后，立刻请求下一轮。
 * 不再人为等待 5 秒。
 */
const runQueryLoop = async () => {
  if (!isProcessing.value || isPaused.value) {
    return;
  }

  const startTime = Date.now();

  await queryFrame();

  const elapsed = Date.now() - startTime;
  console.log(`本轮完整耗时: ${elapsed}ms，准备立即请求下一轮`);

  if (!isProcessing.value || isPaused.value) {
    return;
  }

  clearQueryTimer();

  queryInterval = window.setTimeout(() => {
    runQueryLoop();
  }, QUERY_DELAY_MS);
};

const handleStartProcessing = async () => {
  if (isPaused.value) {
    isPaused.value = false;
    emit('resume-processing');

    clearQueryTimer();
    runQueryLoop();

    return;
  }

  if (isProcessing.value) {
    console.log('已经在处理中，忽略重复开始');
    return;
  }

  isProcessing.value = true;
  isPaused.value = false;

  emit('start-processing');

  clearQueryTimer();

  // 第一轮立即执行，后续由 runQueryLoop 自动连续执行。
  runQueryLoop();
};

const handlePauseProcessing = () => {
  isPaused.value = true;

  clearQueryTimer();

  emit('pause-processing');
};

const handleStopProcessing = () => {
  isProcessing.value = false;
  isPaused.value = false;

  clearQueryTimer();

  emit('stop-processing');
};

const updateMessage = (message: string) => {
  console.log('ControlPanel message:', message);
};

const setConnectionStatus = (status: 'connected' | 'disconnected' | 'connecting') => {
  if (currentConnectionStatus.value !== status) {
    currentConnectionStatus.value = status;
    emit('connection-status-changed', status);
  }
};

const stopProcessing = () => {
  isProcessing.value = false;
  isPaused.value = false;

  clearQueryTimer();
};

defineExpose({
  updateMessage,
  setConnectionStatus,
  stopProcessing,
  isProcessing,
  isPaused,
  queryInterval
});
</script>

<style scoped>
.control-panel {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.control-header {
  padding: 0 2%;
  background: #1890FF;
  color: white;
  border-bottom: 1px solid #f0f0f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 44px;
  min-height: 44px;
  flex-shrink: 0;
}

.panel-title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.control-content {
  flex: 1;
  padding: 2%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2%;
}

.button-group {
  display: flex;
  gap: 1.5%;
}

.action-button {
  flex: 1;
  font-size: 14px;
  font-weight: 600;
}

.status-section {
  display: flex;
  flex-direction: column;
  gap: 1.5%;
  padding: 1.5%;
  background: #f9f9f9;
  border-radius: 6px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 1%;
  font-size: 13px;
}

.status-label {
  font-weight: 600;
  color: #333;
  min-width: 15%;
}

.status-value {
  flex: 1;
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: 600;
}

.status-processing {
  color: #e6a23c;
  background: #fdf6ec;
}

.status-idle {
  color: #67c23a;
  background: #f0f9ff;
}

.status-paused {
  color: #e6a23c;
  background: #fdf6ec;
}

.action-button.paused {
  background-color: #ffc107 !important;
  border-color: #ffc107 !important;
  color: #333 !important;
}

.action-button.paused .el-icon {
  color: #333 !important;
}

.status-connected {
  color: #67c23a;
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-disconnected {
  color: #f56c6c;
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-connecting {
  color: #e6a23c;
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.message-section {
  display: flex;
  flex-direction: column;
  gap: 1%;
}

.message-label {
  font-size: 13px;
  font-weight: 600;
  color: #333;
}

.message-box {
  padding: 1.5%;
  background: #f5f7fa;
  border-radius: 6px;
  border-left: 3px solid #1890FF;
  font-size: 12px;
  color: #666;
  max-height: 10vh;
  overflow-y: auto;
  word-break: break-word;
  white-space: pre-wrap;
}

.settings-section {
  border-top: 1px solid #f0f0f0;
  padding-top: 12px;
}

.setting-item {
  display: flex;
  flex-direction: column;
  gap: 1%;
  margin-bottom: 1.5%;
}

.setting-label {
  font-size: 12px;
  font-weight: 600;
  color: #333;
}

@media (max-width: 768px) {
  .control-content {
    gap: 12px;
  }

  .button-group {
    flex-direction: column;
  }
}
</style>