<script lang="ts" setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { ElMessage } from 'element-plus';
import ImageGallery from '@/components/ImageGallery.vue';
import DetectionViewer from '@/components/DetectionViewer.vue';
import ControlPanel from '@/components/ControlPanel.vue';
import DataDisplay from '@/components/DataDisplay.vue';

// ============ 组件引用 ============
const imageGalleryRef = ref();
const fusionViewerRef = ref();
const controlPanelRef = ref();
const dataDisplayRef = ref();

// ============ 状态管理 ============
const systemOnline = ref(true);
const llmOutputText = ref('');

const settings = ref({
  refreshInterval: 1000,
  wsAddress: 'ws://localhost:8000'
});

let websocket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let dataRefreshTimer: ReturnType<typeof setInterval> | null = null;
let detectionTimer: ReturnType<typeof setInterval> | null = null;
let processingTimeout: ReturnType<typeof setTimeout> | null = null;

// ============ 图片类型 ============
type OriginalImages = {
  camera: string;
  radar: string;
  lidar: string;
  other: string;
  micro?: string;
};

type BoxesImages = {
  camera: string;
  radar: string;
  lidar: string;
  other: string;
};

interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  confidence: number;
}

interface MonoImageSlot {
  label: string;
  src: string;
}

const monoImageSlots = ref<MonoImageSlot[]>([
  { label: '可见光单模态', src: '' },
  { label: '毫米波雷达单模态', src: '' },
  { label: '红外单模态', src: '' },
  { label: '微光夜视单模态', src: '' },
  { label: '', src: '' }
]);

const toImageSrc = (value?: string): string => {
  if (!value) return '';

  const isUrlPath =
    value.startsWith('data:') ||
    value.startsWith('http://') ||
    value.startsWith('https://') ||
    value.startsWith('/') ||
    value.includes('\\');

  if (isUrlPath) return value;

  return `data:image/png;base64,${value}`;
};

const updateMonoImages = (boxesImages?: Partial<BoxesImages>) => {
  monoImageSlots.value = [
    {
      label: '可见光单模态',
      src: toImageSrc(boxesImages?.camera)
    },
    {
      label: '毫米波雷达单模态',
      src: toImageSrc(boxesImages?.radar)
    },
    {
      label: '红外单模态',
      src: toImageSrc(boxesImages?.other)
    },
    {
      label: '微光夜视单模态',
      src: toImageSrc(boxesImages?.lidar)
    },
    {
      label: '',
      src: ''
    }
  ];
};

const updateMonoImagesFromArray = (imageData: string[]) => {
  monoImageSlots.value = [
    {
      label: '可见光单模态',
      src: toImageSrc(imageData[0])
    },
    {
      label: '毫米波雷达单模态',
      src: toImageSrc(imageData[1])
    },
    {
      label: '红外单模态',
      src: toImageSrc(imageData[2])
    },
    {
      label: '微光夜视单模态',
      src: toImageSrc(imageData[3])
    },
    {
      label: '',
      src: ''
    }
  ];
};

// ============ WebSocket 连接管理 ============
const connectWebSocket = () => {
  try {
    websocket = new WebSocket(settings.value.wsAddress);

    websocket.onopen = () => {
      console.log('WebSocket 连接已建立');
      controlPanelRef.value?.setConnectionStatus('connected');
      systemOnline.value = true;
      controlPanelRef.value?.updateMessage('WebSocket 已连接');
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('收到 WebSocket 数据:', data);
        processAndVisualizeData(data);
      } catch (error) {
        console.error('解析 WebSocket 数据失败:', error);
        ElMessage.error('数据格式错误');
      }
    };

    websocket.onclose = () => {
      console.log('WebSocket 连接已关闭');

      if (systemOnline.value) {
        controlPanelRef.value?.updateMessage('WebSocket 连接已关闭');
      }

      if (websocket?.readyState !== WebSocket.CLOSED) {
        scheduleReconnect();
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket 错误:', error);
      controlPanelRef.value?.updateMessage('WebSocket 连接错误');
    };
  } catch (error) {
    console.error('WebSocket 连接失败:', error);
    scheduleReconnect();
  }
};

const scheduleReconnect = () => {
  if (reconnectTimer) clearTimeout(reconnectTimer);

  reconnectTimer = setTimeout(() => {
    controlPanelRef.value?.setConnectionStatus('connecting');
    connectWebSocket();
  }, 5000);
};

// ============ 数据处理函数 ============
const processAndVisualizeData = async (data: {
  detections?: Detection[];
  outputs?: any;
  images?: {
    camera?: string;
    radar?: string;
    lidar1?: string;
    lidar2?: string;
    detectionImages?: string[];
  };
}) => {
  const startTime = Date.now();

  if (processingTimeout) {
    clearTimeout(processingTimeout);
    processingTimeout = null;
  }

  processingTimeout = setTimeout(() => {
    if (Date.now() - startTime > 5000) {
      console.warn('数据处理超时（>5秒）');
      ElMessage.warning('数据处理超时，部分功能可能未完全加载');
    }
  }, 5000);

  try {
    if (!data || typeof data !== 'object') {
      console.error('无效的数据格式');
      ElMessage.error('收到无效的数据格式');
      return;
    }

    if (data.images && imageGalleryRef.value) {
      if (typeof data.images !== 'object') {
        console.error('图像数据格式错误');
        ElMessage.error('图像数据格式错误');
      } else {
        await imageGalleryRef.value.updateImages(data.images);

        if (data.images.detectionImages) {
          updateMonoImagesFromArray(data.images.detectionImages);
          await fusionViewerRef.value?.updateDetectionImages(data.images.detectionImages);
        }
      }
    } else if (data.images?.detectionImages) {
      updateMonoImagesFromArray(data.images.detectionImages);
      await fusionViewerRef.value?.updateDetectionImages(data.images.detectionImages);
    }

    const processingTime = Date.now() - startTime;
    if (processingTime > 5000) {
      console.warn(`数据处理时间过长，耗时: ${processingTime}ms (>5秒)`);
      ElMessage.warning(`数据处理时间过长: ${Math.round(processingTime / 1000)}秒`);
    }
  } catch (error) {
    console.error('【主流程错误】数据处理错误:', error);
    ElMessage.error('数据处理错误');
  } finally {
    if (processingTimeout) {
      clearTimeout(processingTimeout);
      processingTimeout = null;
    }
  }
};

// ============ 事件处理 ============
const handleStartProcessing = async () => {
  console.log('开始处理事件已触发，实际推理由 ControlPanel.vue 执行');
};

const handlePauseProcessing = () => {
  if (detectionTimer) {
    clearInterval(detectionTimer);
    detectionTimer = null;
  }
  ElMessage.info('处理已暂停');
};

const handleResumeProcessing = () => {
  console.log('恢复处理事件已触发，实际恢复逻辑由 ControlPanel.vue 执行');
  ElMessage.info('处理已恢复');
};

const handleStopProcessing = () => {
  controlPanelRef.value?.stopProcessing();

  if (detectionTimer) {
    clearInterval(detectionTimer);
    detectionTimer = null;
  }

  if (dataRefreshTimer) {
    clearInterval(dataRefreshTimer);
    dataRefreshTimer = null;
  }

  if (websocket) {
    websocket.close();
    websocket = null;
    console.log('WebSocket连接已关闭');
  }

  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  imageGalleryRef.value?.updateImages({
    originalImages: {
      camera: '',
      radar: '',
      lidar: '',
      other: '',
      micro: ''
    }
  });

  updateMonoImages();

  fusionViewerRef.value?.updateDetectionImages({
    boxesImages: {
      camera: '',
      radar: '',
      lidar: '',
      other: ''
    },
    fusionImage: ''
  });

  dataDisplayRef.value?.clearData?.();

  llmOutputText.value = '';

  if (controlPanelRef.value && controlPanelRef.value.queryInterval) {
    clearInterval(controlPanelRef.value.queryInterval);
    controlPanelRef.value.queryInterval = null;
  }

  ElMessage.info('处理已停止');
};

const handleConnectionStatusChanged = (
  status: 'connected' | 'disconnected' | 'connecting'
) => {
  console.log('连接状态已更改:', status);
  systemOnline.value = status === 'connected';
};

const handleOutputJsonUpdated = (outputJson: any) => {
  console.log('收到 output.json 数据:', outputJson);
  dataDisplayRef.value?.updateData?.(outputJson);
  llmOutputText.value = '';
};

const handleImagesUpdated = (images: {
  originalImages?: OriginalImages;
  boxesImages?: BoxesImages;
  fusionImage?: string;
}) => {
  console.log('收到更新图片数据:', images);

  if (imageGalleryRef.value && images.originalImages) {
    imageGalleryRef.value.updateImages({
      originalImages: images.originalImages
    });
  }

  if (images.boxesImages || images.fusionImage !== undefined) {
    const detectionImages = {
      boxesImages: images.boxesImages || {
        camera: '',
        radar: '',
        lidar: '',
        other: ''
      },
      fusionImage: images.fusionImage || ''
    };

    updateMonoImages(images.boxesImages);
    fusionViewerRef.value?.updateDetectionImages(detectionImages);
  }
};

const handleDetectionsUpdated = async (detectionData: {
  boxesImages?: BoxesImages;
  fusionImage?: string;
  detections?: any[];
}) => {
  console.log('收到检测图像数据:', detectionData);

  updateMonoImages(detectionData.boxesImages);
  await fusionViewerRef.value?.updateDetectionImages(detectionData);

  if (detectionData.detections) {
    dataDisplayRef.value?.updateData?.(detectionData.detections);
  }
};

const handleMetricsUpdated = (metrics: any) => {
  console.log('收到条形图数据:', metrics);
  dataDisplayRef.value?.updateMetrics?.(metrics);
};

// ============ 生命周期钩子 ============
onMounted(async () => {
  console.log('应用已初始化');
  connectWebSocket();
});

onBeforeUnmount(() => {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  if (dataRefreshTimer) clearInterval(dataRefreshTimer);
  if (detectionTimer) clearInterval(detectionTimer);
  if (websocket) websocket.close();
});
</script>

<template>
  <div class="app-container">
    <el-header class="app-header">
      <h1 class="app-title">多模态融合感知大模型原型系统</h1>
    </el-header>

    <el-container class="main-container">
      <!-- 输入数据：第一列保持 5 个等高框 -->
      <div class="module-wrapper module-input">
        <div class="module-header">
          <span class="module-title">输入数据</span>
        </div>
        <ImageGallery ref="imageGalleryRef" />
      </div>

      <!-- 单模态感知：不用 DetectionViewer，直接做成和输入数据一样的 5 个等高框 -->
      <div class="module-wrapper module-mono">
        <div class="module-header">
          <span class="module-title">单模态感知</span>
        </div>

        <div class="mono-gallery">
          <div
            v-for="(item, index) in monoImageSlots"
            :key="'mono-' + index"
            class="mono-card"
          >
            <img
              v-if="item.src"
              :src="item.src"
              :alt="item.label"
              class="mono-image"
            />
            <div v-else class="mono-placeholder">
              <span v-if="index < 4">等待图片数据...</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 多模态融合感知 -->
      <div class="module-wrapper module-fusion">
        <div class="module-header">
          <span class="module-title">多模态融合感知</span>
        </div>
        <DetectionViewer ref="fusionViewerRef" />
      </div>

      <!-- 场景语义输出：右侧白框缩小 -->
      <div class="module-wrapper module-semantic">
        <div class="module-header">
          <span class="module-title">场景语义输出</span>
        </div>
        <div class="semantic-content">
          {{ llmOutputText }}
        </div>
      </div>

      <!-- 底部：控制面板 + 目标定位 -->
      <div class="fusion-bottom-row">
        <div class="bottom-control-box">
          <ControlPanel
            ref="controlPanelRef"
            :connection-status="systemOnline ? 'connected' : 'disconnected'"
            @start-processing="handleStartProcessing"
            @pause-processing="handlePauseProcessing"
            @resume-processing="handleResumeProcessing"
            @stop-processing="handleStopProcessing"
            @images-updated="handleImagesUpdated"
            @detections-updated="handleDetectionsUpdated"
            @connection-status-changed="handleConnectionStatusChanged"
            @output-json-updated="handleOutputJsonUpdated"
            @metrics-updated="handleMetricsUpdated"
          />
        </div>

        <div class="bottom-data-box">
          <DataDisplay ref="dataDisplayRef" />
        </div>
      </div>
    </el-container>
  </div>
</template>

<style scoped>
/* ================= 全局尺寸变量 ================= */
.app-container {
  --side-strip-width: 146px;   /* 输入数据、单模态感知列宽一致 */
  --semantic-width: 230px;     /* 场景语义输出白框缩小 */
  --control-width: 205px;      /* 控制面板加宽 */
  --header-height: 26px;
  --bottom-row-height: 150px;
  --gap-size: 4px;
}

/* ================= 页面整体 ================= */
.app-container {
  display: flex;
  flex-direction: column;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: #ffffff;
}

.app-header {
  height: 28px;
  min-height: 28px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0050b3;
  color: #ffffff;
  box-shadow: none;
  flex-shrink: 0;
}

.app-title {
  margin: 0;
  color: #ffffff;
  font-size: 18px;
  font-weight: 700;
  line-height: 28px;
}

/* ================= 主布局 ================= */
.main-container {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns:
    var(--side-strip-width)
    var(--side-strip-width)
    minmax(760px, 1fr)
    var(--semantic-width);
  grid-template-rows: minmax(0, 1fr) var(--bottom-row-height);
  gap: var(--gap-size);
  padding: var(--gap-size);
  box-sizing: border-box;
  overflow: hidden;
  background: #ffffff;
}

/* ================= 通用模块 ================= */
.module-wrapper {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #ffffff;
  border-radius: 0;
  box-shadow: none;
}

.module-header {
  height: var(--header-height);
  min-height: var(--header-height);
  padding: 0 7px;
  display: flex;
  align-items: center;
  background: #1890ff;
  color: #ffffff;
  box-sizing: border-box;
  flex-shrink: 0;
}

.module-title {
  margin: 0;
  font-size: 14px;
  font-weight: 700;
  color: #ffffff;
  white-space: nowrap;
}

/* 布局定位 */
.module-input {
  grid-column: 1 / 2;
  grid-row: 1 / 3;
}

.module-mono {
  grid-column: 2 / 3;
  grid-row: 1 / 3;
}

.module-fusion {
  grid-column: 3 / 4;
  grid-row: 1 / 2;
}

.module-semantic {
  grid-column: 4 / 5;
  grid-row: 1 / 2;
}

.semantic-content {
  flex: 1;
  min-height: 0;
  padding: 8px;
  background: #ffffff;
  color: #333333;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow: auto;
}

.fusion-bottom-row {
  grid-column: 3 / 5;
  grid-row: 2 / 3;
  min-width: 0;
  min-height: 0;
  display: flex;
  gap: var(--gap-size);
  overflow: hidden;
}

.bottom-control-box {
  width: var(--control-width);
  flex: 0 0 var(--control-width);
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.bottom-data-box {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

/* ================= 输入数据：第一列保持 5 个等高框 ================= */
.module-input :deep(.image-gallery) {
  width: 100%;
  height: 100%;
  padding: 3px;
  border-radius: 0;
  box-sizing: border-box;
  background: #e6f7ff;
  overflow: hidden;
}

.module-input :deep(.gallery-container) {
  display: grid !important;
  grid-template-columns: 1fr !important;
  grid-template-rows: repeat(5, minmax(0, 1fr)) !important;
  gap: 4px !important;
  width: 100%;
  height: 100%;
}

/* 保持之前的输入数据顺序：可见光、毫米波、红外、点云、微光 */
.module-input :deep(.image-card:nth-child(1)) { order: 1; }
.module-input :deep(.image-card:nth-child(2)) { order: 2; }
.module-input :deep(.image-card:nth-child(3)) { order: 4; }
.module-input :deep(.image-card:nth-child(4)) { order: 3; }
.module-input :deep(.image-card:nth-child(5)) { order: 5; }

.module-input :deep(.camera-card),
.module-input :deep(.radar-card),
.module-input :deep(.lidar-card) {
  grid-row: auto !important;
  grid-column: auto !important;
}

.module-input :deep(.image-card) {
  min-height: 0 !important;
  height: 100% !important;
  border-radius: 2px;
  box-shadow: none;
  background: #ffffff;
}

.module-input :deep(.image-card:hover) {
  transform: none;
  box-shadow: none;
}

.module-input :deep(.image-wrapper) {
  width: 100%;
  height: 100%;
  padding: 2px;
  overflow: hidden;
  background: #f0f2f5;
}

.module-input :deep(.gallery-image) {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.module-input :deep(.placeholder-text),
.module-input :deep(.loading-text) {
  font-size: 10px;
}

.module-input :deep(.load-time) {
  left: 3px;
  right: auto;
  bottom: 2px;
  font-size: 8px;
  padding: 1px 2px;
}

/* ================= 单模态感知：和输入数据一样，5 个等高框 ================= */
.mono-gallery {
  flex: 1;
  min-height: 0;
  width: 100%;
  height: 100%;
  padding: 3px;
  box-sizing: border-box;
  background: #e6f7ff;
  display: grid;
  grid-template-columns: 1fr;
  grid-template-rows: repeat(5, minmax(0, 1fr));
  gap: 4px;
  overflow: hidden;
}

.mono-card {
  min-height: 0;
  height: 100%;
  width: 100%;
  border-radius: 2px;
  background: #ffffff;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.mono-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.mono-placeholder {
  width: 100%;
  height: 100%;
  padding: 3px;
  box-sizing: border-box;
  color: #909399;
  font-size: 10px;
  background: #f7f7f7;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}

/* ================= 多模态融合感知：只显示融合图 ================= */
.module-fusion :deep(.detection-viewer) {
  width: 100%;
  height: 100%;
  border-radius: 0;
  background: #e6f7ff;
}

.module-fusion :deep(.viewer-container) {
  width: 100%;
  height: 100%;
  padding: 3px;
  box-sizing: border-box;
  min-height: 0;
  background: #e6f7ff;
}

.module-fusion :deep(.top-section) {
  display: none !important;
}

.module-fusion :deep(.bottom-section) {
  width: 100%;
  height: 100%;
  min-height: 0;
  padding: 0;
  overflow: hidden;
}

.module-fusion :deep(.bottom-image-card) {
  width: 100%;
  height: 100%;
  min-height: 0;
  border-radius: 2px;
  box-shadow: none;
  background: #ffffff;
}

.module-fusion :deep(.image-wrapper) {
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #f7f7f7;
  display: flex;
  align-items: center;
  justify-content: center;
}

.module-fusion :deep(.detection-image) {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

/* ================= 控制面板 ================= */
.bottom-control-box :deep(.control-panel) {
  width: 100%;
  height: 100%;
  border-radius: 0;
  box-shadow: none;
  overflow: hidden;
}

.bottom-control-box :deep(.control-header) {
  height: var(--header-height);
  min-height: var(--header-height);
  padding: 0 7px;
  background: #1890ff;
}

.bottom-control-box :deep(.panel-title) {
  font-size: 12px;
  font-weight: 700;
}

.bottom-control-box :deep(.control-content) {
  padding: 4px;
  gap: 3px;
  overflow: hidden;
}

.bottom-control-box :deep(.button-group) {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 2px;
  width: 100%;
}

.bottom-control-box :deep(.action-button) {
  width: 100%;
  height: 22px;
  min-height: 22px;
  padding: 0 1px !important;
  margin: 0 !important;
  border-radius: 3px;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 7px !important;
  font-weight: 500;
  line-height: 1 !important;
  overflow: hidden;
}

.bottom-control-box :deep(.action-button > span) {
  width: 100%;
  height: 100%;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 1px;
  font-size: 7px !important;
  line-height: 1 !important;
  white-space: nowrap;
  overflow: hidden;
}

.bottom-control-box :deep(.action-button span),
.bottom-control-box :deep(.action-button .el-button__text),
.bottom-control-box :deep(.action-button .el-icon) {
  font-size: 7px !important;
  line-height: 1 !important;
}

.bottom-control-box :deep(.status-section) {
  gap: 1px;
  padding: 2px;
  border-radius: 3px;
}

.bottom-control-box :deep(.status-item) {
  font-size: 9px;
  gap: 2px;
  line-height: 1.15;
}

.bottom-control-box :deep(.status-label) {
  min-width: 44px;
  font-size: 9px;
}

.bottom-control-box :deep(.status-value) {
  padding: 1px 2px;
  font-size: 9px;
}

/* ================= 目标定位：保留柱状图，隐藏左侧文字内容 ================= */
.bottom-data-box :deep(.data-display) {
  width: 100%;
  height: 100%;
  border-radius: 0;
  box-shadow: none;
}

.bottom-data-box :deep(.data-header) {
  height: var(--header-height);
  min-height: var(--header-height);
  padding: 0 10px;
  background: #1890ff;
}

.bottom-data-box :deep(.panel-title) {
  font-size: 14px;
  font-weight: 700;
}

.bottom-data-box :deep(.data-content) {
  padding: 4px 6px 3px;
  overflow: hidden;
}

.bottom-data-box :deep(.content-three-columns) {
  display: flex;
  gap: 8px;
  height: 100%;
  overflow: hidden;
}

.bottom-data-box :deep(.column-left) {
  display: none !important;
}

.bottom-data-box :deep(.column-middle),
.bottom-data-box :deep(.column-right) {
  flex: 1 1 0;
  min-width: 0;
  padding: 4px;
  border-radius: 4px;
  background: #ffffff;
}

.bottom-data-box :deep(.chart-title) {
  font-size: 11px;
  margin-bottom: 2px;
}

.bottom-data-box :deep(.bar-chart.horizontal) {
  gap: 2px;
  padding-top: 2px;
}

.bottom-data-box :deep(.bar-item.horizontal) {
  min-height: 15px;
}

.bottom-data-box :deep(.bar-label.horizontal) {
  width: 86px;
  font-size: 9px;
}

.bottom-data-box :deep(.bar-wrapper.horizontal) {
  height: 15px;
}

.bottom-data-box :deep(.bar-value-on-bar) {
  font-size: 8px;
}

/* ================= 小屏适配 ================= */
@media (max-width: 1400px) {
  .app-container {
    --side-strip-width: 138px;
    --semantic-width: 215px;
    --control-width: 192px;
    --bottom-row-height: 144px;
  }

  .main-container {
    grid-template-columns:
      var(--side-strip-width)
      var(--side-strip-width)
      minmax(650px, 1fr)
      var(--semantic-width);
  }
}

@media (max-width: 1200px) {
  .app-container {
    --side-strip-width: 128px;
    --semantic-width: 200px;
    --control-width: 178px;
    --bottom-row-height: 136px;
  }

  .main-container {
    grid-template-columns:
      var(--side-strip-width)
      var(--side-strip-width)
      minmax(520px, 1fr)
      var(--semantic-width);
  }

  .module-title {
    font-size: 13px;
  }
}
</style>
