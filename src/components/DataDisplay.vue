<template>
  <div class="data-display">
    <div class="data-header">
      <h3 class="panel-title">目标定位</h3>
      <div class="header-actions">
        <el-button type="text" size="small" @click="resetData">
          重置
        </el-button>
      </div>
    </div>

    <div class="data-content">
      <div v-if="isLoading" class="loading-state">
        <div class="loading-text">数据加载中...</div>
      </div>

      <div v-else class="content-three-columns">
        <!-- 左侧：融合模态识别结果 -->
        <div class="column-left">
          <div v-if="detections.length > 0" class="llm-summary">
            <div class="summary-paragraph">
              <span class="summary-text">
                <!-- 目标定位结果：<strong>目标类别</strong>为<strong>{{ targetClass }}</strong>， -->
                目标定位结果：<strong>识别结果</strong>为共检测到
                <span class="highlight">{{ totalDetections }}</span>
                个目标，
                <strong>定位结果</strong>包含<strong>融合模态三维坐标</strong>和<strong>检测框像素坐标</strong>，
                格式为<strong>xyz=(x, y, z)m</strong>和<strong>[x1, y1, x2, y2] pixel</strong>。
                <strong>目标信息如下：</strong>
              </span>
            </div>

            <div class="instance-section">
              <div class="instance-text-list">
                <div
                  v-for="(detection, index) in detections"
                  :key="index"
                  class="instance-text-item"
                >
                  <span class="instance-number">{{ index + 1 }}.</span>
                  <span class="instance-name">目标 {{ index + 1 }}：</span>
                  <span class="instance-detail">
                    类别 <span class="detail-value">{{ detection.class }}</span> ｜
                    置信度
                    <span class="detail-value">
                      {{ (detection.confidence * 100).toFixed(1) }}%
                    </span>
                    ｜
                    三维坐标
                    <span class="detail-value">
                      {{ detection.positionText || formatXYZ(detection.xyz) }}
                    </span>
                    ｜
                    像素框
                    <span class="detail-value">
                      {{ formatPixelBox(detection.pixelBox) }}
                    </span>
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div v-else class="empty-state">
            暂无检测数据
          </div>
        </div>

        <!-- 中间：平均精确率，包含融合模态和单模态 -->
        <div class="column-middle">
          <div class="chart-section">
            <div class="chart-title">平均精确率</div>
            <div class="bar-chart horizontal">
              <div
                v-for="(item, index) in chartData.precision"
                :key="'precision-' + index"
                class="bar-item horizontal"
              >
                <div class="bar-label horizontal" :title="item.label">
                  {{ item.label }}
                </div>
                <div class="bar-wrapper horizontal">
                  <div
                    class="bar horizontal"
                    :class="item.colorClass"
                    :style="{ width: item.value + '%' }"
                  >
                    <span class="bar-value-on-bar">
                      {{ item.value.toFixed(1) }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 右侧：平均召回率，包含融合模态和单模态 -->
        <div class="column-right">
          <div class="chart-section">
            <div class="chart-title">平均召回率</div>
            <div class="bar-chart horizontal">
              <div
                v-for="(item, index) in chartData.recall"
                :key="'recall-' + index"
                class="bar-item horizontal"
              >
                <div class="bar-label horizontal" :title="item.label">
                  {{ item.label }}
                </div>
                <div class="bar-wrapper horizontal">
                  <div
                    class="bar horizontal"
                    :class="item.colorClass"
                    :style="{ width: item.value + '%' }"
                  >
                    <span class="bar-value-on-bar">
                      {{ item.value.toFixed(1) }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from 'vue';
import { ElButton } from 'element-plus';

interface Lidar3DInfo {
  center: [number, number, number];
  size: [number, number, number];
  yaw: number;
}

interface Detection {
  class: string;
  confidence: number;
  height: number;
  lidar3DInfo: Lidar3DInfo;
  type: 'gt' | 'pred';
  width: number;
  x: number;
  y: number;
  /** 图像像素框：[x1, y1, x2, y2] */
  pixelBox?: [number, number, number, number];
  /** 融合模态推理/毫米波匹配得到的三维坐标：[x_forward_m, y_left_m, z_up_m] */
  xyz?: [number, number, number] | null;
  positionText?: string;
  matchedWithMmwave?: boolean;
  id?: string;
  timestamp?: number;
}

interface DataSummary {
  totalDetections: number;
  averageConfidence: number;
  processingTime: number;
  timestamp: number;
}

interface ChartItem {
  label: string;
  value: number;
  colorClass: string;
}

interface ChartData {
  precision: ChartItem[];
  recall: ChartItem[];
}

const detections = ref<Detection[]>([]);
const selectedRowData = ref<Detection | null>(null);
const isLoading = ref(true);
const targetClass = ref('目标');

const dataSummary = ref<DataSummary>({
  totalDetections: 0,
  averageConfidence: 0,
  processingTime: 0,
  timestamp: Date.now()
});

const defaultPrecisionData = (): ChartItem[] => [
  { label: '融合模态', value: 0, colorClass: 'bar-full' },
  { label: '可见光单模态', value: 0, colorClass: 'bar-camera' },
  { label: '红外单模态', value: 0, colorClass: 'bar-infrared' },
  { label: '微光单模态', value: 0, colorClass: 'bar-micro' },
  { label: '毫米波雷达单模态', value: 0, colorClass: 'bar-radar' }
];

const defaultRecallData = (): ChartItem[] => [
  { label: '融合模态', value: 0, colorClass: 'bar-full' },
  { label: '可见光单模态', value: 0, colorClass: 'bar-camera' },
  { label: '红外单模态', value: 0, colorClass: 'bar-infrared' },
  { label: '微光单模态', value: 0, colorClass: 'bar-micro' },
  { label: '毫米波雷达单模态', value: 0, colorClass: 'bar-radar' }
];

const chartData = ref<ChartData>({
  precision: defaultPrecisionData(),
  recall: defaultRecallData()
});

const totalDetections = computed(() => detections.value.length);

const formatVector = (vector: number[]): string => {
  if (!vector || vector.length === 0) return '-';
  return vector.map(v => Number(v || 0).toFixed(2)).join(', ');
};

const formatXYZ = (xyz?: number[] | null): string => {
  if (!xyz || xyz.length < 3) return '未匹配三维坐标';

  const x = Number(xyz[0] || 0).toFixed(1);
  const y = Number(xyz[1] || 0).toFixed(1);
  const z = Number(xyz[2] || 0).toFixed(1);

  return `xyz=(${x}, ${y}, ${z})m`;
};

const formatPixelBox = (pixelBox?: number[] | null): string => {
  if (!pixelBox || pixelBox.length < 4) return '-';

  const [x1, y1, x2, y2] = pixelBox.map(v => Math.round(Number(v || 0)));
  return `[${x1}, ${y1}, ${x2}, ${y2}] pixel`;
};

const toPercent = (value: any): number => {
  const num = Number(value);
  if (Number.isNaN(num)) return 0;

  const percent = num <= 1 ? num * 100 : num;
  return Math.max(0, Math.min(100, percent));
};

const getMetricValue = (metricObj: any, key: string): number => {
  if (!metricObj || typeof metricObj !== 'object') return 0;
  return toPercent(metricObj[key] ?? 0);
};

/**
 * 从融合模态 prediction_boxes 提取识别结果。
 * 展示信息优先使用融合模态输出中的：
 * 1. xyz_m / position_match：三维坐标，单位 m
 * 2. box_xyxy_pixel：[x1, y1, x2, y2]，单位 pixel
 */
const convertFusionBoxesToDetections = (fusionResult: any): Detection[] => {
  if (!fusionResult || !Array.isArray(fusionResult.prediction_boxes)) {
    return [];
  }

  const result: Detection[] = [];

  fusionResult.prediction_boxes.forEach((group: any) => {
    const boxes = group?.boxes || [];

    boxes.forEach((box: any) => {
      const pixelBox = box.box_xyxy_pixel || [0, 0, 0, 0];

      const x1 = Number(pixelBox[0] || 0);
      const y1 = Number(pixelBox[1] || 0);
      const x2 = Number(pixelBox[2] || 0);
      const y2 = Number(pixelBox[3] || 0);

      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      const width = Math.max(0, x2 - x1);
      const height = Math.max(0, y2 - y1);

      const positionMatch = box.position_match || {};

      const xyz: [number, number, number] | null = Array.isArray(box.xyz_m) && box.xyz_m.length >= 3
        ? [
            Number(box.xyz_m[0] || 0),
            Number(box.xyz_m[1] || 0),
            Number(box.xyz_m[2] || 0)
          ]
        : positionMatch.x_forward_m !== undefined
          ? [
              Number(positionMatch.x_forward_m || 0),
              Number(positionMatch.y_left_m || 0),
              Number(positionMatch.z_up_m || 0)
            ]
          : null;

      result.push({
        x: centerX,
        y: centerY,
        width,
        height,
        type: 'pred',
        class: box.class_name || String(box.class_id ?? 'unknown'),
        confidence: Number(box.score || 0),
        pixelBox: [x1, y1, x2, y2],
        xyz,
        positionText: box.position_text || '',
        matchedWithMmwave: Boolean(box.matched_with_mmwave),
        // 保留原字段，避免其他组件仍然读取 lidar3DInfo 时报错；这里 center 改为三维坐标。
        lidar3DInfo: {
          center: xyz || [0, 0, 0],
          size: [width, height, 0],
          yaw: 0
        }
      });
    });
  });

  return result;
};

/**
 * 更新指标图表。
 * 提取：
 * fusion_result.metrics.mean_precision / mean_recall
 * single_modal_results.visible.metrics.mean_precision / mean_recall
 * single_modal_results.infrared.metrics.mean_precision / mean_recall
 * single_modal_results.micro.metrics.mean_precision / mean_recall
 * single_modal_results.mmwave.metrics.mean_precision / mean_recall
 */
const updateMetrics = (metrics: any) => {
  if (!metrics) {
    isLoading.value = false;
    return;
  }

  console.log('DataDisplay 收到 metrics:', metrics);

  isLoading.value = false;

  const fusionMetrics = metrics?.fusion_result?.metrics || {};
  const singleResults = metrics?.single_modal_results || {};

  const items = [
    {
      label: '融合模态',
      metrics: fusionMetrics,
      colorClass: 'bar-full'
    },
    {
      label: '可见光单模态',
      metrics: singleResults?.visible?.metrics || {},
      colorClass: 'bar-camera'
    },
    {
      label: '红外单模态',
      metrics: singleResults?.infrared?.metrics || {},
      colorClass: 'bar-infrared'
    },
    {
      label: '微光单模态',
      metrics: singleResults?.micro?.metrics || {},
      colorClass: 'bar-micro'
    },
    {
      label: '毫米波雷达单模态',
      metrics: singleResults?.mmwave?.metrics || {},
      colorClass: 'bar-radar'
    }
  ];

  chartData.value.precision = items.map(item => ({
    label: item.label,
    value: getMetricValue(item.metrics, 'mean_precision'),
    colorClass: item.colorClass
  }));

  chartData.value.recall = items.map(item => ({
    label: item.label,
    value: getMetricValue(item.metrics, 'mean_recall'),
    colorClass: item.colorClass
  }));

  console.log('平均精确率/平均召回率已更新:', chartData.value);
};

const convertLabelsToDetections = (labels: any): Detection[] => {
  if (!labels) return [];

  const centers = labels.gt_center || [];
  const sizes = labels.gt_size || [];
  const classes = labels.gt_class || [];
  const angles = labels.gt_angle || [];
  const result: Detection[] = [];

  for (let i = 0; i < centers.length; i++) {
    const c = centers[i] || [];
    const s = sizes[i] || [];
    const clsScores = classes[i] || [];
    const angle = angles[i] || [];

    let className = 'unknown';
    let confidence = 0;

    if (Array.isArray(clsScores) && clsScores.length > 0) {
      const max = Math.max(...clsScores);
      const idx = clsScores.indexOf(max);
      className = String(idx);
      confidence = max;
    }

    const transformedCenter = [
      (c[0] !== undefined ? c[0] : 0) + 2.54,
      (c[1] !== undefined ? c[1] : 0) - 0.3,
      (c[2] !== undefined ? c[2] : 0) - 0.7
    ];

    let yaw = 0;
    if (Array.isArray(angle) && angle.length >= 2 && angle[1] !== undefined) {
      yaw = -Math.atan2(angle[0], angle[1]);
    } else if (angle[0] !== undefined) {
      yaw = angle[0];
    }

    result.push({
      x: transformedCenter[0],
      y: transformedCenter[1],
      width: s[0] ?? 0,
      height: s[1] ?? 0,
      type: 'gt',
      class: className,
      confidence,
      lidar3DInfo: {
        center: [transformedCenter[0], transformedCenter[1], transformedCenter[2]],
        size: [s[0] || 0, s[1] || 0, s[2] || 0],
        yaw
      }
    });
  }

  return result;
};

const updateData = (newData: any, summary?: Partial<DataSummary>) => {
  const ts = summary?.timestamp || Date.now();

  isLoading.value = true;

  try {
    console.log('DataDisplay 收到数据:', newData);

    let detectionsToAdd: Detection[] = [];
    let hasValidData = false;

    if (Array.isArray(newData)) {
      if (newData.length > 0) {
        detectionsToAdd = newData;
        hasValidData = true;
      }
    } else if (newData && typeof newData === 'object') {
      if (newData.fusion_result && newData.fusion_result.prediction_boxes) {
        console.log('从 fusion_result.prediction_boxes 提取融合模态识别结果');

        detectionsToAdd = convertFusionBoxesToDetections(newData.fusion_result);
        hasValidData = detectionsToAdd.length > 0;

        if (hasValidData && detectionsToAdd[0]?.class) {
          targetClass.value = detectionsToAdd[0].class;
        }
      } else if (newData.full) {
        const fullData = newData.full;

        if (fullData.labels) {
          detectionsToAdd = convertLabelsToDetections(fullData.labels);
          hasValidData = detectionsToAdd.length > 0;
        }
      } else if ('labels' in newData) {
        detectionsToAdd = convertLabelsToDetections(newData.labels);
        hasValidData = detectionsToAdd.length > 0;
      } else if (newData.detections && Array.isArray(newData.detections)) {
        if (newData.detections.length > 0) {
          detectionsToAdd = newData.detections.map((det: any) => ({
            x: det.x || 0,
            y: det.y || 0,
            width: det.width || 0,
            height: det.height || 0,
            type: det.type || 'pred',
            class: det.class || 'unknown',
            confidence: det.confidence || 0,
            pixelBox: det.pixelBox || det.box_xyxy_pixel || undefined,
            xyz: det.xyz || det.xyz_m || det.lidar3DInfo?.center || null,
            positionText: det.positionText || det.position_text || '',
            matchedWithMmwave: Boolean(det.matchedWithMmwave || det.matched_with_mmwave),
            lidar3DInfo: det.lidar3DInfo || {
              center: det.xyz || det.xyz_m || [0, 0, 0],
              size: [0, 0, 0],
              yaw: 0
            }
          }));
          hasValidData = true;
        }
      }
    }

    if (hasValidData) {
      const detectionsWithTimestamp = detectionsToAdd.map(d => ({
        ...d,
        timestamp: d.timestamp || ts
      }));

      detections.value = detectionsWithTimestamp;

      const totalConfidence = detectionsToAdd.reduce(
        (sum, d) => sum + (d.confidence || 0),
        0
      );

      dataSummary.value = {
        totalDetections: detectionsToAdd.length,
        averageConfidence:
          detectionsToAdd.length > 0
            ? totalConfidence / detectionsToAdd.length
            : 0,
        processingTime: summary?.processingTime || 0,
        timestamp: ts
      };

      if (summary) {
        dataSummary.value = {
          ...dataSummary.value,
          ...summary
        };
      }

      console.log('已更新融合模态识别结果，检测到', detectionsToAdd.length, '个目标');
    } else {
      console.log('未获得有效检测数据，保持旧的目标定位信息');
    }
  } catch (error) {
    console.error('处理目标定位数据失败:', error);
  } finally {
    setTimeout(() => {
      isLoading.value = false;
    }, 500);
  }
};

const addDetection = (detection: Detection) => {
  detections.value.push(detection);
  isLoading.value = false;
};

const clearData = () => {
  detections.value = [];
  selectedRowData.value = null;
  targetClass.value = '目标';
  dataSummary.value = {
    totalDetections: 0,
    averageConfidence: 0,
    processingTime: 0,
    timestamp: Date.now()
  };
};

const resetData = () => {
  clearData();

  chartData.value.precision = defaultPrecisionData();
  chartData.value.recall = defaultRecallData();

  isLoading.value = false;

  console.log('目标定位模块已重置，等待下一份数据');
};

onMounted(() => {
  console.log('DataDisplay 已初始化');
});

defineExpose({
  updateData,
  addDetection,
  resetData,
  clearData,
  detections,
  updateMetrics
});
</script>

<style scoped>
.data-display {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.data-header {
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

.header-actions {
  display: flex;
  gap: 2px;
}

.data-content {
  flex: 1;
  padding: 0.1%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.content-three-columns {
  display: flex;
  gap: 8px;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.column-left,
.column-middle,
.column-right {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
  box-sizing: border-box;
}

.column-left {
  padding: 8px 12px;
  overflow-y: auto;
}

.column-middle,
.column-right {
  background: #f9f9f9;
  border-radius: 8px;
  padding: 8px;
}

.chart-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.chart-title {
  font-size: 13px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
  padding-bottom: 2px;
  border-bottom: 1px solid #e0e0e0;
  text-align: center;
}

.bar-chart.horizontal {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  justify-content: flex-start;
  overflow-y: auto;
  padding-top: 4px;
}

.bar-item.horizontal {
  display: flex;
  align-items: center;
  gap: 5px;
  min-height: 20px;
  flex-shrink: 0;
}

.bar-label.horizontal {
  width: 95px;
  font-size: 10px;
  font-weight: 500;
  color: #333;
  text-align: right;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bar-wrapper.horizontal {
  flex: 1;
  height: 19px;
  background: #e8e8e8;
  border-radius: 8px;
  overflow: hidden;
}

.bar.horizontal {
  height: 100%;
  border-radius: 8px;
  transition: width 0.3s ease-in-out;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 5px;
  min-width: 24px;
}

.bar-value-on-bar {
  font-size: 9px;
  font-weight: 600;
  color: white;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  white-space: nowrap;
}

.bar-full {
  background: linear-gradient(90deg, #ff6b6b, #ee5a24);
}

.bar-camera {
  background: linear-gradient(90deg, #74b9ff, #0984e3);
}

.bar-infrared {
  background: linear-gradient(90deg, #a29bfe, #6c5ce7);
}

.bar-micro {
  background: linear-gradient(90deg, #55efc4, #00b894);
}

.bar-radar {
  background: linear-gradient(90deg, #fdcb6e, #f39c12);
}

.llm-summary {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 1px 0;
}

.summary-paragraph {
  width: 100%;
  margin-bottom: 4px;
  padding: 6px 10px;
  background: #f0f9ff;
  border-radius: 6px;
  border-left: 2px solid #1890FF;
  box-sizing: border-box;
}

.summary-text {
  font-size: 13px;
  color: #333;
  line-height: 1.4;
}

.summary-text strong {
  color: #1890FF;
  font-weight: 700;
}

.highlight {
  color: #1890FF;
  font-weight: 700;
  font-size: 13px;
}

.instance-section {
  width: 100%;
  margin-bottom: 1px;
}

.instance-text-list {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 1px;
  margin-left: 1px;
}

.instance-text-item {
  display: flex;
  align-items: flex-start;
  gap: 3px;
  line-height: 1.3;
}

.instance-number {
  font-size: 13px;
  font-weight: 600;
  color: #333;
  flex-shrink: 0;
  min-width: 20px;
}

.instance-name {
  font-size: 13px;
  font-weight: 600;
  color: #1890FF;
  flex-shrink: 0;
}

.instance-detail {
  font-size: 13px;
  color: #666;
  flex: 1;
  line-height: 1.3;
}

.detail-value {
  font-size: 12px;
  color: #333;
  font-family: 'Courier New', Courier, monospace;
}

.empty-state {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px 0;
  color: #909399;
  font-size: 14px;
  background: #f9f9f9;
  border-radius: 8px;
  box-sizing: border-box;
}

.loading-state {
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.loading-text {
  text-align: center;
  color: #1890FF;
  font-size: 14px;
}

@media (max-width: 768px) {
  .content-three-columns {
    flex-direction: column;
  }

  .bar-label.horizontal {
    width: 80px;
  }
}
</style>