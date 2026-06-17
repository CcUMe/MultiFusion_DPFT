<template>
  <div class="image-gallery">
    <div class="gallery-container">
      <div
        v-for="(image, index) in images"
        :key="index"
        class="image-card"
        :class="{
          'camera-card': image.type === 'camera',
          'radar-card': image.type === 'radar',
          'lidar-card': image.type === 'lidar'
        }"
      >
        <div class="image-wrapper">
          <!-- 加载中状态 -->
          <div v-if="image.loading" class="image-placeholder loading-state">
            <el-skeleton :rows="1" animated />
            <div class="loading-spinner"></div>
            <div class="loading-text">{{ image.label }} 加载中...</div>
          </div>

          <!-- 错误状态 -->
          <div v-else-if="image.error" class="image-placeholder error-state">
            <div class="error-icon">⚠️</div>
            <span class="placeholder-text error-text">图片加载失败</span>
            <span class="error-hint">请检查网络连接或刷新重试</span>
          </div>

          <!-- 图片加载完成 -->
          <template v-else-if="image.src">
            <el-image
              :src="image.src"
              :alt="image.label"
              class="gallery-image"
              fit="contain"
              preview-teleported
              :preview-src-list="[image.src]"
              :initial-index="0"
              @load="onImageLoad(index)"
              @error="onImageError(index)"
            />
            <div v-if="image.loadTime" class="load-time">
              加载耗时: {{ image.loadTime.toFixed(2) }}ms
            </div>
          </template>

          <!-- 初始状态 -->
          <div v-else class="image-placeholder">
            <span class="placeholder-text">等待图片数据...</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import { ElImage } from 'element-plus';

// 定义自定义事件
const emit = defineEmits(['image-updated']);

interface ImageData {
  label: string;
  type: string;
  src?: string;
  loading?: boolean;
  error?: boolean;
  loadTime?: number;
  loadingStartTime?: number;
}

interface OriginalImages {
  camera?: string;
  radar?: string;
  lidar?: string;
  other?: string; // 红外
  micro?: string; // 微光夜视
}

interface ImageMap {
  originalImages?: OriginalImages;
}

const images = ref<ImageData[]>([]);

// 图片缓存
const imageCache = ref<Map<string, { data: string; timestamp: number }>>(new Map());

// 缓存配置
const CACHE_DURATION = 5 * 60 * 1000; // 5分钟

const onImageLoad = (index: number, event?: Event) => {
  const endTime = performance.now();
  const image = images.value[index];

  if (image.loadingStartTime) {
    image.loadTime = endTime - image.loadingStartTime;
    console.log(`${image.label} 加载完成，耗时: ${image.loadTime.toFixed(2)}ms`);
  }

  image.loading = false;
  image.error = false;

  // 如果是相机图像，通知父组件图像已更新
  if (image.type === 'camera') {
    emit('image-updated', image.src);
  }
};

const onImageError = (index: number) => {
  console.error(`${images.value[index].label} 加载失败`);

  const image = images.value[index];
  image.loading = false;
  image.error = true;
  image.src = '';
};

// 根据图片类型选择合适的文件格式
const getImageFormatByType = (type: string): string => {
  switch (type) {
    case 'camera':
      return 'jpeg';

    case 'other': // 红外原始图一般也是 jpg/jpeg
      return 'jpeg';

    case 'micro': // 微光夜视原始图一般也是 jpg/jpeg
      return 'jpeg';

    case 'radar':
      return 'png';

    case 'lidar':
    case 'lidar1':
    case 'lidar2':
      return 'png';

    default:
      return 'png';
  }
};

// 预加载图片
const preloadImage = (imageData: string, type: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    let imageSrc = imageData;

    const isUrlPath =
      imageData.startsWith('http://') ||
      imageData.startsWith('https://') ||
      imageData.startsWith('/') ||
      imageData.includes('\\');

    if (imageData && !imageData.startsWith('data:') && !isUrlPath) {
      const format = getImageFormatByType(type);
      imageSrc = `data:image/${format};base64,${imageData}`;
    }

    const img = new Image();

    img.onload = () => resolve(imageSrc);
    img.onerror = () => reject(new Error(`Failed to preload ${type} image`));

    img.src = imageSrc;
  });
};

const loadImage = async (index: number, imageData: string): Promise<void> => {
  const image = images.value[index];

  if (!imageData || !image) {
    console.warn(`图片数据无效或索引${index}不存在`);
    return Promise.reject(new Error('Invalid image data or index'));
  }

  const oldSrc = image.src;

  image.loading = true;
  image.error = false;
  image.loadingStartTime = performance.now();

  try {
    const imageSrc = await preloadImage(imageData, image.type);

    image.src = imageSrc;
    image.loading = false;
    image.error = false;

    console.log(`${image.label} 实时加载完成`);
    return Promise.resolve();
  } catch (error) {
    console.error(`${image.label} 加载失败:`, error);

    image.error = true;
    image.loading = false;

    if (oldSrc) {
      image.src = oldSrc;
      image.error = false;
      console.log(`${image.label} 恢复旧图片`);
    }

    return Promise.reject(error);
  }
};

// 加载图片到临时对象
const loadImageToTemp = async (image: ImageData, imageData: string) => {
  if (!imageData) return;

  const cacheKey = `${image.type}_${imageData.substring(0, 100)}`;
  const cached = imageCache.value.get(cacheKey);

  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    image.src = cached.data;
    image.loading = false;
    image.error = false;
    console.log(`${image.label} 使用缓存加载`);
    return;
  }

  const oldSrc = image.src;

  image.loading = true;
  image.error = false;
  image.loadingStartTime = performance.now();

  try {
    const imageSrc = await preloadImage(imageData, image.type);

    imageCache.value.set(cacheKey, {
      data: imageSrc,
      timestamp: Date.now()
    });

    image.src = imageSrc;
  } catch (error) {
    console.error(`${image.label} 加载失败:`, error);

    image.error = true;

    if (oldSrc) {
      image.src = oldSrc;
      image.error = false;
    }
  } finally {
    image.loading = false;
  }
};

const updateImages = async (imageMap: ImageMap) => {
  if (imageMap.originalImages && typeof imageMap.originalImages === 'object') {
    /*
      图片展示顺序：
      1. camera：相机
      2. radar：雷达 / 毫米波雷达
      3. lidar：激光雷达
      4. other：红外
      5. micro：微光夜视

      这里已经把红外 other 和微光夜视 micro 的展示位置互换。
    */
    const imageTypes = [
      { key: 'camera', label: '相机' },
      { key: 'radar', label: '雷达' },
      { key: 'lidar', label: '激光雷达' },
      { key: 'other', label: '红外' },
      { key: 'micro', label: '微光夜视' }
    ];

    // 确保 images 数组已初始化
    if (images.value.length === 0) {
      for (const { key, label } of imageTypes) {
        images.value.push({
          label: `${label}原始图`,
          type: key,
          src: '',
          loading: false,
          error: false
        });
      }
    }

    // 如果已经初始化过，也同步更新 label 和 type，避免旧顺序残留
    for (let i = 0; i < imageTypes.length; i++) {
      const { key, label } = imageTypes[i];

      if (!images.value[i]) {
        images.value[i] = {
          label: `${label}原始图`,
          type: key,
          src: '',
          loading: false,
          error: false
        };
      } else {
        images.value[i].label = `${label}原始图`;
        images.value[i].type = key;
      }
    }

    // 按新的顺序加载图片
    for (let i = 0; i < imageTypes.length; i++) {
      const { key, label } = imageTypes[i];
      const imageData = imageMap.originalImages[key as keyof OriginalImages];

      if (imageData) {
        loadImage(i, imageData)
          .then(() => {
            console.log(`${label}原始图加载完成`);
          })
          .catch((error) => {
            console.error(`${label}原始图加载失败:`, error);
          });
      } else {
        // 没有对应图片时，清空当前位置，防止上一轮旧图残留
        const image = images.value[i];

        if (image) {
          image.src = '';
          image.loading = false;
          image.error = false;
          image.loadTime = undefined;
          image.loadingStartTime = undefined;
        }

        console.log(`${label}原始图暂无数据`);
      }
    }
  }
};

onMounted(() => {
  console.log('ImageGallery 已初始化');
});

defineExpose({
  updateImages,
  images
});
</script>

<style scoped>
.image-gallery {
  width: 99%;
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 0.5%;
  background: #E6F7FF;
  border-radius: 8px;
  overflow-y: auto;
}

.gallery-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(2, 1fr);
  gap: 1%;
  width: 100%;
  height: 100%;
}

/* 相机卡片在第一行第一列 */
.camera-card {
  grid-row: 1 / 2;
  grid-column: 1 / 2;
}

/* 雷达卡片在第一行第二列 */
.radar-card {
  grid-row: 1 / 2;
  grid-column: 2 / 3;
}

/* 激光雷达卡片按顺序排布 */
.lidar-card {
  grid-row: 1 / 2;
  grid-column: 3 / 4;
}

.image-card {
  display: flex;
  flex-direction: column;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  height: 100%;
}

.image-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 3px 16px 0 rgba(0, 0, 0, 0.12);
}

.image-wrapper {
  flex: 1;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  overflow: auto;
  background: #f0f2f5;
  min-height: unset;
  padding: 8px;
  position: relative;
}

.gallery-image {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: 100%;
  cursor: pointer;
  background: #fafafa;
}

.image-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: #909399;
  padding: 20px;
}

/* 加载状态样式 */
.loading-state {
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #e0e0e0;
  border-top: 3px solid #1890FF;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 12px;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

.loading-text {
  font-size: 14px;
  color: #1890FF;
  margin-top: 10px;
  font-weight: 500;
}

/* 错误状态样式 */
.error-state {
  background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%);
  border: 1px dashed #ff4d4f;
  border-radius: 8px;
}

.error-icon {
  font-size: 32px;
  margin-bottom: 8px;
}

.error-text {
  color: #ff4d4f;
  font-weight: 500;
  margin-bottom: 4px;
}

.error-hint {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

.placeholder-text {
  font-size: 14px;
}

.load-time {
  position: absolute;
  bottom: 8px;
  right: 8px;
  font-size: 12px;
  color: rgba(0, 0, 0, 0.6);
  background: rgba(255, 255, 255, 0.7);
  padding: 2px 6px;
  border-radius: 3px;
  pointer-events: none;
}

.image-label {
  padding: 8px 12px;
  font-size: 14px;
  color: #303133;
  background: #f8f9fa;
  text-align: center;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .gallery-container {
    grid-template-columns: repeat(2, 1fr);
  }

  .camera-card,
  .radar-card,
  .lidar-card {
    grid-row: auto;
    grid-column: auto;
  }
}

@media (max-width: 768px) {
  .gallery-container {
    grid-template-columns: 1fr;
  }

  .camera-card,
  .radar-card,
  .lidar-card {
    grid-row: auto;
    grid-column: auto;
  }
}
</style>