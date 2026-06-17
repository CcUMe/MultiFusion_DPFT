<template>
  <div class="detection-viewer">
    <div class="viewer-container">
      <div class="top-section">
        <div
          v-for="(image, index) in topImages"
          :key="'top-' + index"
          class="top-image-card"
        >
          <div class="image-wrapper">
            <div v-if="image.loading" class="image-placeholder">
              <el-skeleton :rows="1" animated />
              <div class="loading-text">加载中...</div>
            </div>

            <div v-else-if="image.error" class="image-placeholder">
              <span class="placeholder-text">图片加载失败</span>
            </div>

            <template v-else-if="image.src">
              <el-image
                :src="image.src"
                :alt="image.label"
                class="detection-image"
                fit="contain"
                preview-teleported
                :preview-src-list="[image.src]"
                @load="onImageLoad('top', index)"
                @error="onImageError('top', index)"
              />
            </template>

            <div v-else class="image-placeholder">
              <span class="placeholder-text">等待图片数据...</span>
            </div>
          </div>
        </div>
      </div>

      <div class="bottom-section">
        <div class="bottom-image-card">
          <div class="image-wrapper">
            <div v-if="bottomImage.loading" class="image-placeholder">
              <el-skeleton :rows="1" animated />
              <div class="loading-text">加载中...</div>
            </div>

            <div v-else-if="bottomImage.error" class="image-placeholder">
              <span class="placeholder-text">图片加载失败</span>
            </div>

            <template v-else-if="bottomImage.src">
              <el-image
                :src="bottomImage.src"
                :alt="bottomImage.label"
                class="detection-image"
                fit="contain"
                preview-teleported
                :preview-src-list="[bottomImage.src]"
                @load="onImageLoad('bottom', 0)"
                @error="onImageError('bottom', 0)"
              />
            </template>

            <div v-else class="image-placeholder">
              <span class="placeholder-text">等待图片数据...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref } from 'vue';

interface ImageData {
  label: string;
  src?: string;
  loading?: boolean;
  error?: boolean;
  loadTime?: number;
  loadingStartTime?: number;
}

interface BoxesImages {
  camera?: string;
  radar?: string;

  // 红外，兼容不同字段名
  infrared?: string;
  ir?: string;
  other?: string;

  // 微光夜视，兼容不同字段名
  micro?: string;
  micro_light?: string;
  mirco_light?: string;
  lidar?: string;
}

interface DetectionImageObject {
  boxesImages?: BoxesImages;
  fusionImage?: string;
}

const topImages = ref<ImageData[]>([
  { label: '可见光单模态', src: '', loading: false, error: false },
  { label: '毫米波雷达单模态', src: '', loading: false, error: false },
  { label: '红外单模态', src: '', loading: false, error: false },
  { label: '微光夜视单模态', src: '', loading: false, error: false }
]);

const bottomImage = ref<ImageData>({
  label: '融合模态检测结果图',
  src: '',
  loading: false,
  error: false
});

const topImageConfigs = [
  {
    label: '可见光单模态',
    keys: ['camera']
  },
  {
    label: '毫米波雷达单模态',
    keys: ['radar']
  },
  {
    label: '红外单模态',
    keys: ['infrared', 'ir', 'other']
  },
  {
    label: '微光夜视单模态',
    keys: ['micro', 'micro_light', 'mirco_light', 'lidar']
  }
];

const onImageLoad = (section: 'top' | 'bottom', index: number) => {
  const endTime = performance.now();

  if (section === 'top') {
    const image = topImages.value[index];

    if (image.loadingStartTime) {
      image.loadTime = endTime - image.loadingStartTime;
      console.log(`${image.label} 加载完成，耗时: ${image.loadTime?.toFixed(2)}ms`);
    }

    image.loading = false;
    image.error = false;
  } else {
    const image = bottomImage.value;

    if (image.loadingStartTime) {
      image.loadTime = endTime - image.loadingStartTime;
      console.log(`${image.label} 加载完成，耗时: ${image.loadTime?.toFixed(2)}ms`);
    }

    image.loading = false;
    image.error = false;
  }
};

const onImageError = (section: 'top' | 'bottom', index: number) => {
  console.error(
    `图片加载失败: ${
      section === 'top' ? topImages.value[index].label : bottomImage.value.label
    }`
  );

  if (section === 'top') {
    const image = topImages.value[index];
    image.loading = false;
    image.error = true;
    image.src = '';
  } else {
    const image = bottomImage.value;
    image.loading = false;
    image.error = true;
    image.src = '';
  }
};

const toImageSrc = (imageData: string): string => {
  let imageSrc = imageData;

  const isUrlPath =
    imageData.startsWith('http://') ||
    imageData.startsWith('https://') ||
    imageData.startsWith('/') ||
    imageData.includes('\\');

  if (imageData && !imageData.startsWith('data:') && !isUrlPath) {
    imageSrc = `data:image/png;base64,${imageData}`;
  }

  return imageSrc;
};

const loadImage = async (
  section: 'top' | 'bottom',
  index: number,
  imageData: string
) => {
  if (!imageData) return;

  const image = section === 'top' ? topImages.value[index] : bottomImage.value;

  if (!image) {
    console.warn(`图片位置不存在: section=${section}, index=${index}`);
    return;
  }

  const oldSrc = image.src;

  image.loading = true;
  image.error = false;
  image.loadingStartTime = performance.now();

  try {
    const imageSrc = toImageSrc(imageData);

    const img = new Image();

    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = imageSrc;
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

const clearTopImage = (index: number) => {
  const image = topImages.value[index];

  if (!image) return;

  image.src = '';
  image.loading = false;
  image.error = false;
  image.loadTime = undefined;
  image.loadingStartTime = undefined;
};

const clearBottomImage = () => {
  bottomImage.value.src = '';
  bottomImage.value.loading = false;
  bottomImage.value.error = false;
  bottomImage.value.loadTime = undefined;
  bottomImage.value.loadingStartTime = undefined;
};

const getBoxImageByKeys = (
  boxesImages: BoxesImages,
  keys: string[]
): string | undefined => {
  for (const key of keys) {
    const value = boxesImages[key as keyof BoxesImages];

    if (value) {
      return value;
    }
  }

  return undefined;
};

const updateTopLabels = () => {
  for (let i = 0; i < topImageConfigs.length; i++) {
    const config = topImageConfigs[i];

    if (!topImages.value[i]) {
      topImages.value[i] = {
        label: config.label,
        src: '',
        loading: false,
        error: false
      };
    } else {
      topImages.value[i].label = config.label;
    }
  }
};

const updateDetectionImages = async (
  imageData: DetectionImageObject | string[] | []
) => {
  console.log('updateDetectionImages called with:', imageData);

  updateTopLabels();

  /*
    数组格式：
    默认输入顺序按：
    [可见光, 毫米波, 红外, 微光, 融合]

    显示顺序也是：
    [可见光, 毫米波, 红外, 微光, 融合]
  */
  if (Array.isArray(imageData)) {
    console.log('Array format detected:', imageData);

    const topArraySourceIndexOrder = [0, 1, 2, 3];

    for (let displayIndex = 0; displayIndex < topArraySourceIndexOrder.length; displayIndex++) {
      const sourceIndex = topArraySourceIndexOrder[displayIndex];
      const imageSrc = imageData[sourceIndex];

      if (imageSrc) {
        console.log(
          `Loading top image displayIndex=${displayIndex}, sourceIndex=${sourceIndex}:`,
          imageSrc
        );

        loadImage('top', displayIndex, imageSrc);
      } else {
        console.warn(
          `Top image missing: displayIndex=${displayIndex}, sourceIndex=${sourceIndex}`
        );

        clearTopImage(displayIndex);
      }
    }

    if (imageData.length >= 5 && imageData[4]) {
      console.log('Loading bottom fusion image:', imageData[4]);
      loadImage('bottom', 0, imageData[4]);
    } else {
      console.warn('Bottom fusion image is missing');
      clearBottomImage();
    }

    return;
  }

  /*
    对象格式：
    显示顺序固定为：
    1. camera -> 可见光
    2. radar -> 毫米波
    3. infrared / ir / other -> 红外
    4. micro / micro_light / mirco_light / lidar -> 微光夜视
    5. fusionImage -> 融合
  */
  if (typeof imageData === 'object' && imageData !== null) {
    console.log('Object format detected:', imageData);

    if (imageData.boxesImages && typeof imageData.boxesImages === 'object') {
      const boxesImages = imageData.boxesImages;

      for (let displayIndex = 0; displayIndex < topImageConfigs.length; displayIndex++) {
        const config = topImageConfigs[displayIndex];
        const imageSrc = getBoxImageByKeys(boxesImages, config.keys);

        if (imageSrc) {
          console.log(`Loading ${config.label}:`, imageSrc);
          loadImage('top', displayIndex, imageSrc);
        } else {
          console.warn(`${config.label} image is missing`);
          clearTopImage(displayIndex);
        }
      }
    } else {
      console.warn('boxesImages is missing or invalid');

      for (let i = 0; i < topImages.value.length; i++) {
        clearTopImage(i);
      }
    }

    if (imageData.fusionImage) {
      console.log('Loading fusion image:', imageData.fusionImage);
      loadImage('bottom', 0, imageData.fusionImage);
    } else {
      console.warn('Fusion image is missing');
      clearBottomImage();
    }

    return;
  }

  console.error('Invalid imageData format:', imageData);
};

defineExpose({
  updateDetectionImages
});
</script>

<style scoped>
.detection-viewer {
  width: 100%;
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #E6F7FF;
  border-radius: 8px;
  overflow: hidden;
}

.viewer-container {
  width: 100%;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-height: 0;
}

.top-section {
  flex: 0 0 25%;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 2px;
  padding: 2px;
  overflow: auto;
}

.bottom-section {
  flex: 1;
  padding: 2px;
  overflow: auto;
}

.top-image-card {
  display: flex;
  flex-direction: column;
  background: white;
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1);
  height: 100%;
  max-height: 100%;
}

.bottom-image-card {
  display: flex;
  flex-direction: column;
  background: white;
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1);
  height: 100%;
}

.image-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.detection-image {
  object-fit: contain;
  max-width: 100%;
  max-height: 100%;
  width: 100%;
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
}

.placeholder-text {
  font-size: 12px;
}

.loading-text {
  font-size: 12px;
  color: #1890FF;
  margin-top: 8px;
}

.image-label {
  padding: 4px 8px;
  font-size: 12px;
  color: #303133;
  background: #f8f9fa;
  text-align: center;
  border-top: 1px solid #e8e8e8;
}
</style>