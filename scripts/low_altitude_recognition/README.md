# Low Altitude Recognition

这个目录用于低空场景的传统目标识别，和 `scripts/radar_bin_analysis/` 分离。

当前目标类别：
- Bridge
- Building complex
- Chimney
- Power line
- Power tower
- Signal tower
- Tall building
- Wind turbine

当前实现是一条 `雷达 + 图像` 的传统识别流水线：
1. `build_candidates.py`
   直接从对齐好的 `RAE` 雷达张量中做阈值、平滑、连通域，生成雷达候选。
   同时基于候选的方位角/距离，在对应图像上生成启发式 ROI。
2. `extract_features.py`
   为每个候选提取手工特征：
   - 雷达几何特征
   - 雷达强度/剖面/跨俯仰层能量特征
   - 图像灰度/梯度/边缘/对称性/缩略图特征
3. `train_traditional_classifier.py`
   使用 `RandomForest` 训练 8 类传统分类器。
4. `infer_traditional_classifier.py`
   读取特征表和模型，输出类别预测与每类概率。

## 运行顺序

### 1. 生成候选
```bash
python scripts/low_altitude_recognition/build_candidates.py   --cap-dir /path/to/capture
```

### 2. 提取特征
```bash
python scripts/low_altitude_recognition/extract_features.py   --candidate-dir /path/to/capture/recognition_candidates
```

### 3. 给 features.csv 添加 `label` 列
手工标成以下 8 类之一：
- Bridge
- Building complex
- Chimney
- Power line
- Power tower
- Signal tower
- Tall building
- Wind turbine

### 4. 训练模型
```bash
python scripts/low_altitude_recognition/train_traditional_classifier.py   --feature-csv /path/to/capture/recognition_candidates/features.csv   --model-out /path/to/models/low_altitude_rf.joblib
```

### 5. 推理
```bash
python scripts/low_altitude_recognition/infer_traditional_classifier.py   --feature-csv /path/to/capture/recognition_candidates/features.csv   --model /path/to/models/low_altitude_rf.joblib   --out-csv /path/to/capture/recognition_candidates/predictions.csv
```

## 说明
- 这是一版传统方法基线，不依赖深度学习，也不依赖 OpenCV。
- 图像 ROI 当前是基于雷达方位角和距离生成的启发式窗口，不是严格几何投影。
- 如果后续接入准确的雷达到相机投影标定，识别效果会明显更好。


### 6. 无监督规则识别 + 图像打框
```bash
python scripts/low_altitude_recognition/rule_based_recognition.py \
  --feature-csv /path/to/capture/recognition_candidates/features.csv
```

输出：
- `predictions.csv`
- `overlays/` 目录下带类别框和三维位置文字的可见光图像

说明：
- 这是纯规则启发式 8 类识别，不需要打标签。
- 三维位置是基于 `range + azimuth + pitch` 估算的雷达传感器坐标系位置。
