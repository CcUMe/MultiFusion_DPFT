`low_altitude_radar_config.jsonc` 参数说明

对应运行配置文件：
[low_altitude_radar_config.jsonc](/home/yangqilin/code/dpft_v4/scripts/radar_bin_analysis/low_altitude_radar_config.jsonc)

当前导出的雷达 `npy` 已调整为 3D RAE 张量：
- shape: `(range, azimuth, elevation_layer)`
- 含义: 同一个完整扫描帧里，保留所有俯仰层，不再把多层直接叠加成一张图
- 可视化时只会按配置选择其中一层来画

说明如下。

`pitch_selection`

- `strategy`：俯仰层选择策略。`dominant_count` 选样本数最多的一层，`fixed_value` 选固定俯仰角，`nearest_zero` 选最接近 0 度的一层。
- `fixed_pitch_deg`：当 `strategy = fixed_value` 时生效，表示希望抽取的目标俯仰角，单位是度。
- `quantization_step_deg`：俯仰角离散化步长，单位是度。用于把存在浮动的原始俯仰角归并到规则栅格。
- `selection_tolerance_deg`：俯仰层归属容差，单位是度。包的俯仰角与目标层差值不超过该值时，视为属于该层。
- `prefer_nearest_zero_on_tie`：当多个俯仰层样本数并列时，是否优先选更接近 0 度的那一层。

`ra_output`

- `mapping_csv_name`：图像与雷达完整扫描帧的对齐关系表文件名。
- `packet_csv_name`：包级别时间对齐结果文件名。
- `ra_dir_name`：导出的雷达 `.npy` 文件所在目录名。当前文件内容是 3D `RAE` 张量，而不是单层二维图。

`visualization`

- `preview_dir_name`：可视化输出总目录名。
- `radar_subdir_name`：单独保存雷达热力图 PNG 的子目录名。
- `compare_subdir_name`：保存“可见光 + 雷达并排图”的子目录名。
- `side_by_side_gap_px`：并排时左右两张图之间的像素间距。
- `side_by_side_margin_px`：画布四周的像素边距。
- `side_by_side_label_height_px`：顶部文字说明区域高度，单位像素。
- `layer_selection`：可视化时选哪一层俯仰切片。`selected_pitch` 用导出阶段选中的代表层，`middle` 用中间层，`fixed_index` 用固定层号，`fixed_pitch_deg` 按真实俯仰角匹配当前帧最接近的一层。
- `fixed_layer_index`：当 `layer_selection = fixed_index` 时生效，表示固定显示第几层，索引从 0 开始。
- `fixed_pitch_deg`：当 `layer_selection = fixed_pitch_deg` 时生效，表示希望显示的目标俯仰角，单位是度。脚本会自动在当前帧已有的 `pitch_layers_deg` 里寻找最接近的一层。
- `layer_fallback`：当指定层无效或越界时的回退策略，可选 `middle`、`first`、`last`。


生成对齐表和 mmwave_ra_npy：
``text
python ~/code/dpft_v4/scripts/radar_bin_analysis/build_image_aligned_ra.py \
  --cap-dir /mnt/disk1/yangqilin/dataset/LH_all_sensor/5_19/with_cameras_capture_20260519_143539
``

解析协议目标：
``text
python ~/code/dpft_v4/scripts/parse_mmwave_bin_with_read.py \
  /mnt/disk1/yangqilin/dataset/LH_all_sensor/5_19/with_cameras_capture_20260519_143539/with_cameras_capture_20260519_143539_mmwave_udp.bin
``

跑纯雷达识别：
``text
python ~/code/dpft_v4/scripts/low_altitude_recognition/run_rule_based_pipeline.py \
  --cap-dir /mnt/disk1/yangqilin/dataset/LH_all_sensor/5_19/with_cameras_capture_20260519_143539 \
  --candidate-source protocol
``