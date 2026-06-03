"""简化的单样本评估器，默认只计算单次推理时间"""
import torch
from tqdm import tqdm
import time
from dprt.training.assigner import HungarianAnassigner

class SimpleEvaluator:
    def __init__(self, device: str = 'cuda', warmup_iters: int = 10):
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self.matcher = HungarianAnassigner(
            loss_weights={'class': 1.0, 'center': 1.0, 'size': 1.0, 'angle': 1.0, 'giou': 1.0})  # 添加匈牙利匹配器 [file:1]

    def __call__(self, model, test_loader):
        """推理 + 单次推理时间 + 输出结果"""
        model = model.to(self.device).eval()

        with torch.no_grad():
            data, labels = next(iter(test_loader))

            print(f"原始labels类型: {type(labels)}")
            print(f"labels内容: {labels}")
            if isinstance(labels, dict):
                for k, v in labels.items():
                    print(f"{k}: {type(v)}, shape={getattr(v, 'shape', 'N/A')}")

            # 1. 预热（避免冷启动）
            print(f"[GPU/CPU预热 {self.warmup_iters}次]")
            data_warm = {k: v.float().to(self.device) for k, v in data.items()}  # 加 .float()
            for _ in range(self.warmup_iters):
                model(data_warm)

            # 2. 查看输入输出结构（用预热后的data）
            print("\n[输入数据keys/shape]")
            for k, v in data_warm.items():
                print(f"  {k}: {v.shape}")

            output = model(data_warm)

            # 安全处理labels格式
            # 安全处理labels格式
            if isinstance(labels, dict):
                labels_device = {}
                # 添加key映射
                key_mapping = {
                    'class': 'gt_class',
                    'center': 'gt_center',
                    'size': 'gt_size',
                    'angle': 'gt_angle'
                }
                for k, v in labels.items():
                    new_key = key_mapping.get(k, k)  # 若无映射则保持原key
                    labels_device[new_key] = v.float().to(self.device)
            elif isinstance(labels, list) and len(labels) > 0:
                if isinstance(labels[0], dict):
                    labels_device = {k: item[k].float().to(self.device)
                                     for item in labels for k in item}
                else:
                    labels_device = {i: label.float().to(self.device)
                                     for i, label in enumerate(labels)}
            else:
                labels_device = {'dummy': torch.zeros(1, 10, 80).to(self.device)}  # 哑数据

            print(f"labels类型: {type(labels)}, labels_device键: {list(labels_device.keys())}")

            # 安全匹配，**保留所有400预测**
            print(f"原始预测: class.shape={output['class'].shape}")  # 确认[1,400,C]


            # ===== 新增：详细Matcher输入检查 =====
            print("=== Matcher输入检查 ===")
            print(f"output keys: {list(output.keys())}")
            print(f"output['class'].shape: {output['class'].shape}")
            print(f"output['center'].shape: {output['center'].shape}")
            print(f"labels_device keys: {list(labels_device.keys())}")
            print(f"labels_device['gt_class'].shape: {labels_device['gt_class'].shape}")
            print(f"labels_device['gt_center'].shape: {labels_device['gt_center'].shape}")
            print(
                f"设备一致: output.device={output['class'].device}, labels.device={list(labels_device.values())[0].device}")

            try:
                indices = self.matcher(output, labels_device)
                print(
                    f"✅ 匹配成功: indices[0].shape {indices[0].shape if isinstance(indices, (list, tuple)) else 'N/A'}")
            except Exception as e:
                print(f"❌ Matcher失败: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()  # 显示完整调用栈
                print(f"fallback: B={output['class'].shape[0]}, N={output['class'].shape[1]}")
                indices_i = torch.arange(output['class'].shape[1]).unsqueeze(0).expand(output['class'].shape[0],
                                                                                      -1).to(self.device)
                indices_j = torch.zeros_like(indices_i)
                indices = indices_i, indices_j  # 确保indices是tuple

            # 提取匹配结果（现在是[1, min(M,400), ...] 或 [1,400,...]）
            matched_predictions = {}
            for key in output:
                pred = output[key]
                idx = indices_i.unsqueeze(-1).expand(-1, -1, *pred.shape[2:])
                matched_predictions[key] = pred.gather(1, idx.long())

            print(f"[匹配完成: {matched_predictions['class'].shape}]")
            print(f"\n[输出结构]")
            if isinstance(matched_predictions, dict):
                for k, v in matched_predictions.items():
                    print(f"  {k}: {v.shape}")
            else:
                print(f"  {matched_predictions.shape}")

            # 3. 单次真实推理时间测量
            print("\n" + "="*60)
            print("[单次推理时间测量]")
            print("="*60)

            if self.device.type == 'cuda':
                # GPU: CUDA Event（最高精度）
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _=model(data_warm)
                end_event.record()
                torch.cuda.synchronize()

                single_time_ms = start_event.elapsed_time(end_event)
                print(f"单次推理时间: {single_time_ms:.3f} ms")
                print(f"单次FPS: {1000/single_time_ms:.1f}")

            else:
                # CPU: time.perf_counter
                start = time.perf_counter()
                _ = model(data_warm)  # 纯模型推理
                matched_predictions = model(data_warm)
                single_time_ms = (time.perf_counter() - start) * 1000
                print(f"单次推理时间: {single_time_ms:.3f} ms")
                print(f"单次FPS: {1000/single_time_ms:.1f}")

        # print("="*60)
        # print("推理完成！最终输出:")
        # print(final_output)
        # print("="*60)

        return {
            'output': matched_predictions,
            'single_inference_time_ms': single_time_ms,
            'single_fps': 1000 / single_time_ms
        }

