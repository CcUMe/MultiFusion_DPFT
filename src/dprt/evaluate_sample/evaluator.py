"""简化的单样本评估器，默认只计算单次推理时间"""
import torch
from tqdm import tqdm
import time

class SimpleEvaluator:
    def __init__(self, device: str = 'cuda', warmup_iters: int = 10):
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters

    def __call__(self, model, test_loader):
        """推理 + 单次推理时间 + 输出结果"""
        model = model.to(self.device).eval()

        with torch.no_grad():
            data, labels = next(iter(test_loader))

            # 1. 预热（避免冷启动）
            print(f"[GPU/CPU预热 {self.warmup_iters}次]")
            data_warm = {k: v.to(self.device) for k, v in data.items()}
            for _ in range(self.warmup_iters):
                _ = model(data_warm)

            # 2. 查看输入输出结构（用预热后的data）
            print("\n[输入数据keys/shape]")
            for k, v in data_warm.items():
                print(f"  {k}: {v.shape}")

            sample_output = model(data_warm)
            print(f"\n[输出结构]")
            if isinstance(sample_output, dict):
                for k, v in sample_output.items():
                    print(f"  {k}: {v.shape}")
            else:
                print(f"  {sample_output.shape}")

            # 3. 单次真实推理时间测量
            print("\n" + "="*60)
            print("[单次推理时间测量]")
            print("="*60)

            if self.device.type == 'cuda':
                # GPU: CUDA Event（最高精度）
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                final_output = model(data_warm)
                end_event.record()
                torch.cuda.synchronize()

                single_time_ms = start_event.elapsed_time(end_event)
                print(f"单次推理时间: {single_time_ms:.3f} ms")
                print(f"单次FPS: {1000/single_time_ms:.1f}")

            else:
                # CPU: time.perf_counter
                start = time.perf_counter()
                final_output = model(data_warm)
                single_time_ms = (time.perf_counter() - start) * 1000
                print(f"单次推理时间: {single_time_ms:.3f} ms")
                print(f"单次FPS: {1000/single_time_ms:.1f}")

        print("="*60)
        print("推理完成！最终输出:")
        print(final_output)
        print("="*60)

        return {
            'output': final_output,
            'single_inference_time_ms': single_time_ms,
            'single_fps': 1000 / single_time_ms
        }

