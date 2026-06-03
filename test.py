import sys

try:
    import torch
except ImportError:
    print("torch is not installed")
    sys.exit(1)


def show_basic_info(name, x):
    print(f"\n=== {name} ===")
    print("type:", type(x))
    print("shape:", getattr(x, "shape", None))
    print("dtype:", getattr(x, "dtype", None))
    print("device:", getattr(x, "device", None))
    print("requires_grad:", getattr(x, "requires_grad", None))
    print("value:")
    print(x)


def breakpoint_here():
    # 在这里打断点
    pass


def main():
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # 1. 普通 CPU tensor
    x_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    x_cpu_dbg = x_cpu.detach().cpu().numpy()

    # 2. requires_grad=True 的 tensor
    x_grad = torch.randn(3, 4, requires_grad=True)
    x_grad_dbg = x_grad.detach().cpu().numpy()

    # 3. 标量 tensor
    x_scalar = torch.tensor(123.456)
    x_scalar_dbg = x_scalar.detach().cpu().numpy()

    # 4. CUDA tensor（如果可用）
    x_cuda = None
    x_cuda_dbg = None
    if torch.cuda.is_available():
        x_cuda = torch.randn(3, 4, device="cuda")
        x_cuda_dbg = x_cuda.detach().cpu().numpy()

    # 打印到控制台，确认数据本身是正常的
    show_basic_info("x_cpu", x_cpu)
    show_basic_info("x_grad", x_grad)
    show_basic_info("x_scalar", x_scalar)
    if x_cuda is not None:
        show_basic_info("x_cuda", x_cuda)

    # 这里打断点，然后在 PyCharm 里观察这些变量：
    # x_cpu
    # x_cpu_dbg
    # x_grad
    # x_grad_dbg
    # x_scalar
    # x_scalar_dbg
    # x_cuda
    # x_cuda_dbg
    breakpoint_here()

    # 防止断点后程序太快退出
    print("\nFinished.")


if __name__ == "__main__":
    main()