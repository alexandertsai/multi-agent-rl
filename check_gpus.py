"""Fail at setup, not mid-serve, if this CUDA 12.9 build cannot run on the driver.

Runs a cuBLAS matmul and a freshly compiled Triton kernel on every visible GPU.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def double(x, y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + i, tl.load(x + i, mask=i < n) * 2, mask=i < n)


def main():
    count = torch.cuda.device_count()
    if not count:
        print('[agent-rl] No GPUs visible; skipped the CUDA kernel check.')
        return
    for gpu in range(count):
        # Triton launches on the current device, so select each GPU before launching.
        with torch.cuda.device(gpu):
            a = torch.randn(4096, 4096, device=f'cuda:{gpu}', dtype=torch.bfloat16)
            b = torch.empty_like(a)
            double[(a.numel() // 1024,)](a, b, a.numel(), BLOCK=1024)
            torch.cuda.synchronize()
        assert torch.equal(b, a * 2) and torch.isfinite(a @ a).all(), f'Kernel check failed on GPU {gpu}'
    import pynvml  # nvidia-ml-py, a vLLM dependency
    pynvml.nvmlInit()
    driver = pynvml.nvmlSystemGetCudaDriverVersion()
    print(f'[agent-rl] CUDA and Triton kernels ran on all {count} GPUs '
          f'(torch CUDA {torch.version.cuda}, driver CUDA {driver // 1000}.{driver % 1000 // 10}).')


if __name__ == '__main__':
    main()
