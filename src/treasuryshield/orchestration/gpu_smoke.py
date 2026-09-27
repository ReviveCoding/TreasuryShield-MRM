"""Real CUDA operations and representative CPU/GPU timing."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = root / "artifacts" / "gpu_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    info: dict[str, object] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        output.write_text(json.dumps(info, indent=2), encoding="utf-8")
        raise RuntimeError("PyTorch CUDA is unavailable; GPU-designated models cannot proceed")
    info["device_name"] = torch.cuda.get_device_name(0)
    torch.manual_seed(20260927)
    n = 2048
    a_cpu = torch.randn(n, n)
    b_cpu = torch.randn(n, n)
    t0 = time.perf_counter()
    cpu_result = a_cpu @ b_cpu
    info["cpu_matmul_seconds"] = time.perf_counter() - t0
    a_gpu = a_cpu.to("cuda", non_blocking=False)
    b_gpu = b_cpu.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_result = a_gpu @ b_gpu
    torch.cuda.synchronize()
    info["gpu_matmul_seconds"] = time.perf_counter() - t0
    info["max_abs_difference"] = float((cpu_result - gpu_result.cpu()).abs().max())
    info["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
