#!/usr/bin/env python3
"""Diagnose GPU device isolation in vllm-omni multi-stage setup.

Run inside distrobox vllm-dev with the vllm-omni venv activated:

    distrobox enter vllm-dev -- bash -c \
      "source /home/luke/.venvs/vllm-omni/bin/activate && \
       python tools/diagnose_device_isolation.py"

Tests:
1. CUDA_VISIBLE_DEVICES inheritance across spawn boundaries
2. NVML PID mapping (detect_pid_host)
3. Per-process memory attribution
4. Full stage device isolation simulation
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys


def _worker_check_device(device_str: str, result_queue: mp.Queue) -> None:
    """Worker that sets CUDA_VISIBLE_DEVICES and reports device info."""
    import torch

    results = {
        "pid": os.getpid(),
        "env_before_cuda": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }

    try:
        device_count = torch.cuda.device_count()
        results["device_count"] = device_count

        if device_count > 0:
            torch.cuda.set_device(0)
            results["current_device"] = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(0)
            results["device_name"] = props.name
            results["device_total_mem_gb"] = round(props.total_memory / 1e9, 2)

            # Allocate 4MB tensor to ensure NVML can detect the process
            t = torch.zeros(2**20, device="cuda:0")  # 4MB (1M float32)
            results["alloc_ok"] = True
            results["torch_mem_mb"] = round(torch.cuda.memory_allocated(0) / 1e6, 1)
        else:
            results["error"] = "No CUDA devices visible"
    except Exception as e:
        results["error"] = str(e)

    # NVML check
    try:
        from vllm.third_party.pynvml import (
            nvmlDeviceGetComputeRunningProcesses,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        nvml_device_count = nvmlDeviceGetCount()
        results["nvml_device_count"] = nvml_device_count

        nvml_report = []
        my_pid = os.getpid()
        for i in range(nvml_device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            procs = nvmlDeviceGetComputeRunningProcesses(handle)
            my_mem = 0
            for p in procs:
                if p.pid == my_pid:
                    my_mem = p.usedGpuMemory or 0
            nvml_report.append({
                "gpu": i,
                "name": name,
                "my_pid_mem_mb": round(my_mem / 1e6, 1),
            })
        results["nvml_per_gpu"] = nvml_report
        nvmlShutdown()
    except Exception as e:
        results["nvml_error"] = str(e)

    result_queue.put(results)


def test_device_isolation():
    """Test 1: Spawn a worker with CUDA_VISIBLE_DEVICES=1, check it uses GPU 1."""
    import torch

    total_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"DEVICE ISOLATION DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Parent PID: {os.getpid()}")
    print(f"Parent CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"Parent torch.cuda.device_count(): {total_gpus}")
    print()

    if total_gpus < 2:
        print("SKIP: Need >= 2 GPUs for device isolation test")
        return

    for target_gpu in range(total_gpus):
        print(f"\n--- Test: CUDA_VISIBLE_DEVICES={target_gpu} ---")

        # Save and set env
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)

        q = mp.Queue()
        p = mp.Process(target=_worker_check_device, args=(str(target_gpu), q))
        p.start()
        p.join(timeout=30)

        # Restore env
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev

        if p.exitcode != 0:
            print(f"  Worker exited with code {p.exitcode}")
            continue

        try:
            results = q.get_nowait()
        except Exception:
            print("  No results from worker")
            continue

        print(f"  Worker PID: {results['pid']}")
        print(f"  Env CUDA_VISIBLE_DEVICES: {results['env_before_cuda']}")
        print(f"  torch device_count: {results.get('device_count', '?')}")
        print(f"  torch current_device: {results.get('current_device', '?')}")
        print(f"  Device name: {results.get('device_name', '?')}")
        print(f"  Device total mem: {results.get('device_total_mem_gb', '?')} GB")

        if "nvml_per_gpu" in results:
            print(f"  NVML device count: {results['nvml_device_count']}")
            for g in results["nvml_per_gpu"]:
                marker = " <-- THIS PROCESS" if g["my_pid_mem_mb"] > 0 else ""
                print(f"    GPU {g['gpu']} ({g['name']}): {g['my_pid_mem_mb']} MB{marker}")
        if "error" in results:
            print(f"  ERROR: {results['error']}")
        if "nvml_error" in results:
            print(f"  NVML ERROR: {results['nvml_error']}")

        # Verdict
        if results.get("device_count") == 1 and "nvml_per_gpu" in results:
            on_correct = any(
                g["gpu"] == target_gpu and g["my_pid_mem_mb"] > 0
                for g in results["nvml_per_gpu"]
            )
            on_wrong = any(
                g["gpu"] != target_gpu and g["my_pid_mem_mb"] > 0
                for g in results["nvml_per_gpu"]
            )
            if on_correct and not on_wrong:
                print(f"  ✅ PASS: Correctly isolated to GPU {target_gpu}")
            elif on_wrong:
                print(f"  ❌ FAIL: Memory on wrong GPU!")
            else:
                print(f"  ⚠ INCONCLUSIVE: NVML did not detect process memory")
        elif results.get("device_count") != 1:
            print(f"  ❌ FAIL: device_count={results.get('device_count')}, expected 1")


def test_detect_pid_host():
    """Test 2: Check detect_pid_host() logic."""
    print(f"\n{'='*60}")
    print(f"PID HOST DETECTION")
    print(f"{'='*60}")

    try:
        from vllm_omni.entrypoints.utils import detect_pid_host, has_pid_host, in_container
    except ImportError:
        print("Cannot import vllm_omni.entrypoints.utils — running basic checks")
        pid2 = None
        try:
            with open("/proc/2/comm") as f:
                pid2 = f.read().strip()
        except Exception:
            pass

        print(f"  /proc/2/comm: {pid2}")
        print(f"  /.dockerenv exists: {os.path.exists('/.dockerenv')}")

        cg = ""
        try:
            with open("/proc/1/cgroup") as f:
                cg = f.read()
        except Exception:
            pass
        markers = ("docker", "containerd", "kubepods", "libpod", "podman")
        print(f"  Container markers in cgroup: {any(m in cg for m in markers)}")
        return

    ic = in_container()
    hph = has_pid_host()
    dph = detect_pid_host()
    print(f"  in_container(): {ic}")
    print(f"  has_pid_host(): {hph}")
    print(f"  detect_pid_host(): {dph}")

    if not dph:
        print("\n  WARNING: detect_pid_host() is False!")
        print("  NVML process memory won't match → profiling fallback used")
        print("  This is the root cause of upstream #1387 OOM in containers")


def _engine_core_sim(target_gpu: str, result_queue: mp.Queue):
    """Simulates EngineCore: spawns a worker subprocess."""
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    results = {"engine_pid": os.getpid(), "engine_env": env_val}

    # EngineCore spawns worker (like MultiprocExecutor does)
    q2 = mp.Queue()
    w = mp.Process(target=_worker_check_device, args=(target_gpu, q2))
    w.start()
    w.join(timeout=30)

    try:
        worker_results = q2.get_nowait()
        results["worker"] = worker_results
    except Exception:
        results["worker_error"] = "No results from nested worker"

    result_queue.put(results)


def test_nested_spawn():
    """Test 3: Simulate vllm's process tree (parent → EngineCore → Worker)."""
    print(f"\n{'='*60}")
    print(f"NESTED SPAWN (EngineCore → Worker simulation)")
    print(f"{'='*60}")

    import torch
    if torch.cuda.device_count() < 2:
        print("SKIP: Need >= 2 GPUs")
        return

    target = "1"
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = target

    q = mp.Queue()
    p = mp.Process(target=_engine_core_sim, args=(target, q))
    p.start()
    p.join(timeout=60)

    if prev is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev

    if p.exitcode != 0:
        print(f"  EngineCore exited with code {p.exitcode}")
        return

    try:
        results = q.get_nowait()
    except Exception:
        print("  No results from EngineCore")
        return

    print(f"  EngineCore PID: {results['engine_pid']}")
    print(f"  EngineCore CUDA_VISIBLE_DEVICES: {results['engine_env']}")

    w = results.get("worker", {})
    if w:
        print(f"  Worker PID: {w.get('pid')}")
        print(f"  Worker CUDA_VISIBLE_DEVICES: {w.get('env_before_cuda')}")
        print(f"  Worker device_count: {w.get('device_count')}")
        print(f"  Worker device name: {w.get('device_name')}")
        if "nvml_per_gpu" in w:
            for g in w["nvml_per_gpu"]:
                marker = " <-- WORKER" if g["my_pid_mem_mb"] > 0 else ""
                print(f"    GPU {g['gpu']} ({g['name']}): {g['my_pid_mem_mb']} MB{marker}")

        # VERDICT
        if w.get("device_count") == 1 and w.get("env_before_cuda") == target:
            nvml = w.get("nvml_per_gpu", [])
            worker_on_correct = any(
                g["gpu"] == int(target) and g["my_pid_mem_mb"] > 0 for g in nvml
            )
            worker_on_wrong = any(
                g["gpu"] != int(target) and g["my_pid_mem_mb"] > 0 for g in nvml
            )
            if worker_on_correct and not worker_on_wrong:
                print("\n  ✅ PASS: Nested spawn correctly isolated to GPU", target)
            elif worker_on_wrong:
                print(f"\n  ❌ FAIL: Worker memory on WRONG GPU!")
                print("  This confirms upstream device isolation bug")
            else:
                print("\n  ⚠ INCONCLUSIVE: No NVML memory detected (allocation too small?)")
        else:
            print("\n  ⚠ Unexpected state")
    else:
        print(f"  Worker error: {results.get('worker_error')}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    test_detect_pid_host()
    test_device_isolation()
    test_nested_spawn()
    print(f"\n{'='*60}")
    print("Diagnostic complete.")
