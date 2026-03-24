"""Base worker class for vLLM-Omni with process-scoped GPU memory accounting."""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib, memory_profiling
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from vllm_omni.entrypoints.utils import detect_pid_host
from vllm_omni.worker.gpu_memory_utils import (
    get_process_gpu_memory,
    is_process_scoped_memory_available,
)

logger = init_logger(__name__)


class OmniGPUWorkerBase(GPUWorker):
    """Base GPU worker for vLLM-Omni with process-scoped memory accounting.

    This class overrides determine_available_memory() to use per-process GPU
    memory tracking via pynvml, allowing multiple stages to initialize
    concurrently on the same GPU without memory accounting interference.
    """

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Process-scoped GPU memory profiling for concurrent stage initialization.

        Algorithm:
            1. requested_memory = total_gpu_memory * gpu_memory_utilization
               (computed in init_device from cache_config)

            2. process_memory = memory used by THIS process only (via pynvml)
               - Uses nvmlDeviceGetComputeRunningProcesses to get per-PID memory
               - Supports CUDA_VISIBLE_DEVICES with indices, UUIDs, or MIG IDs

            3. available_kv_cache = requested_memory - process_memory

        Fallback:
            If NVML is unavailable, falls back to profiling data:
            available = requested - (weights + activations + non_torch)
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            logger.info(
                "Using explicit kv_cache_memory_bytes: %s GiB",
                format_gib(kv_cache_memory_bytes),
            )
            return kv_cache_memory_bytes

        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        process_memory = (
            get_process_gpu_memory(self.local_rank)
            if is_process_scoped_memory_available() and detect_pid_host()
            else None
        )

        if process_memory is not None:
            # NVML available: use per-process memory
            self.available_kv_cache_memory_bytes = max(0, self.requested_memory - process_memory)
            logger.debug(
                "Process-scoped memory (PID %d, GPU %d): requested=%s, used=%s, available=%s",
                os.getpid(),
                self.local_rank,
                format_gib(self.requested_memory),
                format_gib(process_memory),
                format_gib(self.available_kv_cache_memory_bytes),
            )
            logger.info_once(
                "Available KV cache memory: %s GiB (process-scoped)",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )
        else:
            # NVML unavailable: use steady-state torch memory after profiling.
            # torch_peak_increase inflates with transient torch.compile/AOT
            # compilation buffers that are freed after compilation completes.
            # after_profile.torch_memory = memory_reserved() after gc+empty_cache,
            # which reflects model weights + persistent buffers only (compilation
            # artifacts are freed, so memory_reserved ≈ memory_allocated at this point).
            steady_state_torch = profile_result.after_profile.torch_memory
            non_torch = max(0, profile_result.non_torch_increase)
            profiled_usage = steady_state_torch + non_torch
            self.available_kv_cache_memory_bytes = max(0, self.requested_memory - profiled_usage)
            logger.debug(
                "Profiling fallback (PID %d, GPU %d): requested=%s, "
                "steady_torch=%s, non_torch=%s, available=%s",
                os.getpid(),
                self.local_rank,
                format_gib(self.requested_memory),
                format_gib(steady_state_torch),
                format_gib(non_torch),
                format_gib(self.available_kv_cache_memory_bytes),
            )
            logger.info_once(
                "Available KV cache memory: %s GiB (profiling fallback)",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )

        return int(self.available_kv_cache_memory_bytes)
