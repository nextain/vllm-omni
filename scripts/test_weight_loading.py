#!/usr/bin/env python3
"""
Minimal weight loading test — no vllm server, no CUDA graph capture.
Runs in ~2-3 minutes on CPU. Discovers all WeightsMapper issues at once.

Usage:
  /workspace/venv/bin/python3 /workspace/scripts/test_weight_loading.py \
      /workspace/.cache_hf/models--openbmb--MiniCPM-o-4_5/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc

Exit 0 = all 3 stages loaded without error.
Exit 1 = one or more stages failed (errors printed above).
"""

import sys
import os
import copy
import glob
import traceback
import safetensors.torch
import torch

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    "/workspace/.cache_hf/models--openbmb--MiniCPM-o-4_5/snapshots/"
    "44151b35f1b232a280bda5a87ea1a7575d5433fc"
)

# ── load all HF weights ────────────────────────────────────────────────────
print(f"Loading weights from {MODEL_DIR} ...", flush=True)
all_weights = {}
for f in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
    all_weights.update(safetensors.torch.load_file(f, device="cpu"))
print(f"  {len(all_weights)} total keys loaded", flush=True)


def iter_weights(prefix=None):
    """Yield (name, tensor) pairs, optionally filtered by prefix."""
    for k, v in all_weights.items():
        if prefix is None or k.startswith(prefix):
            yield k, v


# ── Fake configs (bypass all pydantic/vllm constructor validation) ─────────

class _FakeAttr:
    """Returns self for any attribute access, avoiding AttributeError chains."""

    def __getattr__(self, name):
        return _FakeAttr()

    def __bool__(self):
        return False

    def __int__(self):
        return 0

    def __iter__(self):
        return iter([])

    def __call__(self, *a, **kw):
        return _FakeAttr()


class FakeModelConfig:
    """Minimal model_config that satisfies vllm model __init__ access patterns."""

    def __init__(self, model_dir: str, stage: str, hf_config):
        self.model = model_dir
        self.model_stage = stage
        self.hf_config = hf_config
        # hf_text_config: needed by some sub-models (Qwen3, Llama, etc.)
        self.hf_text_config = (
            getattr(hf_config, "text_config", None) or hf_config
        )
        self.dtype = torch.float32
        self.seed = 0
        self.quantization = None
        self.quantization_param_path = None
        self.revision = None
        self.tokenizer = model_dir
        self.tokenizer_revision = None
        self.tokenizer_mode = "auto"
        self.trust_remote_code = True
        self.skip_tokenizer_init = False
        self.max_model_len = 32768
        self.served_model_name = "test"
        self.enforce_eager = True
        self.disable_sliding_window = False
        self.mm_processor_kwargs = None
        self.task = None
        self.runner_type = "generate"
        self.convert_type = None
        # OmniModelConfig fields
        self.model_arch = None
        self.worker_type = None
        self.engine_output_type = None
        self.hf_config_name = None
        self.custom_process_next_stage_input_func = None
        self.stage_connector_config = {"name": "SharedMemoryConnector", "extra": {}}
        self.omni_kv_config = None
        self.codec_frame_rate_hz = None
        self.task_type = None
        self.stage_id = 0

    def get_hidden_size(self):
        return getattr(self.hf_config, "hidden_size", 4096)

    def __getattr__(self, name):
        # Return _FakeAttr() for any missing field so attribute chains don't
        # raise AttributeError (e.g. model_config.multimodal_config.mm_encoder_only)
        return _FakeAttr()


class FakeVllmConfig:
    """Minimal VllmConfig that supports with_hf_config without pydantic replace()."""

    def __init__(self, model_config: FakeModelConfig, quant_config=None):
        self.model_config = model_config
        self.quant_config = quant_config

        # CacheConfig stub
        cache = _FakeAttr()
        cache.cache_dtype = "auto"
        self.cache_config = cache

        # ParallelConfig stub
        pc = _FakeAttr()
        pc.tensor_parallel_size = 1
        pc.pipeline_parallel_size = 1
        self.parallel_config = pc

        # SchedulerConfig stub
        sc = _FakeAttr()
        sc.max_num_seqs = 1
        self.scheduler_config = sc

        # DeviceConfig stub
        dc = _FakeAttr()
        dc.device = torch.device("cpu")
        dc.device_type = "cpu"
        self.device_config = dc

        # LoadConfig stub
        lc = _FakeAttr()
        lc.load_format = "auto"
        lc.download_dir = None
        self.load_config = lc

        # CompilationConfig stub — needs concrete lists for splitting_ops etc.
        cc = _FakeAttr()
        cc.splitting_ops = []
        cc.capture_sizes = []
        cc.cudagraph_capture_sizes = []
        cc.cudagraph_num_of_warmup_steps = 0
        cc.use_inductor = False
        cc.use_cudagraph = False
        cc.pass_config = _FakeAttr()
        self.compilation_config = cc

        self.lora_config = None
        self.speculative_config = None
        self.structured_outputs_config = None
        self.observability_config = None
        self.kv_transfer_config = None
        self.kv_events_config = None
        self.ec_transfer_config = None
        self.reasoning_config = None

    def with_hf_config(self, hf_config, architectures=None):
        """Create a sub-config with a different hf_config (no pydantic replace)."""
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        new_mc = copy.copy(self.model_config)
        new_mc.hf_config = hf_config
        new_mc.hf_text_config = (
            getattr(hf_config, "text_config", None) or hf_config
        )

        new_cfg = FakeVllmConfig(new_mc, self.quant_config)
        return new_cfg

    def __getattr__(self, name):
        return _FakeAttr()


# ── helpers ────────────────────────────────────────────────────────────────
def make_fake_vllm_config(stage: str, model_dir: str) -> FakeVllmConfig:
    """Create a minimal VllmConfig-like object sufficient for model __init__."""
    import transformers

    hf_config = transformers.AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True
    )
    model_config = FakeModelConfig(model_dir, stage, hf_config)
    return FakeVllmConfig(model_config)


# ── test each stage ────────────────────────────────────────────────────────
errors = []

# ---- Stage 0: Thinker -------------------------------------------------------
print("\n=== Stage 0: Thinker ===", flush=True)
try:
    from vllm_omni.model_executor.models.minicpm_o.minicpm_o import (
        MiniCPMOForConditionalGeneration,
    )
    vllm_config = make_fake_vllm_config("thinker", MODEL_DIR)
    model = MiniCPMOForConditionalGeneration(vllm_config=vllm_config)
    model.eval()

    loaded = model.load_weights(iter_weights())
    print(f"  OK — {len(loaded)} keys loaded", flush=True)
except Exception as e:
    print(f"  FAIL: {e}", flush=True)
    traceback.print_exc()
    errors.append(("thinker", str(e)))

# ---- Stage 1: Talker --------------------------------------------------------
print("\n=== Stage 1: Talker ===", flush=True)
try:
    vllm_config = make_fake_vllm_config("talker", MODEL_DIR)
    model = MiniCPMOForConditionalGeneration(vllm_config=vllm_config)
    model.eval()

    loaded = model.load_weights(iter_weights())
    print(f"  OK — {len(loaded)} keys loaded", flush=True)
except Exception as e:
    print(f"  FAIL: {e}", flush=True)
    traceback.print_exc()
    errors.append(("talker", str(e)))

# ---- Stage 2: Code2Wav -------------------------------------------------------
print("\n=== Stage 2: Code2Wav ===", flush=True)
try:
    vllm_config = make_fake_vllm_config("code2wav", MODEL_DIR)
    model = MiniCPMOForConditionalGeneration(vllm_config=vllm_config)
    model.eval()

    loaded = model.load_weights(iter_weights())
    print(f"  OK — {len(loaded)} keys loaded", flush=True)
except Exception as e:
    print(f"  FAIL: {e}", flush=True)
    traceback.print_exc()
    errors.append(("code2wav", str(e)))

# ── summary ───────────────────────────────────────────────────────────────
print("\n=== Summary ===", flush=True)
if errors:
    for stage, msg in errors:
        print(f"  FAIL [{stage}]: {msg}", flush=True)
    sys.exit(1)
else:
    print("  All stages PASS", flush=True)
    sys.exit(0)
