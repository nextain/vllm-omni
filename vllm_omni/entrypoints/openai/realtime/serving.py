from typing import Any, List, Optional
from vllm.entrypoints.openai.serving_engine import OpenAIServing

class OpenAIServingRealtime(OpenAIServing):
    """Minimal serving class for OpenAI Realtime API."""
    def __init__(self, engine_client: Any, model_config: Any, served_model_names: List[str]):
        super().__init__(engine_client, model_config, served_model_names)
