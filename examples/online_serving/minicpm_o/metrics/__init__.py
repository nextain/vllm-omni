"""
Multi-dimensional conversation quality metrics for speech dialogue evaluation.

Based on 2024-2025 research on objective dialogue evaluation:

Dimensions:
1. STT Accuracy: Speech recognition quality (CER for CJK, WER for English)
2. TTS Quality: Audio intelligibility, naturalness, silence ratio
3. Conversation Flow: Context awareness, coherence
4. Semantic Similarity: Embedding-based meaning alignment
5. Knowledge Retention: Memory across conversation turns

References:
- DialZara: 5 Metrics for Conversational AI Evaluation (2025)
- arXiv:2503.22458: Evaluating LLM-based Multi-Turn Conversations
- arXiv:2407.18538: Multidimensional Empathy Evaluation Framework
- PairEval: Pairwise dialogue comparison with learned metrics
"""

from __future__ import annotations

from .cjk_metrics import (
    calculate_cjk_metrics,
    cer,
    semantic_similarity,
    unicode_normalize,
)


__all__ = [
    "calculate_cjk_metrics",
    "cer",
    "semantic_similarity",
    "unicode_normalize",
]
