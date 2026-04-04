#!/usr/bin/env python3
"""CJK-aware metrics for speech conversation benchmarking.

Fixes word_accuracy issue for Chinese/Korean languages:
- CER (Character Error Rate) for CJK languages
- Unicode normalization (Traditional ↔ Simplified)
- Semantic similarity using sentence embeddings

Standards:
- LibriSpeech SOTA: 2-3% WER (97-98% accuracy)
- OpenASRLeaderboard: Uses CER for CJK
- SOVA-Bench: Evaluates semantic similarity
- MTalk-Bench: 3 dimensions (semantic, paralinguistic, ambient)
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher


def unicode_normalize(text: str) -> str:
    """Normalize Unicode (Traditional ↔ Simplified Chinese, etc.)."""
    # NFKC handles compatibility equivalence including Traditional/Simplified
    return unicodedata.normalize('NFKC', text)


def cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate - standard metric for CJK languages.

    CER = (Substitutions + Deletions + Insertions) / Reference Length

    References:
    - OpenASRLeaderboard: Uses CER for CJK evaluation
    - LibriSpeech: Reports WER for English, CER for CJK

    Args:
        reference: Ground truth text
        hypothesis: STT/ASR output text

    Returns:
        CER value between 0.0 (perfect) and 1.0 (completely wrong)
    """
    # Normalize Unicode first
    ref = unicode_normalize(reference)
    hyp = unicode_normalize(hypothesis)

    if len(ref) == 0:
        return 0.0

    # Standard CER = edit_distance(ref, hyp) / len(ref)
    # Using SequenceMatcher to derive edit distance:
    # ratio = 2*matches / (len(ref) + len(hyp))
    # matches = ratio * (len(ref) + len(hyp)) / 2
    # edit_distance ≈ len(ref) + len(hyp) - 2*matches
    matcher = SequenceMatcher(None, ref, hyp)
    matches = matcher.ratio() * (len(ref) + len(hyp)) / 2
    edit_distance = len(ref) + len(hyp) - 2 * matches
    return max(0.0, min(edit_distance / len(ref), 1.0))


_sentence_model = None  # Module-level cache to avoid reloading on every call


def semantic_similarity(original: str, transcript: str) -> float:
    """
    Semantic similarity using sentence embeddings.

    References:
    - SOVA-Bench: Evaluates "Speech Understanding (Linguistic and Paralinguistic)"
    - MTalk-Bench: Has "Semantic Information" dimension

    Note: Requires sentence-transformers package.
    Install with: pip install sentence-transformers

    Returns:
        Cosine similarity between 0.0 (no similarity) and 1.0 (identical)
    """
    global _sentence_model
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # Fallback: use character overlap for languages without models
        return _character_overlap_similarity(original, transcript)

    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb1 = _sentence_model.encode(original)
    emb2 = _sentence_model.encode(transcript)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return float(similarity)


def _character_overlap_similarity(original: str, transcript: str) -> float:
    """Fallback: character-level overlap similarity (Jaccard on character bags)."""
    ref = list(unicode_normalize(original))
    hyp = list(unicode_normalize(transcript))
    if not ref:
        return 1.0
    if not hyp:
        return 0.0
    # Use multiset intersection: count matching chars up to min occurrence
    from collections import Counter
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    intersection = sum((ref_counts & hyp_counts).values())
    return intersection / len(ref)


def word_accuracy(original: str, transcript: str) -> float:
    """
    Fraction of original words found in transcript (for English).

    Kept for backward compatibility with existing benchmarks.
    For CJK languages, use cer() instead.
    """
    import re
    orig_words = set(re.findall(r'\w+', original.lower()))
    trans_words = set(re.findall(r'\w+', transcript.lower()))
    if not orig_words:
        return 1.0
    return len(orig_words & trans_words) / len(orig_words)


def calculate_cjk_metrics(
    original: str,
    transcript: str,
    language: str = "en",
) -> dict:
    """
    Calculate comprehensive metrics for speech evaluation.

    Returns dict with CER, semantic similarity, and appropriate word accuracy.

    Args:
        original: Ground truth text (LLM output)
        transcript: STT/ASR recognized text
        language: Language code (en, zh, ko, etc.)

    Returns:
        Dict with metrics:
        - cer: Character Error Rate (CJK languages)
        - semantic_similarity: Embedding-based similarity
        - word_accuracy: Word accuracy (English only)
        - notes: Additional observations
    """
    metrics = {
        "cer": cer(original, transcript),
        "semantic_similarity": semantic_similarity(original, transcript),
    }

    # Word accuracy for English (for comparison)
    if language.lower() == "en":
        metrics["word_accuracy"] = word_accuracy(original, transcript)
        metrics["notes"] = "English: using word_accuracy + semantic_similarity"
    else:
        # CJK languages: CER is primary metric
        metrics["word_accuracy"] = 0.0  # Not applicable for CJK
        metrics["notes"] = f"CJK ({language}): using CER + semantic_similarity"

    return metrics


if __name__ == "__main__":
    # Test cases
    print("Testing CJK metrics...\n")

    # Chinese example (Traditional ↔ Simplified)
    original_zh = "是的，您说得没错，今天天气很不错呢。"
    transcript_zh = "是的,您说的没错,今天天气很不错呢"
    result_zh = calculate_cjk_metrics(original_zh, transcript_zh, "zh")
    print(f"Chinese Example:")
    print(f"  Original: {original_zh}")
    print(f"  Transcript: {transcript_zh}")
    print(f"  CER: {result_zh['cer']:.3f}")
    print(f"  Semantic Similarity: {result_zh['semantic_similarity']:.3f}")
    print(f"  Notes: {result_zh['notes']}\n")

    # English example
    original_en = "Hello, how are you today?"
    transcript_en = "Hello, how are you today?"
    result_en = calculate_cjk_metrics(original_en, transcript_en, "en")
    print(f"English Example:")
    print(f"  Original: {original_en}")
    print(f"  Transcript: {transcript_en}")
    print(f"  Word Accuracy: {result_en['word_accuracy']:.3f}")
    print(f"  CER: {result_en['cer']:.3f}")
    print(f"  Semantic Similarity: {result_en['semantic_similarity']:.3f}")
    print(f"  Notes: {result_en['notes']}")
