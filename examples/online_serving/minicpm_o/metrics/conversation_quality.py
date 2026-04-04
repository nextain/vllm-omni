#!/usr/bin/env python3
"""Objective conversation quality metrics for multi-turn dialogue evaluation.

Based on 2024-2025 research:
- DialZara: 5 Metrics for Conversational AI (2025)
- arXiv:2503.22458: Multi-turn conversation evaluation
- ChatChecker: Dialogue System Testing Framework (arXiv:2507.16792)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class ConversationQualityMetrics:
    """Comprehensive conversation quality metrics."""

    # STT/Audio Input Quality
    stt_accuracy: float  # 1.0 - CER (CJK) or 1.0 - WER (English); higher = better
    stt_semantic_similarity: float  # Embedding-based meaning match

    # TTS/Audio Output Quality
    tts_speech_ratio: float  # Ratio of speech vs silence in audio
    tts_intelligibility: float  # Round-trip STT of TTS output

    # Conversation Quality
    relevance: float  # Response relevance to user input
    coherence: float  # Logical flow and context awareness
    knowledge_retention: float  # Memory across turns

    # Efficiency
    response_time_s: float
    audio_duration_s: float

    # Overall scores
    overall_quality: float  # Weighted combination (0-1)


class ConversationEvaluator:
    """
    Objective conversation quality evaluator.

    References:
    - DialZara (2025): 5 metrics framework
    - PairEval: Learned metrics with 0.6-0.8 human correlation
    - ChatChecker: LLM-as-a-judge framework
    """

    def __init__(self, semantic_model: Optional[Callable] = None):
        """
        Args:
            semantic_model: Optional semantic similarity function. If None, uses fallback.
        """
        self.semantic_model = semantic_model

    def calculate_relevance(self, user_input: str, response: str) -> float:
        """
        Calculate response relevance to user input.

        References:
        - Conversation Relevancy Score (2025 research)
        - DialZara: Measures how well response addresses user query

        Returns:
            Relevance score 0.0-1.0
        """
        # Character overlap as baseline
        user_words = set(re.findall(r'\w+', user_input.lower()))
        resp_words = set(re.findall(r'\w+', response.lower()))

        if not user_words:
            return 1.0

        # Overlap + length penalty for verbose responses
        overlap = len(user_words & resp_words) / len(user_words)
        length_ratio = min(1.0, len(user_input) / max(len(response), 1))

        # Relevance = overlap * optimal_length_factor
        optimal_length_factor = 1.0 - abs(length_ratio - 0.5) * 0.5
        return overlap * optimal_length_factor

    def calculate_coherence(self, turns: list[dict]) -> float:
        """
        Calculate conversation coherence (logical flow).

        References:
        - Conversation Flow (2025 research)
        - DialZara: Smooth, context-aware dialogue

        Args:
            turns: List of conversation turns with 'role' and 'text'

        Returns:
            Coherence score 0.0-1.0
        """
        if len(turns) < 2:
            return 1.0

        scores = []
        for i in range(len(turns) - 1):
            current = turns[i]
            next_turn = turns[i + 1]

            # Coherence: response should be contextually relevant
            if current['role'] != next_turn['role']:
                # Different speakers: check topic relevance
                scores.append(self._calculate_relevance(
                    current['text'],
                    next_turn['text']
                ))
            else:
                # Same speaker: check flow continuity
                scores.append(self._calculate_flow_continuity(
                    current['text'],
                    next_turn['text']
                ))

        return float(np.mean(scores)) if scores else 1.0

    def _calculate_relevance(self, prev: str, curr: str) -> float:
        """Helper: calculate relevance between two consecutive turns."""
        if self.semantic_model:
            return self.semantic_model(prev, curr)
        else:
            # Fallback: character overlap
            return self._character_overlap(prev, curr)

    def _calculate_flow_continuity(self, prev: str, curr: str) -> float:
        """Helper: check if conversation flows smoothly."""
        # Simple heuristic: check for abrupt topic changes
        prev_words = set(prev.lower().split())
        curr_words = set(curr.lower().split())

        # Some overlap = coherent; no overlap = possible topic switch
        overlap = len(prev_words & curr_words) / max(len(prev_words), 1)
        return min(1.0, overlap + 0.3)  # 0.3 baseline for new topics

    def _character_overlap(self, text1: str, text2: str) -> float:
        """Fallback character overlap for semantic similarity (multiset)."""
        from collections import Counter
        from .cjk_metrics import unicode_normalize

        ref = list(unicode_normalize(text1))
        hyp = list(unicode_normalize(text2))

        if not ref:
            return 1.0
        if not hyp:
            return 0.0
        ref_counts = Counter(ref)
        hyp_counts = Counter(hyp)
        intersection = sum((ref_counts & hyp_counts).values())
        return intersection / len(ref)

    def calculate_knowledge_retention(
        self,
        original_fact: str,
        later_recall: str,
    ) -> float:
        """
        Calculate knowledge retention across conversation.

        References:
        - DialZara: Knowledge retention metric
        - 2025 research: Memory across conversations

        Args:
            original_fact: Fact mentioned earlier (e.g., "I have two cats named Luna")
            later_recall: Later reference to that fact

        Returns:
            Retention score 0.0-1.0
        """
        from .cjk_metrics import semantic_similarity

        # Extract key entities (simple heuristic: proper nouns)
        original_entities = set(re.findall(r'[A-Z][a-z]*', original_fact))
        later_entities = set(re.findall(r'[A-Z][a-z]*', later_recall))

        # Check if entities are referenced
        entity_recall = len(original_entities & later_entities) / max(
            len(original_entities), 1
        )

        # Combine with semantic similarity
        semantic_score = semantic_similarity(original_fact, later_recall)

        return (entity_recall + semantic_score) / 2.0

    def calculate_audio_quality(
        self,
        audio_path: str,
        text: str,
    ) -> dict:
        """
        Calculate audio quality metrics.

        Returns:
            Dict with speech_ratio, silence_ratio, intelligibility
        """
        import soundfile as sf

        # Load audio
        data, sr = sf.read(audio_path)

        # Energy-based VAD for speech detection
        frame_length = int(sr * 0.025)  # 25ms frames
        speech_frames = 0

        for i in range(0, len(data), frame_length):
            frame = data[i:i+frame_length]
            energy = np.mean(np.abs(frame))
            # Simple energy threshold
            if energy > 0.01:
                speech_frames += 1

        total_frames = math.ceil(len(data) / frame_length)
        speech_ratio = speech_frames / total_frames if total_frames > 0 else 0.0

        return {
            "speech_ratio": float(speech_ratio),
            "silence_ratio": float(1.0 - speech_ratio),
        }

    def evaluate_conversation(
        self,
        turns: list[dict],
    ) -> ConversationQualityMetrics:
        """
        Comprehensive conversation quality evaluation.

        Args:
            turns: List of conversation turns with:
                - role: Speaker role
                - text: Spoken text
                - audio_path: Path to audio file (optional)

        Returns:
            ConversationQualityMetrics with all dimensions
        """
        if len(turns) < 2:
            return ConversationQualityMetrics(
                stt_accuracy=1.0,
                stt_semantic_similarity=1.0,
                tts_speech_ratio=1.0,
                tts_intelligibility=1.0,
                relevance=1.0,
                coherence=1.0,
                knowledge_retention=1.0,
                response_time_s=0.0,
                audio_duration_s=0.0,
                overall_quality=1.0,
            )

        # Calculate individual metrics
        stt_accuracies = []
        stt_similarities = []
        speech_ratios = []
        response_times = []

        for turn in turns:
            # STT metrics (if transcript available)
            if 'stt_text' in turn:
                from .cjk_metrics import cer
                stt_acc = 1.0 - cer(turn['text'], turn['stt_text'])
                stt_sim = self._calculate_relevance(
                    turn['text'],
                    turn['stt_text']
                )
                stt_accuracies.append(stt_acc)
                stt_similarities.append(stt_sim)

            # TTS audio quality
            if 'audio_path' in turn:
                audio_qual = self.calculate_audio_quality(
                    turn['audio_path'],
                    turn['text']
                )
                speech_ratios.append(audio_qual['speech_ratio'])

            # Efficiency
            if 'response_time' in turn:
                response_times.append(turn['response_time'])

        # Conversation-level metrics
        relevance_scores = []
        for i in range(len(turns) - 1):
            if turns[i]['role'] != turns[i+1]['role']:
                rel = self.calculate_relevance(
                    turns[i]['text'],
                    turns[i+1]['text']
                )
                relevance_scores.append(rel)

        # Knowledge retention: compare each turn with the most recent prior
        # turn by the same speaker (skip if no prior same-role turn exists)
        knowledge_scores = []
        for i in range(2, len(turns)):
            later = turns[i]
            # Find most recent prior turn with same role
            for j in range(i - 1, -1, -1):
                if turns[j]['role'] == later['role']:
                    fact = turns[j]['text']
                    knowledge = self.calculate_knowledge_retention(
                        fact, later['text']
                    )
                    knowledge_scores.append(knowledge)
                    break

        # Calculate overall scores
        metrics = ConversationQualityMetrics(
            stt_accuracy=float(np.mean(stt_accuracies)) if stt_accuracies else 1.0,
            stt_semantic_similarity=float(np.mean(stt_similarities)) if stt_similarities else 1.0,
            tts_speech_ratio=float(np.mean(speech_ratios)) if speech_ratios else 1.0,
            tts_intelligibility=1.0,  # Requires round-trip STT of audio
            relevance=float(np.mean(relevance_scores)) if relevance_scores else 1.0,
            coherence=self.calculate_coherence(turns),
            knowledge_retention=float(np.mean(knowledge_scores)) if knowledge_scores else 1.0,
            response_time_s=float(np.mean(response_times)) if response_times else 0.0,
            audio_duration_s=0.0,  # Calculate from audio files
            overall_quality=0.0,  # Weighted average below
        )

        # Overall quality: weighted combination (text/semantic dimensions only)
        # tts_speech_ratio (VAD ratio) excluded — measures speech frame fraction,
        # not model quality; would penalize slower or shorter responses unfairly.
        weights = {
            'stt_accuracy': 0.30,
            'relevance': 0.30,
            'coherence': 0.25,
            'knowledge_retention': 0.15,
        }

        metrics.overall_quality = (
            weights['stt_accuracy'] * metrics.stt_accuracy +
            weights['relevance'] * metrics.relevance +
            weights['coherence'] * metrics.coherence +
            weights['knowledge_retention'] * metrics.knowledge_retention
        )

        return metrics


def format_conversation_report(metrics: ConversationQualityMetrics) -> str:
    """Format metrics for human-readable report."""
    report = f"""
    === Conversation Quality Report ===

    STT / Input Quality:
      Accuracy:           {metrics.stt_accuracy:6.2%}
      Semantic Similarity:  {metrics.stt_semantic_similarity:6.2%}

    TTS / Output Quality:
      Speech Ratio:        {metrics.tts_speech_ratio:6.2%}
      Intelligibility:      {metrics.tts_intelligibility:6.2%}

    Conversation Quality:
      Relevance:          {metrics.relevance:6.2%}
      Coherence:          {metrics.coherence:6.2%}
      Knowledge Retention:  {metrics.knowledge_retention:6.2%}

    Efficiency:
      Response Time:      {metrics.response_time_s:.2f}s
      Audio Duration:     {metrics.audio_duration_s:.2f}s

    === Overall Quality: {metrics.overall_quality:6.2%} ===
    """
    return report.strip()


if __name__ == "__main__":
    # Test conversation quality evaluator
    print("Testing Conversation Quality Metrics...\n")

    evaluator = ConversationEvaluator()

    # Sample conversation
    sample_turns = [
        {
            "role": "User",
            "text": "I have two cats named Luna and Stella. Luna is orange.",
            "response_time": 1.5,
        },
        {
            "role": "AI",
            "text": "That's great! What colors are your cats?",
            "response_time": 2.0,
        },
        {
            "role": "User",
            "text": "Stella is gray. They're both very cute.",
            "response_time": 1.2,
        },
        {
            "role": "AI",
            "text": "Orange Luna and gray Stella sound like lovely companions!",
            "response_time": 1.8,
        },
    ]

    # Add mock STT for AI responses
    for i, turn in enumerate(sample_turns):
        if turn['role'] == 'AI':
            turn['stt_text'] = turn['text']  # Perfect recognition for now

    metrics = evaluator.evaluate_conversation(sample_turns)
    print(format_conversation_report(metrics))
