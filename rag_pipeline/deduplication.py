"""Advanced deduplication system using multiple strategies."""

import hashlib
from typing import Set, Tuple
from collections import defaultdict
import re


class ContentNormalizer:
    """Normalize content for better deduplication."""

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text by:
        - Converting to lowercase
        - Removing extra whitespace
        - Removing punctuation variations
        - Stripping special characters
        """
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters but keep words
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra spaces again
        text = ' '.join(text.split())

        return text

    @staticmethod
    def get_content_hash(text: str) -> str:
        """Get SHA256 hash of normalized content."""
        normalized = ContentNormalizer.normalize(text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    @staticmethod
    def get_content_signature(text: str, window_size: int = 5) -> Set[str]:
        """
        Generate a set of n-grams as content signature.
        This helps detect similar content even with minor variations.
        """
        normalized = ContentNormalizer.normalize(text)
        words = normalized.split()

        ngrams = set()
        for i in range(len(words) - window_size + 1):
            ngram = ' '.join(words[i:i + window_size])
            ngrams.add(ngram)

        return ngrams

    @staticmethod
    def similarity_score(text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts using n-grams.
        Returns score between 0 and 1 (1 = identical).
        """
        sig1 = ContentNormalizer.get_content_signature(text1)
        sig2 = ContentNormalizer.get_content_signature(text2)

        if not sig1 or not sig2:
            return 0.0

        intersection = len(sig1 & sig2)
        union = len(sig1 | sig2)

        return intersection / union if union > 0 else 0.0


class DedupliationTracker:
    """Track and manage deduplication across ingestions."""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplication tracker.

        Args:
            similarity_threshold: Score (0-1) above which chunks are considered duplicates
        """
        self.similarity_threshold = similarity_threshold

        # Store exact hashes for fast lookup
        self.exact_hashes: Set[str] = set()

        # Store content signatures for similarity detection
        self.content_signatures: dict = {}  # hash -> signature

        # Store chunk IDs for each hash (for tracking)
        self.hash_to_ids: defaultdict = defaultdict(list)

    def is_duplicate_exact(self, content: str) -> bool:
        """Check if content is an exact duplicate."""
        content_hash = ContentNormalizer.get_content_hash(content)
        return content_hash in self.exact_hashes

    def is_duplicate_similar(self, content: str) -> Tuple[bool, float]:
        """
        Check if content is similar to existing content.

        Returns:
            Tuple of (is_duplicate, max_similarity_score)
        """
        max_similarity = 0.0

        for existing_content in self.content_signatures.values():
            similarity = ContentNormalizer.similarity_score(content, existing_content)
            max_similarity = max(max_similarity, similarity)

            if similarity >= self.similarity_threshold:
                return True, similarity

        return False, max_similarity

    def add_content(self, content: str, chunk_id: str) -> bool:
        """
        Add content to deduplication tracker.

        Returns:
            True if added (not duplicate), False if duplicate
        """
        # Check exact match first (fast)
        if self.is_duplicate_exact(content):
            return False

        # Check similarity (slower but more thorough)
        is_similar, _ = self.is_duplicate_similar(content)
        if is_similar:
            return False

        # Not a duplicate, add it
        content_hash = ContentNormalizer.get_content_hash(content)
        signature = ContentNormalizer.get_content_signature(content)

        self.exact_hashes.add(content_hash)
        self.content_signatures[content_hash] = content
        self.hash_to_ids[content_hash].append(chunk_id)

        return True

    def clear(self):
        """Clear all deduplication data."""
        self.exact_hashes.clear()
        self.content_signatures.clear()
        self.hash_to_ids.clear()

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            "unique_hashes": len(self.exact_hashes),
            "total_chunk_ids": sum(len(ids) for ids in self.hash_to_ids.values()),
        }


# Global deduplication tracker instance
_dedup_tracker: DedupliationTracker = None


def get_dedup_tracker() -> DedupliationTracker:
    """Get or create global deduplication tracker."""
    global _dedup_tracker
    if _dedup_tracker is None:
        _dedup_tracker = DedupliationTracker(similarity_threshold=0.95)
    return _dedup_tracker
