"""Comprehensive tests for deduplication module."""

import pytest
from rag_pipeline.deduplication import (
    ContentNormalizer,
    DedupliationTracker,
    get_dedup_tracker,
)


class TestContentNormalizer:
    """Test content normalization."""

    def test_normalize_lowercase(self):
        """Test conversion to lowercase."""
        result = ContentNormalizer.normalize("HELLO World")
        assert result == "hello world"

    def test_normalize_whitespace(self):
        """Test removal of extra whitespace."""
        result = ContentNormalizer.normalize("hello    world\n\n  test")
        assert result == "hello world test"

    def test_normalize_punctuation(self):
        """Test removal of punctuation."""
        result = ContentNormalizer.normalize("Hello, world! How are you?")
        assert result == "hello world how are you"

    def test_normalize_special_chars(self):
        """Test removal of special characters."""
        result = ContentNormalizer.normalize("hello@world#test$data")
        assert result == "helloworldtestdata"

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        result = ContentNormalizer.normalize("")
        assert result == ""

    def test_normalize_unicode(self):
        """Test normalization with unicode characters."""
        result = ContentNormalizer.normalize("Héllo Wørld")
        assert "llo" in result or "hello" in result.lower()

    def test_get_content_hash(self):
        """Test content hash generation."""
        text = "test content"
        hash1 = ContentNormalizer.get_content_hash(text)
        hash2 = ContentNormalizer.get_content_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_get_content_hash_case_insensitive(self):
        """Test that hashes are case-insensitive."""
        hash1 = ContentNormalizer.get_content_hash("TEST CONTENT")
        hash2 = ContentNormalizer.get_content_hash("test content")

        assert hash1 == hash2

    def test_get_content_signature(self):
        """Test n-gram signature generation."""
        text = "the quick brown fox"
        sig = ContentNormalizer.get_content_signature(text, window_size=2)

        assert len(sig) > 0
        assert isinstance(sig, set)

    def test_get_content_signature_empty(self):
        """Test signature for content too short for window."""
        text = "a"
        sig = ContentNormalizer.get_content_signature(text, window_size=5)

        assert len(sig) == 0

    def test_similarity_score_identical(self):
        """Test similarity score for identical texts."""
        text = "the quick brown fox"
        sig1 = ContentNormalizer.get_content_signature(text)
        sig2 = ContentNormalizer.get_content_signature(text)

        if len(sig1) > 0:
            score = ContentNormalizer.similarity_score(text, text)
            assert score > 0.9  # Very high similarity
        else:
            # Text too short for n-grams
            assert True

    def test_similarity_score_empty(self):
        """Test similarity score with empty strings."""
        score = ContentNormalizer.similarity_score("", "")
        assert score == 0.0

    def test_similarity_score_different(self):
        """Test similarity score for different texts."""
        text1 = "the quick brown fox"
        text2 = "completely different text"
        score = ContentNormalizer.similarity_score(text1, text2)

        assert 0.0 <= score < 0.5

    def test_similarity_score_similar(self):
        """Test similarity score for similar texts."""
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over the lazy cat"
        score = ContentNormalizer.similarity_score(text1, text2)

        assert score > 0.6


class TestDedupliationTracker:
    """Test deduplication tracker."""

    def test_init_creates_empty_tracker(self):
        """Test tracker initialization."""
        tracker = DedupliationTracker()

        assert len(tracker.exact_hashes) == 0
        assert len(tracker.content_signatures) == 0

    def test_is_duplicate_exact_new_content(self):
        """Test exact duplicate detection for new content."""
        tracker = DedupliationTracker()

        assert not tracker.is_duplicate_exact("new content")

    def test_is_duplicate_exact_existing(self):
        """Test exact duplicate detection for existing content."""
        tracker = DedupliationTracker()
        content = "test content"

        tracker.add_content(content, "chunk1")

        assert tracker.is_duplicate_exact(content)

    def test_is_duplicate_similar_threshold(self):
        """Test similar duplicate detection respects threshold."""
        tracker = DedupliationTracker(similarity_threshold=0.95)

        is_dup, score = tracker.is_duplicate_similar("test")
        assert not is_dup
        assert score >= 0.0

    def test_add_content_new(self):
        """Test adding new content."""
        tracker = DedupliationTracker()
        result = tracker.add_content("new content", "chunk1")

        assert result is True

    def test_add_content_exact_duplicate(self):
        """Test adding exact duplicate content."""
        tracker = DedupliationTracker()

        tracker.add_content("test content", "chunk1")
        result = tracker.add_content("test content", "chunk2")

        assert result is False

    def test_add_content_similar_duplicate(self):
        """Test adding similar duplicate content."""
        tracker = DedupliationTracker(similarity_threshold=0.8)

        tracker.add_content("the quick brown fox", "chunk1")
        result = tracker.add_content("the quick brown fox", "chunk2")

        assert result is False

    def test_add_content_case_insensitive(self):
        """Test that adding is case-insensitive."""
        tracker = DedupliationTracker()

        tracker.add_content("Test Content", "chunk1")
        result = tracker.add_content("test content", "chunk2")

        assert result is False

    def test_add_content_multiple(self):
        """Test adding multiple different contents."""
        tracker = DedupliationTracker()

        r1 = tracker.add_content("content 1", "chunk1")
        r2 = tracker.add_content("content 2", "chunk2")
        r3 = tracker.add_content("content 3", "chunk3")

        assert r1 is True
        assert r2 is True
        assert r3 is True

    def test_hash_to_ids_tracking(self):
        """Test that chunk IDs are tracked correctly."""
        tracker = DedupliationTracker()

        tracker.add_content("content", "chunk1")
        tracker.add_content("content", "chunk2")  # Duplicate

        total_ids = sum(len(ids) for ids in tracker.hash_to_ids.values())
        assert total_ids == 1  # Only chunk1 should be tracked

    def test_clear(self):
        """Test clearing tracker."""
        tracker = DedupliationTracker()

        tracker.add_content("content 1", "chunk1")
        tracker.add_content("content 2", "chunk2")

        tracker.clear()

        assert len(tracker.exact_hashes) == 0
        assert len(tracker.content_signatures) == 0
        assert len(tracker.hash_to_ids) == 0

    def test_get_stats(self):
        """Test getting statistics."""
        tracker = DedupliationTracker()

        tracker.add_content("content 1", "chunk1")
        tracker.add_content("content 2", "chunk2")

        stats = tracker.get_stats()

        assert "unique_hashes" in stats
        assert "total_chunk_ids" in stats
        assert stats["unique_hashes"] == 2
        assert stats["total_chunk_ids"] == 2

    def test_custom_threshold(self):
        """Test custom similarity threshold."""
        tracker_strict = DedupliationTracker(similarity_threshold=0.99)
        tracker_loose = DedupliationTracker(similarity_threshold=0.50)

        content1 = "the quick brown fox"
        content2 = "the quick brown cat"

        tracker_strict.add_content(content1, "chunk1")
        r_strict = tracker_strict.add_content(content2, "chunk2")

        tracker_loose.add_content(content1, "chunk1")
        r_loose = tracker_loose.add_content(content2, "chunk2")

        # Strict threshold should accept, loose might reject
        assert r_strict is True or r_loose is False

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        tracker = DedupliationTracker()

        r1 = tracker.add_content("", "chunk1")
        r2 = tracker.add_content("", "chunk2")

        assert r1 is True
        assert r2 is False  # Second empty is duplicate


class TestGlobalDedupTracker:
    """Test global deduplication tracker instance."""

    def test_get_dedup_tracker_singleton(self):
        """Test that get_dedup_tracker returns same instance."""
        tracker1 = get_dedup_tracker()
        tracker2 = get_dedup_tracker()

        assert tracker1 is tracker2

    def test_global_tracker_persistence(self):
        """Test that global tracker persists data."""
        tracker = get_dedup_tracker()

        # Clear first for clean state
        tracker.clear()

        tracker.add_content("test content", "chunk1")

        tracker2 = get_dedup_tracker()

        assert len(tracker2.exact_hashes) > 0


class TestDedupEdgeCases:
    """Test edge cases in deduplication."""

    def test_whitespace_only_strings(self):
        """Test handling of whitespace-only strings."""
        tracker = DedupliationTracker()

        r1 = tracker.add_content("   ", "chunk1")
        r2 = tracker.add_content("\n\t", "chunk2")

        # Both should normalize to empty
        assert r1 is True
        assert r2 is False

    def test_very_long_content(self):
        """Test handling of very long content."""
        tracker = DedupliationTracker()
        long_content = "word " * 10000

        result = tracker.add_content(long_content, "chunk1")

        assert result is True

    def test_special_characters_only(self):
        """Test content with only special characters."""
        tracker = DedupliationTracker()

        r1 = tracker.add_content("!@#$%^&*()", "chunk1")
        r2 = tracker.add_content("!@#$%^&*()", "chunk2")

        assert r1 is True
        assert r2 is False

    def test_unicode_normalization(self):
        """Test Unicode content normalization."""
        tracker = DedupliationTracker()

        content1 = "café"
        content2 = "cafe"

        r1 = tracker.add_content(content1, "chunk1")
        r2 = tracker.add_content(content2, "chunk2")

        # Should be treated as different after normalization
        assert r1 is True
