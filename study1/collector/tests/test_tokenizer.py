"""Tests for tokenizer module - Phase 05: Tokenization and Word Frequency"""

import pytest
from src.tokenizer import (
    extract_words,
    load_stopwords,
    remove_stopwords,
    tokenize_and_count,
    add_word_frequencies,
    get_vocabulary
)


def test_extract_words_basic():
    """Extract words from simple text"""
    text = "The quick brown fox jumps over the lazy dog"
    words = extract_words(text.lower())

    assert "quick" in words
    assert "brown" in words
    assert "jumps" in words
    # Single and two-letter words excluded
    assert "a" not in words  # if present


def test_extract_words_min_length():
    """Only extract words with 3+ characters"""
    text = "I am a developer who writes code"
    words = extract_words(text.lower())

    # 1-2 char words excluded
    assert "am" not in words
    # 3+ char words included
    assert "developer" in words
    assert "writes" in words
    assert "code" in words


def test_extract_words_punctuation():
    """Handle punctuation correctly"""
    text = "code, formatting; rules: structure."
    words = extract_words(text.lower())

    assert "code" in words
    assert "formatting" in words
    assert "rules" in words
    assert "structure" in words
    # Punctuation should be removed
    assert "code," not in words


def test_load_stopwords():
    """Load stopwords successfully"""
    stopwords = load_stopwords()

    assert isinstance(stopwords, set)
    assert len(stopwords) > 0
    assert "the" in stopwords
    assert "and" in stopwords
    assert "is" in stopwords


def test_remove_stopwords():
    """Remove stopwords from word list"""
    words = ["the", "quick", "brown", "fox", "is", "fast"]
    stopwords = {"the", "is", "a", "an"}

    filtered = remove_stopwords(words, stopwords)

    assert "quick" in filtered
    assert "brown" in filtered
    assert "fox" in filtered
    assert "fast" in filtered
    assert "the" not in filtered
    assert "is" not in filtered


def test_tokenize_and_count():
    """Tokenize text and count frequencies"""
    text = "code code code format format rule"

    frequencies = tokenize_and_count(text)

    assert frequencies["code"] == 3
    assert frequencies["format"] == 2
    assert frequencies["rule"] == 1


def test_tokenize_and_count_with_stopwords():
    """Stopwords should be excluded from counts"""
    text = "the code is the best code in the world"

    frequencies = tokenize_and_count(text)

    assert "code" in frequencies
    assert frequencies["code"] == 2
    assert "best" in frequencies
    # Stopwords should not be in frequencies
    assert "the" not in frequencies
    assert "is" not in frequencies
    assert "in" not in frequencies


def test_tokenize_case_insensitive():
    """Tokenization should be case-insensitive"""
    text = "Code CODE code"

    frequencies = tokenize_and_count(text)

    # All variants should count as same word
    assert frequencies["code"] == 3


def test_add_word_frequencies_to_artifacts():
    """Add word frequencies to artifact list"""
    artifacts = [
        {
            "file_path": "test.txt",
            "text_content": "code format rule code",
            "is_binary": False
        }
    ]

    result = add_word_frequencies(artifacts)

    assert len(result) == 1
    assert "word_frequencies" in result[0]
    assert result[0]["word_frequencies"]["code"] == 2
    assert result[0]["word_frequencies"]["format"] == 1
    assert "word_count" in result[0]
    assert "unique_terms" in result[0]


def test_skip_binary_files():
    """Skip binary files when adding frequencies"""
    artifacts = [
        {
            "file_path": "binary.bin",
            "text_content": None,
            "is_binary": True
        }
    ]

    result = add_word_frequencies(artifacts)

    assert "word_frequencies" not in result[0] or result[0]["word_frequencies"] == {}


def test_word_count_calculation():
    """Calculate total word count correctly"""
    artifacts = [
        {
            "text_content": "code format rule structure design pattern",
            "is_binary": False
        }
    ]

    result = add_word_frequencies(artifacts)

    # Should count total words (after stopword removal)
    assert result[0]["word_count"] == 6


def test_unique_terms_calculation():
    """Calculate unique terms correctly"""
    artifacts = [
        {
            "text_content": "code code format format rule",
            "is_binary": False
        }
    ]

    result = add_word_frequencies(artifacts)

    assert result[0]["unique_terms"] == 3  # code, format, rule


def test_get_vocabulary():
    """Get vocabulary across all artifacts"""
    artifacts = [
        {"word_frequencies": {"code": 5, "rule": 3}},
        {"word_frequencies": {"format": 2, "structure": 4}},
        {"word_frequencies": {"code": 1, "design": 2}}
    ]

    vocab = get_vocabulary(artifacts)

    assert "code" in vocab
    assert "rule" in vocab
    assert "format" in vocab
    assert "structure" in vocab
    assert "design" in vocab
    assert len(vocab) == 5


def test_empty_text():
    """Handle empty text gracefully"""
    frequencies = tokenize_and_count("")

    assert frequencies == {}


def test_special_characters():
    """Handle special characters and unicode"""
    text = "code@#$%format!!!rule"

    frequencies = tokenize_and_count(text)

    assert "code" in frequencies
    assert "format" in frequencies
    assert "rule" in frequencies


def test_numbers_excluded():
    """Numbers should be excluded"""
    text = "code 123 format 456 rule"

    frequencies = tokenize_and_count(text)

    assert "123" not in frequencies
    assert "456" not in frequencies
    assert "code" in frequencies


def test_hyphenated_words():
    """Handle hyphenated words"""
    text = "test-driven development well-structured code"

    frequencies = tokenize_and_count(text)

    # Depending on implementation, might split or keep
    # Define expected behavior - we split hyphenated words
    assert "driven" in frequencies or "test" in frequencies
    assert "development" in frequencies
    assert "structured" in frequencies or "well" in frequencies
    assert "code" in frequencies
