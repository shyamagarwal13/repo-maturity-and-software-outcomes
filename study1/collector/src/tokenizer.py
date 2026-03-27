"""Tokenizer module for text analysis and word frequency extraction.

This module provides functions to:
- Extract words from text (3+ alphabetic characters)
- Remove common English stopwords
- Count word frequencies
- Build vocabulary across multiple artifacts

Tokenization rules:
- Case-insensitive (all text converted to lowercase)
- Minimum word length: 3 characters
- Only alphabetic characters (no numbers or punctuation)
- Stopwords removed (common English words)
- No stemming or lemmatization
- Raw frequency counts only
"""

import re
from collections import Counter
from typing import Any, Dict, List, Set


def load_stopwords() -> Set[str]:
    """Load English stopwords list.

    Returns:
        Set of common English stopwords to filter out
    """
    # Using a minimal stopwords list
    return {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "they", "them", "their"
    }


def extract_words(text: str) -> List[str]:
    """Extract valid words from text.

    Rules:
    - Words must have 3+ alphabetic characters
    - Numbers, punctuation, and special characters are excluded

    Args:
        text: Input text to extract words from

    Returns:
        List of extracted words
    """
    # Use regex to find words with 3+ alphabetic characters
    words = re.findall(r'[a-zA-Z]{3,}', text)
    return words


def remove_stopwords(words: List[str], stopwords: Set[str]) -> List[str]:
    """Filter out stopwords from word list.

    Args:
        words: List of words to filter
        stopwords: Set of stopwords to remove

    Returns:
        List of words without stopwords
    """
    return [word for word in words if word not in stopwords]


def tokenize_and_count(text: str) -> Dict[str, int]:
    """Tokenize text and count word frequencies.

    Steps:
    1. Convert text to lowercase
    2. Extract words (3+ characters, alphabetic)
    3. Remove stopwords
    4. Count word frequencies

    Args:
        text: Input text to tokenize

    Returns:
        Dictionary mapping words to their frequencies
    """
    # Convert to lowercase
    text_lower = text.lower()

    # Extract words
    words = extract_words(text_lower)

    # Remove stopwords
    stopwords = load_stopwords()
    filtered_words = remove_stopwords(words, stopwords)

    # Count frequencies
    frequencies = Counter(filtered_words)

    return dict(frequencies)


def add_word_frequencies(artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add word frequencies to all artifacts.

    For each artifact:
    - Skip if is_binary or no text_content
    - Calculate word frequencies from text_content
    - Add word_frequencies, word_count, and unique_terms fields

    Args:
        artifacts: List of artifact dictionaries with text_content field

    Returns:
        Updated artifacts with word_frequencies, word_count, and unique_terms fields

    Example:
        >>> artifacts = [{"text_content": "code code format", "is_binary": False}]
        >>> result = add_word_frequencies(artifacts)
        >>> result[0]["word_frequencies"]
        {'code': 2, 'format': 1}
    """
    for artifact in artifacts:
        # Skip binary files or files without text content
        if artifact.get("is_binary", False) or not artifact.get("text_content"):
            artifact["word_frequencies"] = {}
            continue

        # Tokenize and count
        frequencies = tokenize_and_count(artifact["text_content"])

        # Add fields to artifact
        artifact["word_frequencies"] = frequencies
        artifact["word_count"] = sum(frequencies.values())
        artifact["unique_terms"] = len(frequencies)

    return artifacts


def get_vocabulary(artifacts: List[Dict[str, Any]]) -> Set[str]:
    """Get unique vocabulary across all artifacts.

    Args:
        artifacts: List of artifacts with word_frequencies field

    Returns:
        Set of all unique words found across all artifacts

    Example:
        >>> artifacts = [
        ...     {"word_frequencies": {"code": 5, "rule": 3}},
        ...     {"word_frequencies": {"format": 2, "code": 1}}
        ... ]
        >>> vocab = get_vocabulary(artifacts)
        >>> sorted(vocab)
        ['code', 'format', 'rule']
    """
    vocabulary = set()

    for artifact in artifacts:
        if "word_frequencies" in artifact:
            vocabulary.update(artifact["word_frequencies"].keys())

    return vocabulary
