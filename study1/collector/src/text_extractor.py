"""
Text Extraction Module

This module provides utilities for extracting text content from files,
detecting binary files, handling various text encodings, and managing
file size constraints.

Functions:
    - is_binary_file: Determine if a file is binary or text
    - read_text_file: Read text file with automatic encoding detection
    - get_file_size: Get file size in bytes
    - should_skip_file: Determine if a file should be skipped
    - extract_text_from_artifacts: Extract text from multiple artifact files
"""

import os
from pathlib import Path
from typing import List, Dict, Optional


def is_binary_file(file_path: str) -> bool:
    """
    Determine if a file is binary or text.

    Analyzes the first 8192 bytes of a file to determine if it contains
    binary data. A file is considered binary if it contains null bytes
    or has a high ratio of non-text characters.

    Args:
        file_path: Absolute path to the file to check

    Returns:
        True if the file is binary, False if it's text

    Examples:
        >>> is_binary_file('/path/to/image.png')
        True
        >>> is_binary_file('/path/to/readme.txt')
        False
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(8192)

        # Empty file is considered text
        if not chunk:
            return False

        # Check for null bytes (common in binary files)
        if b'\x00' in chunk:
            return True

        # Check for high ratio of non-text characters
        # Text files typically have mostly printable ASCII + common Unicode
        non_text_chars = sum(1 for byte in chunk if byte < 7 or (byte > 13 and byte < 32) or byte == 127)
        text_ratio = 1 - (non_text_chars / len(chunk))

        # If less than 70% looks like text, consider it binary
        return text_ratio < 0.70

    except Exception:
        # If we can't read it, assume binary
        return True


def read_text_file(file_path: str, encodings: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Read text file with automatic encoding detection.

    Attempts to read a text file using multiple encodings in order.
    If all encodings fail, falls back to UTF-8 with error='ignore'.

    Args:
        file_path: Absolute path to the file to read
        encodings: Optional list of encodings to try (defaults to common encodings)

    Returns:
        Dictionary containing:
            - content: File content as string (None if failed)
            - encoding: Encoding that successfully decoded the file
            - success: True if read succeeded, False otherwise
            - error: Error message if failed, None otherwise

    Examples:
        >>> result = read_text_file('/path/to/file.txt')
        >>> if result['success']:
        ...     print(result['content'])
    """
    if encodings is None:
        encodings = ["utf-8", "utf-16", "iso-8859-1", "cp1252", "ascii"]

    # Check if file exists
    if not os.path.exists(file_path):
        return {
            "content": None,
            "encoding": None,
            "success": False,
            "error": f"File not found: {file_path}"
        }

    # Try each encoding
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return {
                "content": content,
                "encoding": encoding,
                "success": True,
                "error": None
            }
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            return {
                "content": None,
                "encoding": None,
                "success": False,
                "error": str(e)
            }

    # If all encodings fail, try with 'ignore' error handling
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return {
            "content": content,
            "encoding": "utf-8",
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "content": None,
            "encoding": None,
            "success": False,
            "error": str(e)
        }


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Absolute path to the file

    Returns:
        File size in bytes

    Raises:
        OSError: If file doesn't exist or is inaccessible
    """
    return os.path.getsize(file_path)


def should_skip_file(file_path: str, max_size_mb: int = 10) -> bool:
    """
    Determine if a file should be skipped during text extraction.

    Files are skipped if they meet any of these conditions:
    - File doesn't exist
    - File has a known binary extension (images, archives, executables, etc.)
    - File size exceeds the maximum allowed size

    Args:
        file_path: Absolute path to the file to check
        max_size_mb: Maximum file size in megabytes (default: 10MB)

    Returns:
        True if the file should be skipped, False otherwise

    Examples:
        >>> should_skip_file('/path/to/image.png')
        True
        >>> should_skip_file('/path/to/small.txt', max_size_mb=1)
        False
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return True

    # Known binary extensions
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
        '.exe', '.dll', '.so', '.dylib',
        '.bin', '.dat', '.db', '.sqlite',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        '.ttf', '.otf', '.woff', '.woff2'
    }

    # Check extension
    ext = Path(file_path).suffix.lower()
    if ext in binary_extensions:
        return True

    # Check file size
    try:
        size_bytes = get_file_size(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        if size_bytes > max_size_bytes:
            return True
    except Exception:
        return True

    return False


def _mark_as_binary(artifact: Dict, file_path: str) -> Dict:
    """
    Helper function to mark an artifact as binary.

    Args:
        artifact: Artifact dictionary to update
        file_path: Path to the file

    Returns:
        Updated artifact dictionary with binary markers
    """
    artifact['is_binary'] = True
    artifact['text_content'] = None
    artifact['file_size'] = get_file_size(file_path) if os.path.exists(file_path) else 0
    return artifact


def extract_text_from_artifacts(artifacts: List[Dict]) -> List[Dict]:
    """
    Extract text content from multiple artifact files.

    Processes a list of artifact dictionaries, extracting text content
    from each file. Binary files are identified and skipped. Text files
    are read with automatic encoding detection.

    For each artifact, adds these fields:
    - text_content: Extracted text (None for binary files)
    - is_binary: True if file is binary, False if text
    - encoding: Character encoding used (for text files)
    - file_size: File size in bytes

    Args:
        artifacts: List of artifact dictionaries, each containing either
                   'absolute_path' or 'file_path' key

    Returns:
        List of updated artifact dictionaries with text content and metadata

    Examples:
        >>> artifacts = [
        ...     {"absolute_path": "/path/to/config.txt", "file_path": "config.txt"},
        ...     {"absolute_path": "/path/to/image.png", "file_path": "image.png"}
        ... ]
        >>> result = extract_text_from_artifacts(artifacts)
        >>> result[0]['text_content']  # Contains text from config.txt
        >>> result[1]['is_binary']  # True
    """
    updated_artifacts = []

    for artifact in artifacts:
        # Get the absolute path (use 'absolute_path' key if available, else 'file_path')
        file_path = artifact.get('absolute_path', artifact.get('file_path'))

        # Create a copy of the artifact to update
        updated_artifact = artifact.copy()

        # Check if file should be skipped (too large, binary extension, missing)
        if should_skip_file(file_path):
            updated_artifact = _mark_as_binary(updated_artifact, file_path)
            updated_artifacts.append(updated_artifact)
            continue

        # Check if file is binary by content analysis
        if is_binary_file(file_path):
            updated_artifact = _mark_as_binary(updated_artifact, file_path)
            updated_artifacts.append(updated_artifact)
            continue

        # File is text - extract content
        result = read_text_file(file_path)

        if result['success']:
            updated_artifact['text_content'] = result['content']
            updated_artifact['is_binary'] = False
            updated_artifact['encoding'] = result['encoding']
            updated_artifact['file_size'] = get_file_size(file_path)
        else:
            # Failed to read - treat as binary
            updated_artifact = _mark_as_binary(updated_artifact, file_path)

        updated_artifacts.append(updated_artifact)

    return updated_artifacts
