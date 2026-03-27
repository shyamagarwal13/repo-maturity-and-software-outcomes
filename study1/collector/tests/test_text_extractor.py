import pytest
from pathlib import Path
from src.text_extractor import (
    extract_text_from_artifacts,
    is_binary_file,
    read_text_file,
    get_file_size,
    should_skip_file
)


def test_read_text_file_utf8(tmp_path):
    """Read UTF-8 encoded text file"""
    test_file = tmp_path / "test.txt"
    test_content = "Hello World! 🚀"
    test_file.write_text(test_content, encoding="utf-8")

    result = read_text_file(str(test_file))

    assert result["success"] == True
    assert result["content"] == test_content
    assert result["encoding"] == "utf-8"


def test_read_text_file_different_encoding(tmp_path):
    """Read file with non-UTF-8 encoding"""
    test_file = tmp_path / "test_latin1.txt"
    test_content = "Héllo Wörld"
    test_file.write_text(test_content, encoding="iso-8859-1")

    result = read_text_file(str(test_file))

    assert result["success"] == True
    assert result["content"] == test_content
    # Should detect encoding


def test_is_binary_file_text(tmp_path):
    """Identify text files correctly"""
    text_file = tmp_path / "text.txt"
    text_file.write_text("This is plain text")

    assert is_binary_file(str(text_file)) == False


def test_is_binary_file_binary(tmp_path):
    """Identify binary files correctly"""
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03\xFF\xFE')

    assert is_binary_file(str(binary_file)) == True


def test_extract_text_from_artifacts(tmp_path):
    """Extract text from multiple artifacts"""
    # Create test files
    text_file = tmp_path / "config.txt"
    text_file.write_text("Config content")

    binary_file = tmp_path / "cache.bin"
    binary_file.write_bytes(b'\x00\x00\x00')

    artifacts = [
        {"absolute_path": str(text_file), "file_path": "config.txt"},
        {"absolute_path": str(binary_file), "file_path": "cache.bin"}
    ]

    result = extract_text_from_artifacts(artifacts)

    assert len(result) == 2
    # Text file should have content
    text_artifact = [a for a in result if a["file_path"] == "config.txt"][0]
    assert text_artifact["text_content"] == "Config content"
    assert text_artifact["is_binary"] == False

    # Binary file should be marked
    binary_artifact = [a for a in result if a["file_path"] == "cache.bin"][0]
    assert binary_artifact["is_binary"] == True
    assert binary_artifact["text_content"] is None


def test_get_file_size(tmp_path):
    """Get correct file size"""
    test_file = tmp_path / "test.txt"
    test_content = "A" * 1000  # 1000 bytes
    test_file.write_text(test_content)

    size = get_file_size(str(test_file))
    assert size == 1000


def test_should_skip_large_file(tmp_path):
    """Skip files larger than max size"""
    large_file = tmp_path / "large.txt"
    # Create 11MB file
    large_file.write_text("A" * (11 * 1024 * 1024))

    assert should_skip_file(str(large_file), max_size_mb=10) == True


def test_should_skip_binary_extensions(tmp_path):
    """Skip known binary file extensions"""
    binary_files = [
        tmp_path / "image.png",
        tmp_path / "doc.pdf",
        tmp_path / "archive.zip"
    ]

    for bf in binary_files:
        bf.write_bytes(b'\x00')
        assert should_skip_file(str(bf)) == True


def test_should_not_skip_text_files(tmp_path):
    """Don't skip text files"""
    text_files = [
        tmp_path / "config.txt",
        tmp_path / "rules.md",
        tmp_path / "settings.json"
    ]

    for tf in text_files:
        tf.write_text("content")
        assert should_skip_file(str(tf)) == False


def test_handle_missing_file():
    """Handle missing file gracefully"""
    result = read_text_file("/nonexistent/file.txt")

    assert result["success"] == False
    assert result["error"] is not None


def test_empty_file(tmp_path):
    """Handle empty files"""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    result = read_text_file(str(empty_file))

    assert result["success"] == True
    assert result["content"] == ""


def test_special_characters(tmp_path):
    """Handle special characters and unicode"""
    special_file = tmp_path / "special.txt"
    content = "Special: @#$%^&* Unicode: 你好 Emoji: 🎉"
    special_file.write_text(content, encoding="utf-8")

    result = read_text_file(str(special_file))

    assert result["success"] == True
    assert result["content"] == content


def test_line_endings(tmp_path):
    """Handle different line endings (LF, CRLF)"""
    # Unix line endings
    unix_file = tmp_path / "unix.txt"
    unix_file.write_text("line1\nline2\nline3")

    # Windows line endings
    windows_file = tmp_path / "windows.txt"
    windows_file.write_text("line1\r\nline2\r\nline3")

    unix_result = read_text_file(str(unix_file))
    windows_result = read_text_file(str(windows_file))

    assert unix_result["success"] == True
    assert windows_result["success"] == True


def test_corrupted_encoding(tmp_path):
    """Handle files with encoding issues"""
    # Create file with mixed encodings (not valid UTF-8)
    bad_file = tmp_path / "bad_encoding.txt"
    bad_file.write_bytes(b'Valid text \xFF\xFE invalid bytes')

    result = read_text_file(str(bad_file))

    # Should still succeed with fallback encoding or 'ignore' errors
    assert result["success"] == True
    assert result["content"] is not None
