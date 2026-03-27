import pytest
from src.file_data_collector import (
    generate_file_id,
    extract_repo_name,
    get_artifact_name,
    build_file_metadata,
    build_file_tf_matrix
)


def test_generate_file_id():
    """Generate properly formatted file IDs"""
    assert generate_file_id(0) == "file_000"
    assert generate_file_id(1) == "file_001"
    assert generate_file_id(999) == "file_999"


def test_extract_repo_name():
    """Extract repo name from various path formats"""
    assert extract_repo_name("/repos/my-repo") == "my-repo"
    assert extract_repo_name("/path/to/repos/test-project") == "test-project"
    assert extract_repo_name("https://github.com/user/repo.git") == "user/repo"


def test_get_artifact_name():
    """Extract standardized artifact name"""
    assert get_artifact_name(".cursorrules") == ".cursorrules"
    assert get_artifact_name(".claude/commands/test.md") == "commands/test.md"
    assert get_artifact_name(".cursor/rules/python.mdc") == "rules/python.mdc"


def test_build_file_metadata():
    """Build file metadata table"""
    artifacts = [
        {
            "file_path": ".cursorrules",
            "absolute_path": "/repo/.cursorrules",
            "tool_name": "cursor",
            "is_standard": True,
            "word_count": 500,
            "unique_terms": 150,
            "file_size": 2048,
            "is_binary": False
        }
    ]

    metadata = build_file_metadata(artifacts)

    assert len(metadata) == 1
    assert metadata[0]["file_id"] == "file_000"
    assert metadata[0]["tool_name"] == "cursor"
    assert metadata[0]["artifact_path"] == ".cursorrules"
    assert metadata[0]["is_standard"] == True
    assert metadata[0]["word_count"] == 500
    assert metadata[0]["unique_terms"] == 150


def test_build_file_tf_matrix():
    """Build term frequency matrix"""
    artifacts = [
        {
            "file_path": "file1.txt",
            "word_frequencies": {"code": 5, "rule": 3}
        },
        {
            "file_path": "file2.txt",
            "word_frequencies": {"code": 2, "format": 4}
        }
    ]
    vocabulary = {"code", "rule", "format"}

    tf_matrix = build_file_tf_matrix(artifacts, vocabulary)

    assert len(tf_matrix["file_ids"]) == 2
    assert set(tf_matrix["vocabulary"]) == vocabulary
    assert len(tf_matrix["matrix"]) == 2
    # First file: code=5, rule=3, format=0
    # Second file: code=2, rule=0, format=4


def test_tf_matrix_sparse():
    """TF matrix handles sparse data (many zeros)"""
    artifacts = [
        {"word_frequencies": {"code": 10}},
        {"word_frequencies": {"format": 5}},
        {"word_frequencies": {"rule": 3}}
    ]
    vocabulary = {"code", "format", "rule"}

    tf_matrix = build_file_tf_matrix(artifacts, vocabulary)

    # Each file should have mostly zeros
    for row in tf_matrix["matrix"]:
        zeros = row.count(0)
        assert zeros >= 2  # At least 2 zeros per row


def test_vocabulary_union():
    """Vocabulary includes all unique terms"""
    artifacts = [
        {"word_frequencies": {"code": 5, "rule": 3}},
        {"word_frequencies": {"format": 2, "structure": 4}},
        {"word_frequencies": {"code": 1, "design": 2}}
    ]

    from src.tokenizer import get_vocabulary
    vocab = get_vocabulary(artifacts)

    assert len(vocab) == 5
    assert {"code", "rule", "format", "structure", "design"} == vocab


def test_binary_files_excluded_from_tf():
    """Binary files have no word frequencies"""
    artifacts = [
        {
            "file_path": "text.txt",
            "is_binary": False,
            "word_frequencies": {"code": 5},
            "absolute_path": "/repo/text.txt",
            "tool_name": "cursor",
            "is_standard": True,
            "word_count": 5,
            "unique_terms": 1,
            "file_size": 100
        },
        {
            "file_path": "binary.bin",
            "is_binary": True,
            "word_frequencies": {},
            "absolute_path": "/repo/binary.bin",
            "tool_name": "cursor",
            "is_standard": False,
            "word_count": 0,
            "unique_terms": 0,
            "file_size": 200
        }
    ]

    metadata = build_file_metadata(artifacts)

    text_meta = [m for m in metadata if m["artifact_path"] == "text.txt"][0]
    binary_meta = [m for m in metadata if m["artifact_path"] == "binary.bin"][0]

    assert text_meta["word_count"] > 0
    assert binary_meta["word_count"] == 0


def test_file_metadata_complete():
    """All required fields present in metadata"""
    artifacts = [
        {
            "file_path": ".cursorrules",
            "absolute_path": "/repo/.cursorrules",
            "tool_name": "cursor",
            "is_standard": True,
            "word_count": 500,
            "unique_terms": 150,
            "file_size": 2048,
            "is_binary": False
        }
    ]

    metadata = build_file_metadata(artifacts)
    required_fields = [
        "file_id", "repo_name", "tool_name", "artifact_path",
        "artifact_name", "is_standard", "word_count", "unique_terms"
    ]

    for field in required_fields:
        assert field in metadata[0]


def test_multiple_tools():
    """Handle artifacts from multiple tools"""
    artifacts = [
        {
            "tool_name": "cursor",
            "word_frequencies": {"code": 5},
            "file_path": ".cursorrules",
            "absolute_path": "/repo/.cursorrules",
            "is_standard": True,
            "word_count": 5,
            "unique_terms": 1,
            "file_size": 100,
            "is_binary": False
        },
        {
            "tool_name": "claude-code",
            "word_frequencies": {"agent": 3},
            "file_path": ".claude/settings.json",
            "absolute_path": "/repo/.claude/settings.json",
            "is_standard": True,
            "word_count": 3,
            "unique_terms": 1,
            "file_size": 100,
            "is_binary": False
        },
        {
            "tool_name": "aider",
            "word_frequencies": {"format": 2},
            "file_path": ".aider.conf.yml",
            "absolute_path": "/repo/.aider.conf.yml",
            "is_standard": True,
            "word_count": 2,
            "unique_terms": 1,
            "file_size": 100,
            "is_binary": False
        }
    ]

    metadata = build_file_metadata(artifacts)

    tools = [m["tool_name"] for m in metadata]
    assert "cursor" in tools
    assert "claude-code" in tools
    assert "aider" in tools


def test_empty_artifacts():
    """Handle empty artifact list"""
    metadata = build_file_metadata([])
    assert len(metadata) == 0


def test_tf_matrix_column_order():
    """TF matrix maintains consistent column order"""
    artifacts = [
        {"word_frequencies": {"zebra": 1, "apple": 2, "monkey": 3}}
    ]
    vocab = {"apple", "monkey", "zebra"}

    tf_matrix = build_file_tf_matrix(artifacts, vocab)

    # Vocabulary should be sorted for consistency
    assert tf_matrix["vocabulary"] == sorted(vocab)
