"""Tests for embedding_generator module.

This module tests the embedding generation functionality using mocked
SentenceTransformer models to avoid actual model downloads during testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call

from src.embedding_generator import (
    load_embedding_model,
    generate_embedding,
    generate_embeddings_batch,
    add_embeddings_to_artifacts,
    _chunk_text,
    _embed_long_text,
    _estimate_safe_batch_size,
    _batch_encode,
    _detect_device,
    DEFAULT_TASK_PREFIX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP,
)


class TestDetectDevice:
    """Tests for _detect_device function."""

    def test_detect_cuda(self):
        """CUDA is preferred when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict('sys.modules', {'torch': mock_torch}):
            assert _detect_device() == "cuda"

    def test_detect_mps(self):
        """MPS is used when CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict('sys.modules', {'torch': mock_torch}):
            assert _detect_device() == "mps"

    def test_detect_cpu_fallback(self):
        """Falls back to CPU when no GPU is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict('sys.modules', {'torch': mock_torch}):
            assert _detect_device() == "cpu"


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model function."""

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_default_cache(self, mock_st, mock_device):
        """Load model with default HuggingFace cache and trust_remote_code=True."""
        mock_st.return_value = Mock()

        model = load_embedding_model("test-model", cache_dir=None)

        mock_st.assert_called_once_with(
            "test-model", trust_remote_code=True, device="cpu"
        )
        assert model is not None

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_custom_cache(self, mock_st, mock_device):
        """Load model from custom cache directory with trust_remote_code=True."""
        mock_st.return_value = Mock()

        model = load_embedding_model("test-model", cache_dir="/custom/path")

        mock_st.assert_called_once_with(
            "test-model", cache_folder="/custom/path",
            trust_remote_code=True, device="cpu"
        )

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_returns_model(self, mock_st, mock_device):
        """Function returns the loaded model."""
        mock_model = Mock()
        mock_st.return_value = mock_model

        result = load_embedding_model("test-model")

        assert result == mock_model

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_trust_remote_code_false(self, mock_st, mock_device):
        """Explicitly pass trust_remote_code=False."""
        mock_st.return_value = Mock()

        load_embedding_model("test-model", trust_remote_code=False)

        mock_st.assert_called_once_with(
            "test-model", trust_remote_code=False, device="cpu"
        )

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_cache_and_trust_remote_code(self, mock_st, mock_device):
        """Cache dir and trust_remote_code=False combined."""
        mock_st.return_value = Mock()

        load_embedding_model(
            "test-model", cache_dir="/my/cache", trust_remote_code=False
        )

        mock_st.assert_called_once_with(
            "test-model", cache_folder="/my/cache",
            trust_remote_code=False, device="cpu"
        )

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_onnx_backend(self, mock_st, mock_device):
        """ONNX backend is passed to SentenceTransformer."""
        mock_st.return_value = Mock()

        load_embedding_model("test-model", backend="onnx")

        mock_st.assert_called_once_with(
            "test-model", trust_remote_code=True,
            device="cpu", backend="onnx"
        )

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_onnx_backend_with_cache(self, mock_st, mock_device):
        """ONNX backend combined with custom cache directory."""
        mock_st.return_value = Mock()

        load_embedding_model("test-model", cache_dir="/my/cache", backend="onnx")

        mock_st.assert_called_once_with(
            "test-model", trust_remote_code=True,
            cache_folder="/my/cache", device="cpu", backend="onnx"
        )

    @patch('src.embedding_generator._detect_device', return_value='cpu')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_no_backend_default(self, mock_st, mock_device):
        """Default backend=None does not pass backend kwarg."""
        mock_st.return_value = Mock()

        load_embedding_model("test-model")

        # Ensure 'backend' is NOT in the call kwargs
        call_kwargs = mock_st.call_args[1]
        assert "backend" not in call_kwargs

    @patch('src.embedding_generator._detect_device', return_value='mps')
    @patch('src.embedding_generator.SentenceTransformer')
    def test_load_model_uses_detected_mps(self, mock_st, mock_device):
        """MPS device is passed when detected on Apple Silicon."""
        mock_st.return_value = Mock()

        load_embedding_model("test-model")

        mock_st.assert_called_once_with(
            "test-model", trust_remote_code=True, device="mps"
        )


class TestGenerateEmbedding:
    """Tests for generate_embedding function."""

    def test_generate_embedding_shape(self):
        """Embedding has correct shape."""
        mock_model = Mock()
        mock_model.encode.return_value = np.zeros(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        embedding = generate_embedding("test text", mock_model)

        assert embedding.shape == (768,)

    def test_generate_embedding_calls_encode(self):
        """Function calls model.encode with prefixed text."""
        mock_model = Mock()
        mock_model.encode.return_value = np.zeros(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        generate_embedding("sample text", mock_model)

        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        # Default prefix "clustering: " should be prepended
        assert "clustering: sample text" in str(call_args)

    def test_generate_embedding_empty_text(self):
        """Handle empty text gracefully."""
        mock_model = Mock()
        mock_model.encode.return_value = np.zeros(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(5))

        embedding = generate_embedding("", mock_model)

        assert embedding.shape == (768,)

    def test_generate_embedding_returns_numpy(self):
        """Returns numpy array."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        embedding = generate_embedding("text", mock_model)

        assert isinstance(embedding, np.ndarray)

    def test_generate_embedding_default_prefix(self):
        """Default task prefix 'clustering' is prepended."""
        mock_model = Mock()
        mock_model.encode.return_value = np.zeros(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        generate_embedding("hello world", mock_model)

        encoded_text = mock_model.encode.call_args[0][0]
        assert encoded_text == "clustering: hello world"

    def test_generate_embedding_custom_prefix(self):
        """Custom task prefix is prepended."""
        mock_model = Mock()
        mock_model.encode.return_value = np.zeros(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        generate_embedding("hello world", mock_model, task_prefix="search_query")

        encoded_text = mock_model.encode.call_args[0][0]
        assert encoded_text == "search_query: hello world"

    def test_generate_embedding_none_prefix(self):
        """None prefix means no prefix is added."""
        mock_model = Mock()
        mock_model.encode.return_value = np.zeros(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        generate_embedding("hello world", mock_model, task_prefix=None)

        encoded_text = mock_model.encode.call_args[0][0]
        assert encoded_text == "hello world"


class TestGenerateEmbeddingsBatch:
    """Tests for generate_embeddings_batch function."""

    @staticmethod
    def _batch_encode(texts, **kwargs):
        """Mock encode that handles both string and list inputs."""
        if isinstance(texts, str):
            return np.zeros(768)
        return np.zeros((len(texts), 768))

    def test_batch_embedding_shape(self):
        """Batch embeddings have correct shape."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        texts = ["text1", "text2", "text3"]
        embeddings = generate_embeddings_batch(texts, mock_model)

        assert embeddings.shape == (3, 768)

    def test_batch_embedding_single_text(self):
        """Single text in batch works."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        embeddings = generate_embeddings_batch(["single"], mock_model)

        assert embeddings.shape == (1, 768)

    def test_batch_embedding_empty_list(self):
        """Empty list returns empty array."""
        mock_model = Mock()

        embeddings = generate_embeddings_batch([], mock_model)

        assert embeddings.shape[0] == 0

    def test_batch_embedding_passes_batch_size(self):
        """Batch size parameter limits the maximum batch size."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        generate_embeddings_batch(["a", "b"], mock_model, batch_size=16)

        # batch_size acts as upper bound; actual size may be smaller
        for c in mock_model.encode.call_args_list:
            assert c[1].get("batch_size", 1) <= 16

    def test_batch_embedding_progress_bar_option(self):
        """Progress bar option is accepted without error."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        # Should not raise
        generate_embeddings_batch(["a", "b"], mock_model, show_progress=False)

    def test_batch_embedding_default_prefix(self):
        """Default task prefix is applied to each text in batch."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        generate_embeddings_batch(["hello", "world"], mock_model)

        # Collect all texts passed to encode across all batch calls
        all_encoded = []
        for c in mock_model.encode.call_args_list:
            all_encoded.extend(c[0][0])
        assert "clustering: hello" in all_encoded
        assert "clustering: world" in all_encoded

    def test_batch_embedding_none_prefix(self):
        """None prefix means no prefix added."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        generate_embeddings_batch(["hello"], mock_model, task_prefix=None)

        all_encoded = []
        for c in mock_model.encode.call_args_list:
            all_encoded.extend(c[0][0])
        assert "hello" in all_encoded


class TestAddEmbeddingsToArtifacts:
    """Tests for add_embeddings_to_artifacts function."""

    @staticmethod
    def _batch_encode(texts, **kwargs):
        """Mock encode that handles both string and list inputs."""
        if isinstance(texts, str):
            return np.zeros(768)
        return np.zeros((len(texts), 768))

    def test_add_embeddings_basic(self):
        """Add embeddings to artifacts."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        artifacts = [
            {"text_content": "test 1", "is_binary": False},
            {"text_content": "test 2", "is_binary": False}
        ]

        result = add_embeddings_to_artifacts(artifacts, mock_model, "test-model")

        assert result[0]["embedding"].shape == (768,)
        assert result[0]["embedding_model"] == "test-model"
        assert result[0]["embedding_dim"] == 768

    def test_skip_binary_files(self):
        """Binary files get None embedding."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        artifacts = [
            {"text_content": "test", "is_binary": False},
            {"text_content": None, "is_binary": True}
        ]

        result = add_embeddings_to_artifacts(artifacts, mock_model, "test-model")

        assert result[0]["embedding"] is not None
        assert result[1]["embedding"] is None

    def test_skip_empty_content(self):
        """Files without text_content get None embedding."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        artifacts = [
            {"text_content": "test", "is_binary": False},
            {"text_content": None, "is_binary": False},
            {"text_content": "", "is_binary": False}
        ]

        result = add_embeddings_to_artifacts(artifacts, mock_model, "test-model")

        assert result[0]["embedding"] is not None
        assert result[1]["embedding"] is None
        assert result[2]["embedding"] is None  # Empty string skipped

    def test_skip_whitespace_only_content(self):
        """Files with only whitespace get None embedding."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        artifacts = [
            {"text_content": "   ", "is_binary": False},
            {"text_content": "\n\n\n", "is_binary": False},
            {"text_content": "\t  \n", "is_binary": False},
        ]

        result = add_embeddings_to_artifacts(artifacts, mock_model, "test-model")

        assert result[0]["embedding"] is None
        assert result[1]["embedding"] is None
        assert result[2]["embedding"] is None
        # encode should not be called for whitespace-only files
        mock_model.encode.assert_not_called()

    def test_model_name_stored(self):
        """Model name is stored in artifact."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        artifacts = [{"text_content": "test", "is_binary": False}]

        result = add_embeddings_to_artifacts(
            artifacts, mock_model, "nomic-ai/nomic-embed-text-v1.5"
        )

        assert result[0]["embedding_model"] == "nomic-ai/nomic-embed-text-v1.5"

    def test_embedding_dimension_stored(self):
        """Embedding dimension is stored in artifact."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        artifacts = [{"text_content": "test", "is_binary": False}]

        result = add_embeddings_to_artifacts(artifacts, mock_model, "test-model")

        assert result[0]["embedding_dim"] == 768

    def test_all_binary_files(self):
        """All binary files handled gracefully."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        artifacts = [
            {"text_content": None, "is_binary": True},
            {"text_content": None, "is_binary": True}
        ]

        result = add_embeddings_to_artifacts(artifacts, mock_model, "test-model")

        assert all(a["embedding"] is None for a in result)
        # encode should not be called for binary files
        mock_model.encode.assert_not_called()

    def test_add_embeddings_prefixed_texts(self):
        """Verify task prefix is applied when embedding artifacts."""
        mock_model = Mock()
        mock_model.encode.side_effect = self._batch_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        artifacts = [
            {"text_content": "some code", "is_binary": False},
        ]

        add_embeddings_to_artifacts(artifacts, mock_model, "test-model")

        # Collect all texts passed to encode across batch calls
        all_encoded = []
        for c in mock_model.encode.call_args_list:
            all_encoded.extend(c[0][0])
        assert all(t.startswith("clustering: ") for t in all_encoded)


class TestChunkText:
    """Tests for _chunk_text helper function."""

    def test_chunk_text_short(self):
        """Text under token limit returns single chunk (no splitting)."""
        mock_tokenizer = Mock()
        # 100 tokens — well under max_tokens=8192
        mock_tokenizer.encode.return_value = list(range(100))
        mock_tokenizer.decode.return_value = "short text"

        chunks = _chunk_text("short text", mock_tokenizer, max_tokens=8192, overlap=256)

        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_chunk_text_long(self):
        """Text over token limit returns multiple overlapping chunks."""
        mock_tokenizer = Mock()
        # 20 tokens, max_tokens=10 (effective_max=8 after reserving 2 for special tokens)
        # overlap=2 → stride=6
        # Chunks: [0:8], [6:14], [12:20]
        token_ids = list(range(20))
        mock_tokenizer.encode.return_value = token_ids
        mock_tokenizer.decode.side_effect = lambda ids, **kwargs: f"chunk_{ids[0]}_{ids[-1]}"

        chunks = _chunk_text("long text", mock_tokenizer, max_tokens=10, overlap=2)

        assert len(chunks) == 3
        # Verify overlapping windows (effective_max=8)
        mock_tokenizer.decode.assert_any_call(list(range(0, 8)), skip_special_tokens=True)
        mock_tokenizer.decode.assert_any_call(list(range(6, 14)), skip_special_tokens=True)
        mock_tokenizer.decode.assert_any_call(list(range(12, 20)), skip_special_tokens=True)

    def test_chunk_text_exact_boundary(self):
        """Text exactly at effective token limit returns single chunk."""
        mock_tokenizer = Mock()
        # effective_max = 8192 - 2 = 8190, so 8190 tokens fits in one chunk
        mock_tokenizer.encode.return_value = list(range(8190))
        mock_tokenizer.decode.return_value = "exact text"

        chunks = _chunk_text("exact text", mock_tokenizer, max_tokens=8192, overlap=256)

        assert len(chunks) == 1


class TestEmbedLongText:
    """Tests for _embed_long_text helper function."""

    def test_embed_long_text_single_chunk(self):
        """Short text goes through model.encode directly (single chunk)."""
        mock_model = Mock()
        mock_model.encode.return_value = np.ones(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        result = _embed_long_text(mock_model, "short text", task_prefix="clustering")

        assert result.shape == (768,)
        mock_model.encode.assert_called_once()
        # Verify the prefixed text was passed
        assert mock_model.encode.call_args[0][0] == "clustering: short text"

    def test_embed_long_text_multiple_chunks(self):
        """Long text is chunked, each chunk encoded, result is mean of embeddings."""
        mock_model = Mock()
        mock_model.tokenizer = Mock()
        # 20 tokens, max_tokens=10 (effective_max=8), overlap=2 → 3 chunks
        mock_model.tokenizer.encode.return_value = list(range(20))
        mock_model.tokenizer.decode.side_effect = lambda ids, **kwargs: f"chunk_{ids[0]}"

        # Return different embeddings for each chunk
        chunk_embeddings = [
            np.array([1.0] * 768),
            np.array([2.0] * 768),
            np.array([3.0] * 768),
        ]
        mock_model.encode.side_effect = chunk_embeddings

        result = _embed_long_text(
            mock_model, "long text", task_prefix="clustering",
            max_tokens=10, chunk_overlap=2
        )

        assert result.shape == (768,)
        # Mean of [1, 2, 3] = 2.0
        np.testing.assert_allclose(result, np.array([2.0] * 768))
        assert mock_model.encode.call_count == 3

    def test_embed_long_text_none_prefix(self):
        """None prefix means no prefix prepended."""
        mock_model = Mock()
        mock_model.encode.return_value = np.ones(768)
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        _embed_long_text(mock_model, "hello", task_prefix=None)

        assert mock_model.encode.call_args[0][0] == "hello"


class TestChunkingEndToEnd:
    """End-to-end tests for chunking through the public API."""

    def test_generate_embedding_with_chunking(self):
        """Long text produces a valid 768-dim vector via generate_embedding."""
        mock_model = Mock()
        mock_model.tokenizer = Mock()
        # Simulate a long document (20000 tokens)
        mock_model.tokenizer.encode.return_value = list(range(20000))
        mock_model.tokenizer.decode.side_effect = lambda ids, **kwargs: "decoded chunk"

        # Each chunk returns a 768-dim vector
        mock_model.encode.return_value = np.random.randn(768)

        embedding = generate_embedding("very long text " * 5000, mock_model)

        assert embedding.shape == (768,)
        assert isinstance(embedding, np.ndarray)
        # Multiple chunks should have been encoded
        assert mock_model.encode.call_count > 1

    def test_generate_embeddings_batch_mixed_lengths(self):
        """Batch with both short and long texts produces correct results."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()

        def mock_tokenizer_encode(text, **kwargs):
            # First text is short (100 tokens), second is long (20000 tokens)
            if "short" in text:
                return list(range(100))
            else:
                return list(range(20000))

        mock_model.tokenizer.encode.side_effect = mock_tokenizer_encode
        mock_model.tokenizer.decode.side_effect = lambda ids, **kwargs: "decoded"

        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.ones(768)
            return np.ones((len(texts), 768))

        mock_model.encode.side_effect = mock_encode

        embeddings = generate_embeddings_batch(
            ["short text", "very long text " * 5000], mock_model
        )

        assert embeddings.shape == (2, 768)


class TestIntegration:
    """Integration-style tests (still mocked but testing flow)."""

    def test_full_workflow(self):
        """Test complete workflow from model load to artifact embedding."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode.return_value = list(range(100))

        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.random.randn(768)
            return np.random.randn(len(texts), 768)

        mock_model.encode.side_effect = mock_encode

        artifacts = [
            {"text_content": "First document about code", "is_binary": False},
            {"text_content": "Second document about testing", "is_binary": False},
            {"text_content": None, "is_binary": True}
        ]

        result = add_embeddings_to_artifacts(
            artifacts, mock_model, "nomic-ai/nomic-embed-text-v1.5"
        )

        # First two have embeddings
        assert result[0]["embedding"] is not None
        assert result[1]["embedding"] is not None

        # Third (binary) has None
        assert result[2]["embedding"] is None

        # All have metadata
        for i in range(2):
            assert result[i]["embedding_model"] == "nomic-ai/nomic-embed-text-v1.5"
            assert result[i]["embedding_dim"] == 768
