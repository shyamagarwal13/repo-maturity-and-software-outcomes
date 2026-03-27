"""Embedding generation for artifact text content.

This module provides functions to generate dense vector embeddings
for artifact text content using sentence-transformer models.

The default model is 'nomic-ai/nomic-embed-text-v1.5' which:
- Produces 768-dimensional embeddings
- Supports 8,192-token context window
- Requires a task prefix (e.g., 'clustering: ') for optimal results
- Size: ~550MB
"""

from typing import List, Dict, Any, Optional
import gc
import os
import tempfile
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Default embedding model
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_TASK_PREFIX = "clustering"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_CHUNK_OVERLAP = 256


def _detect_device() -> str:
    """Detect the best available device for inference.

    Prefers CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _is_model_cached(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """Check if a model exists in the local HuggingFace cache."""
    if cache_dir:
        hf_cache = cache_dir
    else:
        hf_home = os.environ.get(
            "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        )
        hf_cache = os.path.join(hf_home, "hub")

    model_dir = os.path.join(hf_cache, f"models--{model_name.replace('/', '--')}")
    return os.path.isdir(model_dir)


def load_embedding_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    backend: Optional[str] = None
) -> SentenceTransformer:
    """Load or download the embedding model.

    If the model is already cached locally, loads in offline mode to skip
    network checks and avoid the HF_TOKEN warning. Otherwise downloads
    normally.

    Args:
        model_name: HuggingFace model identifier
            (e.g., "nomic-ai/nomic-embed-text-v1.5")
        cache_dir: Optional path to cache directory.
            If None, uses HuggingFace default cache.
        trust_remote_code: Whether to trust remote code for model loading.
            Required True for nomic models. Default True.
        backend: Optional inference backend. Set to "onnx" for faster
            CPU inference (requires optimum[onnxruntime]).
            Default None uses PyTorch.

    Returns:
        Loaded SentenceTransformer model object.
    """
    cached = _is_model_cached(model_name, cache_dir)

    if cached:
        print(f"Loading embedding model from cache: {model_name}")
    else:
        print(f"Downloading embedding model: {model_name} (~550MB)")

    if backend:
        print(f"Using backend: {backend}")

    device = _detect_device()
    print(f"Using device: {device}")

    # Enable offline mode when cached to skip network checks / HF_TOKEN warning
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    if cached:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        kwargs = {"trust_remote_code": trust_remote_code, "device": device}
        if cache_dir is not None:
            kwargs["cache_folder"] = cache_dir
        if backend is not None:
            kwargs["backend"] = backend
        model = SentenceTransformer(model_name, **kwargs)
    finally:
        # Restore original env state
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline

    print("Model loaded successfully")
    return model


def _chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk.
        tokenizer: The model's tokenizer (has .encode() and .decode()).
        max_tokens: Maximum tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of text chunks. Single-element list if no chunking needed.
    """
    # Tokenize without special tokens for accurate content token count.
    # Suppress the long-sequence warning since we handle chunking ourselves.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Token indices sequence length")
        token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Reserve space for special tokens ([CLS], [SEP]) added by model.encode()
    effective_max = max_tokens - 2

    if len(token_ids) <= effective_max:
        return [text]

    stride = effective_max - overlap
    if stride <= 0:
        stride = max(1, effective_max)

    chunks = []
    for start in range(0, len(token_ids), stride):
        end = min(start + effective_max, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(token_ids):
            break

    return chunks


def _estimate_safe_batch_size(max_seq_len: int, base_batch_size: int = 32, memory_budget_gib: int = 2) -> int:
    """Estimate a safe batch size given the longest sequence in a batch.

    Transformer attention memory scales as O(batch_size * seq_len^2).
    This heuristic keeps total attention memory under the given budget.

    Args:
        max_seq_len: Estimated max token count for the longest text in the batch.
        base_batch_size: Upper bound on batch size.
        memory_budget_gib: Memory budget in GiB for attention estimation.

    Returns:
        Safe batch size (at least 1, at most base_batch_size).
    """
    if max_seq_len <= 0:
        return base_batch_size

    # Total memory per batch item ≈ attention + activations + FFN intermediates.
    # Attention alone: 12 heads * seq_len^2 * 4 bytes = 48 * seq_len^2.
    # Activations/FFN add roughly 3-5x on top. We use a 5x multiplier
    # to stay conservative and leave headroom for the OS and other processes.
    BYTES_PER_POSITION_PAIR = 48 * 5  # 48 (attention) * 5 (total overhead)
    MEMORY_BUDGET = memory_budget_gib * 1024 ** 3

    mem_per_item = BYTES_PER_POSITION_PAIR * max_seq_len * max_seq_len
    safe_batch = max(1, int(MEMORY_BUDGET / mem_per_item))
    return min(safe_batch, base_batch_size)


def _flush_torch_cache():
    """Release PyTorch's internal memory cache back to the OS."""
    try:
        import torch
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _batch_encode(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    show_progress_bar: bool = True,
    labels: Optional[List[str]] = None,
    memory_budget_gib: int = 2,
) -> np.ndarray:
    """Encode texts with length-adaptive batching to prevent OOM.

    Sorts texts by estimated token count so similar-length texts are grouped,
    then encodes in batches whose size is adjusted down for longer sequences
    to keep attention memory bounded.

    Intermediate embeddings are written to a memory-mapped temp file so that
    only one batch's worth of tensors lives in RAM at a time.  The OS manages
    paging the mmap, which keeps file-cache pressure low and avoids the 3×
    peak-memory spike that came from accumulating a list + concatenating +
    un-sorting in RAM.

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of text strings to encode.
        batch_size: Maximum batch size (used for short texts). Default 32.
        show_progress_bar: Whether to show a tqdm progress bar. Default True.
        labels: Optional list of labels (e.g., file paths) for progress display.

    Returns:
        2D numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        dim = model.get_sentence_embedding_dimension() or DEFAULT_EMBEDDING_DIM
        return np.array([]).reshape(0, dim)

    dim = model.get_sentence_embedding_dimension() or DEFAULT_EMBEDDING_DIM

    # Estimate token counts (~4 chars per token)
    CHARS_PER_TOKEN = 4
    est_tokens = [max(len(t) // CHARS_PER_TOKEN, 1) for t in texts]

    # Sort by estimated length for efficient grouping
    sorted_order = sorted(range(len(texts)), key=lambda i: est_tokens[i])
    sorted_texts = [texts[i] for i in sorted_order]
    sorted_tokens = [est_tokens[i] for i in sorted_order]
    sorted_labels = [labels[i] for i in sorted_order] if labels else None

    # Create a memory-mapped temp file to hold embeddings on disk
    # instead of accumulating them in RAM.
    MAX_CHUNK_CAP = 64
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mmap')
    os.close(tmp_fd)

    try:
        sorted_emb = np.memmap(
            tmp_path, dtype=np.float32, mode='w+',
            shape=(len(texts), dim)
        )

        pos = 0
        chunk_count = 0
        pbar = tqdm(
            total=len(texts), desc="Encoding", unit="file",
            disable=not show_progress_bar
        )

        while pos < len(sorted_texts):
            # Keep each model.encode() call small so gc + torch cache
            # flush run frequently.  Texts are sorted by length ascending,
            # so sorted_tokens[pos] is the shortest in the upcoming chunk.
            # Scale inversely: short texts → 64, long texts → 4.
            est_here = sorted_tokens[pos]
            adaptive_chunk = max(4, 16384 // max(est_here, 1))
            adaptive_chunk = min(adaptive_chunk, MAX_CHUNK_CAP)
            chunk_end = min(pos + adaptive_chunk, len(sorted_texts))
            chunk_texts = sorted_texts[pos:chunk_end]
            max_seq = sorted_tokens[chunk_end - 1]
            safe_bs = _estimate_safe_batch_size(max_seq, batch_size, memory_budget_gib)

            if sorted_labels:
                pbar.set_postfix_str(sorted_labels[pos], refresh=False)

            embs = model.encode(
                chunk_texts, batch_size=safe_bs, show_progress_bar=False
            )
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)

            # Write directly to mmap, then release the tensor
            sorted_emb[pos:chunk_end] = embs
            del embs

            pbar.update(len(chunk_texts))
            pos = chunk_end
            chunk_count += 1

            # Flush mmap every 2 chunks to release embedding memory
            if chunk_count % 2 == 0:
                sorted_emb.flush()

            # Full GC + torch cache flush every 10 chunks to avoid
            # progressive slowdown from scanning the growing heap
            if chunk_count % 10 == 0:
                gc.collect()
                _flush_torch_cache()

        pbar.close()

        # Final flush to ensure all embeddings are written
        sorted_emb.flush()

        # Un-sort to original order — only one full-size array in RAM
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for sorted_pos, orig_idx in enumerate(sorted_order):
            result[orig_idx] = sorted_emb[sorted_pos]

        return result
    finally:
        # Clean up: delete the memmap object, then remove the temp file
        try:
            del sorted_emb
        except Exception:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _embed_long_text(
    model: SentenceTransformer,
    text: str,
    task_prefix: Optional[str] = DEFAULT_TASK_PREFIX,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> np.ndarray:
    """Embed a text that may exceed the model's token limit.

    Prepends the task prefix, then chunks if needed. For multi-chunk texts,
    each chunk is embedded independently and the results are mean-pooled.

    Args:
        model: Loaded SentenceTransformer model.
        text: Text content to embed.
        task_prefix: Task prefix for nomic models (e.g., "clustering").
            If None, no prefix is added.
        max_tokens: Maximum tokens per chunk.
        chunk_overlap: Token overlap between chunks.

    Returns:
        768-dimensional numpy array.
    """
    # Apply task prefix
    if task_prefix is not None:
        prefixed_text = f"{task_prefix}: {text}"
    else:
        prefixed_text = text

    # Chunk the prefixed text
    chunks = _chunk_text(prefixed_text, model.tokenizer, max_tokens, chunk_overlap)

    if len(chunks) == 1:
        return np.array(model.encode(chunks[0]))

    # Encode each chunk and mean-pool
    chunk_embeddings = []
    for chunk in chunks:
        emb = model.encode(chunk)
        chunk_embeddings.append(np.array(emb))

    return np.mean(chunk_embeddings, axis=0)


def generate_embedding(
    text: str,
    model: SentenceTransformer,
    task_prefix: Optional[str] = DEFAULT_TASK_PREFIX,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> np.ndarray:
    """Generate embedding for a single text.

    Args:
        text: Text content to embed.
        model: Loaded SentenceTransformer model.
        task_prefix: Task prefix for nomic models. Default "clustering".
            Set to None for no prefix.
        max_tokens: Maximum tokens per chunk. Default 8192.

    Returns:
        768-dimensional numpy array (or model's output dimension).
    """
    return _embed_long_text(model, text, task_prefix=task_prefix, max_tokens=max_tokens)


def generate_embeddings_batch(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
    task_prefix: Optional[str] = DEFAULT_TASK_PREFIX,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    memory_budget_gib: int = 2,
) -> np.ndarray:
    """Generate embeddings for multiple texts using batched encoding.

    Texts within the token limit are batch-encoded together for efficiency.
    Texts exceeding the limit are chunked, and all chunks are batch-encoded
    then mean-pooled per text.

    Args:
        texts: List of text contents.
        model: Loaded SentenceTransformer model.
        batch_size: Batch size for model.encode(). Default 32.
        show_progress: Show progress bar during encoding.
        task_prefix: Task prefix for nomic models. Default "clustering".
        max_tokens: Maximum tokens per chunk. Default 8192.
        chunk_overlap: Token overlap between chunks. Default 256.

    Returns:
        2D numpy array of shape (n_texts, embedding_dim).
    """
    if not texts:
        return np.array([]).reshape(0, DEFAULT_EMBEDDING_DIM)

    # Prefix and classify texts by chunk count
    single_indices = []
    single_texts = []
    multi_indices = []
    multi_chunks_list = []

    for i, text in enumerate(texts):
        prefixed = f"{task_prefix}: {text}" if task_prefix is not None else text
        chunks = _chunk_text(prefixed, model.tokenizer, max_tokens, chunk_overlap)
        if len(chunks) == 1:
            single_indices.append(i)
            single_texts.append(chunks[0])
        else:
            multi_indices.append(i)
            multi_chunks_list.append(chunks)

    result = [None] * len(texts)

    # Batch encode single-chunk texts
    if single_texts:
        single_embs = _batch_encode(
            model, single_texts, batch_size=batch_size,
            show_progress_bar=show_progress,
            memory_budget_gib=memory_budget_gib,
        )
        for idx, emb in zip(single_indices, single_embs):
            result[idx] = np.array(emb)

    # Batch encode all chunks from multi-chunk texts, then mean-pool
    if multi_chunks_list:
        all_chunks = []
        boundaries = []
        for chunks in multi_chunks_list:
            start = len(all_chunks)
            all_chunks.extend(chunks)
            boundaries.append((start, len(all_chunks)))

        all_embs = _batch_encode(
            model, all_chunks, batch_size=batch_size,
            show_progress_bar=show_progress,
            memory_budget_gib=memory_budget_gib,
        )
        for idx, (start, end) in zip(multi_indices, boundaries):
            result[idx] = np.mean(all_embs[start:end], axis=0)

    return np.stack(result)


def add_embeddings_to_artifacts(
    artifacts: List[Dict[str, Any]],
    model: SentenceTransformer,
    model_name: str,
    task_prefix: Optional[str] = DEFAULT_TASK_PREFIX,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 32,
    memory_budget_gib: int = 2,
) -> List[Dict[str, Any]]:
    """Add embeddings to all artifacts using batched encoding.

    Texts within the token limit are batch-encoded together. Texts that
    exceed the limit are chunked, and all chunks are batch-encoded then
    mean-pooled per file.

    Args:
        artifacts: List of artifact dictionaries with 'text_content' field.
        model: Loaded SentenceTransformer model.
        model_name: Model name string for metadata.
        task_prefix: Task prefix for nomic models. Default "clustering".
        max_tokens: Maximum tokens per chunk. Default 8192.
        chunk_overlap: Token overlap between chunks. Default 256.
        batch_size: Batch size for model.encode(). Default 32.

    Returns:
        Updated artifacts with embedding fields added:
        - embedding: np.ndarray or None
        - embedding_model: str
        - embedding_dim: int
    """
    embedding_dim = model.get_sentence_embedding_dimension()

    # Initialize all artifacts with None embedding
    for artifact in artifacts:
        artifact["embedding"] = None
        artifact["embedding_model"] = model_name
        artifact["embedding_dim"] = embedding_dim

    # Filter embeddable artifacts
    embeddable = [
        a for a in artifacts
        if not a.get("is_binary", False)
        and a.get("text_content") is not None
        and a.get("text_content", "").strip() != ""
    ]

    # Log skipped artifacts
    n_skipped = len(artifacts) - len(embeddable)
    if n_skipped > 0:
        skipped_binary = [a for a in artifacts if a.get("is_binary", False)]
        skipped_no_content = [
            a for a in artifacts
            if not a.get("is_binary", False) and a.get("text_content") is None
        ]
        skipped_whitespace = [
            a for a in artifacts
            if not a.get("is_binary", False)
            and a.get("text_content") is not None
            and a.get("text_content", "").strip() == ""
        ]
        print(f"\n⏭️  Skipped {n_skipped} files:")
        if skipped_binary:
            print(f"  Binary files ({len(skipped_binary)}):")
            for a in skipped_binary:
                print(f"    - {a.get('file_path', 'unknown')}")
        if skipped_no_content:
            print(f"  No content ({len(skipped_no_content)}):")
            for a in skipped_no_content:
                print(f"    - {a.get('file_path', 'unknown')}")
        if skipped_whitespace:
            print(f"  Whitespace-only ({len(skipped_whitespace)}):")
            for a in skipped_whitespace:
                print(f"    - {a.get('file_path', 'unknown')}")

    if not embeddable:
        return artifacts

    # Prefix and classify texts by chunk count
    single_chunk_indices = []
    single_chunk_texts = []
    single_chunk_labels = []
    multi_chunk_indices = []
    multi_chunk_data = []

    for i, artifact in enumerate(embeddable):
        text = artifact["text_content"]
        prefixed = f"{task_prefix}: {text}" if task_prefix is not None else text
        chunks = _chunk_text(prefixed, model.tokenizer, max_tokens, chunk_overlap)
        if len(chunks) == 1:
            single_chunk_indices.append(i)
            single_chunk_texts.append(chunks[0])
            single_chunk_labels.append(artifact.get("file_path", f"file_{i}"))
        else:
            multi_chunk_indices.append(i)
            multi_chunk_data.append(chunks)

    print(f"\nEmbedding {len(single_chunk_texts)} single-chunk files")
    if multi_chunk_data:
        total_chunks = sum(len(c) for c in multi_chunk_data)
        print(f"Embedding {len(multi_chunk_data)} multi-chunk files ({total_chunks} chunks total)")

    # Batch encode single-chunk texts
    if single_chunk_texts:
        single_embeddings = _batch_encode(
            model, single_chunk_texts, batch_size=batch_size,
            show_progress_bar=True, labels=single_chunk_labels,
            memory_budget_gib=memory_budget_gib,
        )
        for idx, emb in zip(single_chunk_indices, single_embeddings):
            embeddable[idx]["embedding"] = np.array(emb)

    # Batch encode all chunks from multi-chunk files, then mean-pool
    if multi_chunk_data:
        all_chunks = []
        all_chunk_labels = []
        chunk_boundaries = []
        for idx, chunks in zip(multi_chunk_indices, multi_chunk_data):
            start = len(all_chunks)
            file_path = embeddable[idx].get("file_path", f"file_{idx}")
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_chunk_labels.append(f"{file_path} [{j+1}/{len(chunks)}]")
            chunk_boundaries.append((start, len(all_chunks)))

        all_chunk_embeddings = _batch_encode(
            model, all_chunks, batch_size=batch_size,
            show_progress_bar=True, labels=all_chunk_labels,
            memory_budget_gib=memory_budget_gib,
        )
        for idx, (start, end) in zip(multi_chunk_indices, chunk_boundaries):
            file_embeddings = all_chunk_embeddings[start:end]
            embeddable[idx]["embedding"] = np.mean(file_embeddings, axis=0)

    return artifacts
