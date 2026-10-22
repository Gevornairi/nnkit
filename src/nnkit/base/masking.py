"""Boolean attention masks: ``True`` means the query may attend to that key.

Masks are shaped to broadcast against attention scores of shape
``(batch, heads, query_len, key_len)``.
"""

from .backend import Array, xp


def padding_mask(ids: Array, pad_id: int) -> Array:
    """Hide PAD keys. ``ids`` (batch, T) -> mask (batch, 1, 1, T)."""
    return (ids != pad_id)[:, None, None, :]


def causal_mask(size: int) -> Array:
    """Hide future keys: query i sees keys 0..i. Returns (1, 1, size, size)."""
    return (xp.tril(xp.ones((size, size))) > 0)[None, None, :, :]


def chunked_causal_mask(size: int, chunk: int) -> Array:
    """Causal attention within fixed blocks of ``chunk`` tokens (Llama 4's local attention).

    Query i sees keys j with j <= i in the same block (i // chunk == j // chunk).
    With ``chunk >= size`` there is one block and this equals ``causal_mask(size)``.
    Returns (1, 1, size, size).
    """
    if chunk < 1:
        raise ValueError(f"chunk={chunk} must be >= 1")
    pos = xp.arange(size)
    same_chunk = (pos[:, None] // chunk) == (pos[None, :] // chunk)

    return (causal_mask(size)[0, 0] & same_chunk)[None, None, :, :]


def sliding_window_mask(size: int, window: int) -> Array:
    """Causal attention over the last ``window`` tokens (Mistral, Gemma 2/3, gpt-oss local layers).

    Query i sees keys j with i - window < j <= i, so every token gets the same amount of
    recent context, unlike chunked attention. With ``window >= size`` this equals
    ``causal_mask(size)``. Returns (1, 1, size, size).
    """
    if window < 1:
        raise ValueError(f"window={window} must be >= 1")
    pos = xp.arange(size)
    near = (pos[:, None] - pos[None, :]) < window

    return (causal_mask(size)[0, 0] & near)[None, None, :, :]


def document_causal_mask(doc_ids: Array) -> Array:
    """Causal attention that stays inside each document of a packed sequence.

    ``doc_ids`` (batch, T) gives each token's document number; query i sees key j if
    j <= i and doc_ids[i] == doc_ids[j]. Returns (batch, 1, T, T).
    """
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]

    return (causal_mask(doc_ids.shape[1])[0] & same_doc)[:, None, :, :]


def decoder_mask(tgt_ids: Array, pad_id: int) -> Array:
    """Decoder self-attention mask: padding and causal. ``tgt_ids`` (batch, T) -> (batch, 1, T, T)."""
    return padding_mask(tgt_ids, pad_id) & causal_mask(tgt_ids.shape[1])
