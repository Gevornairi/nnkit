"""Check KV-cached decoding and batched sampling.

python -m tests.cache_check

Decoding a prompt and then one token at a time through a KVCache must give the
same attention outputs as one full causal forward, for every attention variant
whose computation depends on position. Exits non-zero if any check fails.
"""

import numpy as np

from nnkit.base import (
    BACKEND,
    KVCache,
    asarray,
    causal_mask,
    normal,
    seed,
    to_numpy,
    xp,
)
from nnkit.layers import MultiHeadAttention, RoPE
from nnkit.training import sample_batch

D_MODEL, T, PROMPT = 16, 9, 4
# MLX GPU matmuls are ~1e-3 relative and round differently per shape; a position bug gives O(1)
TOLERANCE = 1e-2 if BACKEND == "mlx" else 1e-10


def check_cached_attention(n_heads: int = 4, use_rope: bool = True, **kwargs) -> float:
    """Largest gap between cached and full-forward outputs, relative to the largest output."""
    seed(0)
    d_qk = D_MODEL // n_heads // (2 if kwargs.get("diff_attn") else 1)
    rope = RoPE(d_qk, max_len=16) if use_rope else None
    attn = MultiHeadAttention(D_MODEL, n_heads, 0.0, rope, **kwargs)
    attn.eval()
    x = normal(1.0, (2, T, D_MODEL))
    full = to_numpy(attn.forward(x, x, causal_mask(T)))

    cache, outs = KVCache(), []
    for start, end in [(0, PROMPT)] + [(t, t + 1) for t in range(PROMPT, T)]:
        chunk = x[:, start:end]
        outs.append(
            to_numpy(
                attn.forward(chunk, chunk, causal_mask(T)[:, :, start:end, :end], cache)
            )
        )
    assert cache.length == T
    try:
        attn.backward(xp.ones((2, 1, D_MODEL)))
        raise AssertionError("backward after a cached forward should fail")
    except RuntimeError:
        pass

    return float(np.abs(np.concatenate(outs, axis=1) - full).max() / np.abs(full).max())


def check_sample_batch() -> None:
    """A fake model that always predicts ``last token + 1``: greedy output, EOS and PAD handling."""
    vocab, eos, pad = 8, 6, 0
    fed = []

    def step(ids):
        fed.append(to_numpy(ids))
        nxt = (to_numpy(ids)[:, -1] + 1) % vocab
        return asarray(np.eye(vocab)[nxt][:, None, :] * 10.0)

    prompts = np.array([[1, 2], [3, 4]])
    out = sample_batch(step, prompts, max_new=5, eos_id=eos, pad_id=pad, temperature=0)
    # row 0 counts 3, 4, 5, 6=EOS then pads; row 1 hits EOS at once
    assert out.tolist() == [[3, 4, 5, 6, pad], [5, 6, pad, pad, pad]], out.tolist()
    assert fed[0].shape == (2, 2) and all(f.shape == (2, 1) for f in fed[1:])
    assert len(fed) == 4, "should stop once every row has emitted EOS"

    rng = np.random.default_rng(0)
    sampled = sample_batch(
        step, np.zeros((64, 1), int), 1, eos, pad, temperature=1.0, rng=rng
    )
    assert (sampled[:, 0] == 1).mean() > 0.9, (
        "sharp logits should almost always give the argmax"
    )


def main() -> int:
    ok = True
    variants = [
        ("rope", {}),
        ("rope gqa qk_norm", {"n_kv_heads": 2, "qk_norm": True}),
        ("rope qk_norm_gain", {"qk_norm": True, "qk_norm_gain": True}),
        ("rope diff_attn gate", {"diff_attn": True, "gate": "elementwise"}),
        (
            "nope temp scaling",
            {"use_rope": False, "attn_scale_alpha": 0.5, "attn_scale_floor": 2},
        ),
    ]
    for label, kwargs in variants:
        err = check_cached_attention(**kwargs)
        passed = err < TOLERANCE
        ok &= passed
        status = "ok" if passed else "FAIL"
        print(
            f"[nnkit/{BACKEND}] cached attention {label}: max err {err:.2e}  {status}"
        )
    try:
        check_sample_batch()
        print(f"[nnkit/{BACKEND}] sample_batch: ok")
    except AssertionError as e:
        ok = False
        print(f"[nnkit/{BACKEND}] sample_batch: FAIL {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
