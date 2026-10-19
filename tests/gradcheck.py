"""Gradient check for nnkit layers and losses (finite differences in float64 & MLX autodiff)."""

import numpy as np

from nnkit.base import BACKEND, asarray, decoder_mask, padding_mask, seed, to_numpy, xp
from nnkit.layers import DecoderLayer, Embedding, EncoderLayer, Linear, RMSNorm, RoPE
from nnkit.losses import LabelSmoothingLoss

PAD, BOS, VOCAB_SIZE = 0, 1, 7
SRC = [[3, 4, 5, 6], [5, 3, 0, 0]]
TGT_IN = [[BOS, 6, 5, 4, 3], [BOS, 3, 5, 0, 0]]
TGT_OUT = [[6, 5, 4, 3, 2], [3, 5, 2, 0, 0]]


def check(n_heads: int = 2, n_kv_heads: int | None = None) -> float:
    """Verify hand-written backward passes for all core layers against reference gradients."""
    seed(0)
    d_model, d_ff, max_len = 8, 16, 16
    embed = Embedding(VOCAB_SIZE, d_model)
    rope = RoPE(d_model // n_heads, max_len)
    enc_layer = EncoderLayer(d_model, n_heads, d_ff, 0.0, rope, n_kv_heads)
    enc_norm = RMSNorm(d_model)
    dec_layer = DecoderLayer(d_model, n_heads, d_ff, 0.0, rope, n_kv_heads)
    dec_norm = RMSNorm(d_model)
    proj = Linear(d_model, VOCAB_SIZE)
    loss_fn = LabelSmoothingLoss(VOCAB_SIZE, PAD, 0.1)

    modules = [embed, enc_layer, enc_norm, dec_layer, dec_norm, proj]
    params = [p for m in modules for p in m.parameters()]
    src, tgt_in, tgt_out = asarray(SRC), asarray(TGT_IN), asarray(TGT_OUT)
    s_mask = padding_mask(src, PAD)
    t_mask = decoder_mask(tgt_in, PAD)

    def forward_logits():
        mem = enc_norm.forward(enc_layer.forward(embed.forward(src), s_mask))
        dec = dec_norm.forward(
            dec_layer.forward(embed.forward(tgt_in), mem, s_mask, t_mask)
        )
        return proj.forward(dec)

    def loss_of_model():
        return loss_fn(forward_logits(), tgt_out)[0]

    for m in modules:
        m.zero_grad()
    _, d_logits = loss_fn(forward_logits(), tgt_out)
    d_dec = dec_norm.backward(proj.backward(d_logits))
    d_tgt_emb, d_mem = dec_layer.backward(d_dec)
    embed.backward(d_tgt_emb, tgt_in)
    d_src_emb = enc_layer.backward(enc_norm.backward(d_mem))
    embed.backward(d_src_emb, src)
    manual = [to_numpy(p.grad) for p in params]

    if BACKEND == "mlx":

        def loss_of_values(values):
            for p, v in zip(params, values):
                p.value = v
            return loss_of_model()

        reference = [
            to_numpy(g) for g in xp.grad(loss_of_values)([p.value for p in params])
        ]
    else:
        reference = []
        for p in params:
            num = np.zeros(p.shape)
            for ix in np.ndindex(*p.shape):
                old = p.value[ix]
                p.value[ix] = old + 1e-6
                loss_plus = loss_of_model().item()
                p.value[ix] = old - 1e-6
                loss_minus = loss_of_model().item()
                p.value[ix] = old
                num[ix] = (loss_plus - loss_minus) / 2e-6
            reference.append(num)

    scale = max(np.abs(r).max() for r in reference)
    return max(float(np.abs(g - r).max()) for g, r in zip(manual, reference)) / scale


def main() -> int:
    tol = 1e-4 if BACKEND == "mlx" else 1e-7
    ok = True
    for h, g in [(2, None), (4, 2), (4, 1)]:
        err = check(h, g)
        status = "ok" if err < tol else "FAIL"
        ok &= err < tol
        print(
            f"[nnkit/{BACKEND}] heads={h} kv_heads={g or h}: "
            f"max err {err:.2e} (tol {tol:.0e})  {status}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
