"""GRPO: group-relative policy-gradient loss with a k3 KL penalty, and its gradient w.r.t. the logits."""

from ..activations.softmax import log_softmax
from ..base.backend import DTYPE, Array, xp


def group_advantages(
    rewards: Array, group_size: int, scale_std: bool = True, eps: float = 1e-4
) -> Array:
    """Each sample's reward minus its group's mean, optionally over the group's std.

    ``rewards`` (n_groups * group_size,) holds consecutive groups, one per prompt;
    returns advantages of the same shape. ``scale_std=False`` is Dr. GRPO's
    variant: dividing by the std up-weights prompts that are almost always (or
    almost never) solved. A group whose rewards are all equal gets zeros.
    """

    r = rewards.reshape(-1, group_size)
    adv = r - r.mean(axis=1, keepdims=True)
    if scale_std:
        adv = adv / (xp.std(r, axis=1, keepdims=True) + eps)

    return adv.reshape(-1).astype(DTYPE)


def token_logp(logits: Array, targets: Array) -> Array:
    """log p(target) at each position: logits (batch, T, vocab), targets (batch, T) -> (batch, T)."""
    log_probs = log_softmax(logits, axis=-1)

    return xp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]


class GRPOLoss:
    """``-A * log p(y) + beta * k3`` averaged over non-PAD target tokens.

    ``A`` is the sample's advantage (``group_advantages``), the same for every
    token of a completion. ``k3 = ratio - log(ratio) - 1`` with
    ``ratio = p_ref(y) / p(y)`` is Schulman's non-negative KL estimator against
    a frozen reference model; ``beta = 0`` drops it (DAPO), and ``ref_logp``
    is then not needed.

    ``targets`` must hold PAD at prompt positions as well as at padding, so
    only completion tokens count. The samples must come from the current
    weights (one update per batch of rollouts), so there is no importance
    ratio or clipping. Since d(-log p_y)/d(logits) = p - onehot(y), the
    gradient is cross-entropy's scaled per token by ``A + beta * (ratio - 1)``.

    Not a ``Module``: it has no parameters and returns the gradient directly.
    """

    def __init__(
        self, pad_id: int, beta: float = 0.0, max_log_ratio: float = 20.0
    ) -> None:
        if beta < 0:
            raise ValueError("beta must be >= 0")
        self.pad_id = pad_id
        self.beta = beta
        self.max_log_ratio = max_log_ratio

    def __call__(
        self,
        logits: Array,
        targets: Array,
        advantages: Array,
        ref_logp: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """logits (batch, T, vocab), targets (batch, T), advantages (batch,),
        ref_logp (batch, T) -> (loss, d_logits, mean k3 KL).

        ``loss`` and the KL are 0-d backend arrays; ``d_logits`` has the shape of ``logits``.
        """

        if self.beta and ref_logp is None:
            raise ValueError("ref_logp is required when beta > 0")
        v = logits.shape[-1]
        log_probs = log_softmax(logits, axis=-1)
        probs = xp.exp(log_probs)
        logp = xp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]

        valid = (targets != self.pad_id).astype(DTYPE)
        n_tokens = xp.maximum(valid.sum(), 1.0)
        adv = advantages[:, None]

        per_token = -adv * logp
        coef = adv
        kl = xp.zeros_like(logp)
        if self.beta:
            # clipped so a token the reference finds far likelier can't overflow exp
            log_ratio = xp.clip(
                ref_logp - logp, -self.max_log_ratio, self.max_log_ratio
            )
            ratio = xp.exp(log_ratio)
            kl = ratio - log_ratio - 1.0
            per_token = per_token + self.beta * kl
            coef = coef + self.beta * (ratio - 1.0)

        loss = (per_token * valid).sum() / n_tokens
        coef = coef * valid / n_tokens
        is_target = xp.arange(v)[None, None, :] == targets[..., None]
        d_logits = coef[..., None] * (probs - is_target.astype(DTYPE))
        mean_kl = (kl * valid).sum() / n_tokens

        return loss, d_logits.astype(DTYPE), mean_kl
