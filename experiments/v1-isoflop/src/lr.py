"""V1 multi-step LR: warmup to max, 31.6% at 80% tokens, 10% at 90%."""


def lr_at_frac(frac: float, max_lr: float, warmup_frac: float) -> float:
    frac = min(max(frac, 0.0), 1.0)
    if frac < warmup_frac and warmup_frac > 0:
        return max_lr * (frac / warmup_frac)
    if frac < 0.80:
        return max_lr
    if frac < 0.90:
        return max_lr * 0.316
    return max_lr * 0.10
