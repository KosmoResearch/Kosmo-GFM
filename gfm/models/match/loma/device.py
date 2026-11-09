import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    amp_dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    amp_dtype = torch.float16
else:
    device = torch.device("cpu")
    amp_dtype = torch.bfloat16


def resolve_amp_dtype(precision: str = "auto") -> torch.dtype:
    """Per-model AMP policy; never mutate the process-wide default."""
    if precision == "auto":
        return amp_dtype
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported LoMa precision: {precision!r}")
