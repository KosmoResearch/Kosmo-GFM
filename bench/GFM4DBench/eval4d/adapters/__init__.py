"""GFM4D adapter and the framework-independent prediction contract."""
from .base import AdapterPrediction, PredictionAdapter, align_batch_queries_with_sequence

__all__ = ["AdapterPrediction", "PredictionAdapter", "align_batch_queries_with_sequence", "get_adapter"]


def get_adapter(name: str, **kwargs):
    """Instantiate the bundled GFM4D adapter."""
    if str(name).strip().lower() != "gfm4d":
        raise ValueError(f"Unknown adapter {name!r}; available: gfm4d")
    from .gfm4d import GFM4DAdapter
    return GFM4DAdapter(**kwargs)
