from typing import Any

from .lstm import LSTMClassifier
from .transformer import TransformerClassifier


def build_model(name: str, vocab_size: int, cfg: dict[str, Any], max_length: int):
    """Build a classifier from the ``model`` config block. Both return logits of shape (B,)."""
    if name == "lstm":
        return LSTMClassifier(
            vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_size=cfg["lstm"]["hidden_size"],
            num_layers=cfg["lstm"]["num_layers"],
            dropout=cfg["dropout"],
        )
    if name == "transformer":
        return TransformerClassifier(
            vocab_size,
            d_model=cfg["embed_dim"],
            num_heads=cfg["transformer"]["num_heads"],
            num_layers=cfg["transformer"]["num_layers"],
            ff_dim=cfg["transformer"]["ff_dim"],
            dropout=cfg["dropout"],
            max_length=max_length,
        )
    raise ValueError(f"Unknown model {name!r}")


__all__ = ["LSTMClassifier", "TransformerClassifier", "build_model"]
