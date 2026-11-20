import pytest

torch = pytest.importorskip("torch")

from lidds_gen.data.experiment import EncodedSplit  # noqa: E402
from lidds_gen.models import build_model  # noqa: E402
from lidds_gen.training import WindowDataset, collate_windows, weighted_bce  # noqa: E402

MODEL_CFG = {
    "embed_dim": 16, "dropout": 0.0,
    "lstm": {"hidden_size": 16, "num_layers": 2},
    "transformer": {"num_layers": 1, "num_heads": 2, "ff_dim": 32},
}


def _batch():
    import numpy as np
    split = EncodedSplit(
        sequences=[np.array([2, 3, 4]), np.array([5]), np.arange(2, 12)],
        labels=[0, 1, 0], target_lengths=[50, 50, 50], scenarios=["S"] * 3,
    )
    ds = WindowDataset(split, max_length=200)
    return collate_windows([ds[i] for i in range(len(ds))])


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_forward_shape_and_padding_invariance(name):
    torch.manual_seed(0)
    model = build_model(name, vocab_size=20, cfg=MODEL_CFG, max_length=200).eval()
    tokens, lengths, y = _batch()
    logits = model(tokens, lengths)
    assert logits.shape == (3,)
    # Extra padding must not change predictions.
    padded = torch.nn.functional.pad(tokens, (0, 7), value=0)
    assert torch.allclose(model(padded, lengths), logits, atol=1e-5)
    assert weighted_bce(logits, y, 1.0, 2.0).ndim == 0
