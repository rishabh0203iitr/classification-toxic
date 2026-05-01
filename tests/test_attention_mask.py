"""The model must ignore pad positions: changing the *content* of pad
positions must not change the model output for non-pad positions when
pooling is CLS."""
import torch

from toxic_classifier.model.classifier import ToxicClassifier


def test_padding_invariance():
    torch.manual_seed(0)
    m = ToxicClassifier(
        vocab_size=64, max_len=16, d_model=32, n_heads=2, n_layers=2, dim_ff=64, pool="cls"
    )
    m.eval()
    ids_a = torch.tensor([[2, 5, 6, 7, 0, 0, 0, 0]])
    ids_b = ids_a.clone()
    ids_b[0, 4:] = torch.tensor([13, 14, 15, 16])  # different "content" in pad slots
    pad_mask = torch.tensor([[False, False, False, False, True, True, True, True]])
    with torch.no_grad():
        out_a = m(ids_a, key_padding_mask=pad_mask)
        out_b = m(ids_b, key_padding_mask=pad_mask)
    assert torch.allclose(out_a, out_b, atol=1e-5), (out_a, out_b)
