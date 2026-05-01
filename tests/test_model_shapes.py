import torch

from toxic_classifier.model.classifier import ToxicClassifier


def test_forward_shape():
    m = ToxicClassifier(vocab_size=512, max_len=32, d_model=32, n_heads=2, n_layers=2, dim_ff=64)
    ids = torch.randint(low=3, high=512, size=(4, 16))  # avoid PAD/UNK/CLS for content
    ids[:, 0] = 2  # CLS
    out = m(ids)
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_param_count_reasonable():
    m = ToxicClassifier(vocab_size=4096, max_len=64, d_model=64, n_heads=2, n_layers=2, dim_ff=128)
    n = m.num_parameters()
    # Not a brittle exact-number check; just sanity bounds.
    assert 100_000 < n < 5_000_000


def test_pool_mean_path():
    m = ToxicClassifier(
        vocab_size=128, max_len=16, d_model=16, n_heads=2, n_layers=1, dim_ff=32, pool="mean"
    )
    ids = torch.randint(low=3, high=128, size=(2, 8))
    ids[:, 0] = 2
    pad_mask = torch.zeros_like(ids, dtype=torch.bool)
    pad_mask[0, 6:] = True  # pad last 2 of row 0
    out = m(ids, key_padding_mask=pad_mask)
    assert out.shape == (2,)
