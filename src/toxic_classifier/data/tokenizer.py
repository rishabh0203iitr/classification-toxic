"""BPE tokenizer trained from scratch on Jigsaw text.

We use the HuggingFace `tokenizers` library purely as a *trainer* — no
pretrained weights are loaded. The vocabulary is learned from the project's
training corpus and saved to disk as a single JSON file.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]
PAD_ID, UNK_ID, CLS_ID = 0, 1, 2


def build_tokenizer(lowercase: bool = True) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    norms = [normalizers.NFKC()]
    if lowercase:
        norms.append(normalizers.Lowercase())
    tok.normalizer = normalizers.Sequence(norms)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    texts: Iterable[str],
    save_path: str | Path,
    vocab_size: int = 30000,
    lowercase: bool = True,
    min_frequency: int = 2,
) -> Tokenizer:
    tok = build_tokenizer(lowercase=lowercase)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode(tok: Tokenizer, text: str, max_len: int) -> list[int]:
    """Encode a single string with [CLS] prepended, truncating to max_len."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    ids = tok.encode(text).ids
    out = [CLS_ID, *ids[: max_len - 1]]
    return out
