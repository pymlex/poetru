from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


def build_byte_level_bpe_tokenizer() -> Tokenizer:
    """Creates an empty ByteLevel BPE tokenizer with byte fallback and no trained merges.

    Args:
        None.

    Returns:
        Tokenizer: Configured yet untrained tokenizer instance.
    """

    tokenizer = Tokenizer(BPE(byte_level=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    return tokenizer


def train_byte_level_bpe(
    iterator,
    vocab_size: int,
    output_dir: Path,
    min_frequency: int,
) -> Tokenizer:
    """Fits ByteLevel BPE on a text iterator and writes `tokenizer.json` to disk.

    Args:
        iterator: Iterable of strings covering the corpus.
        vocab_size: Target vocabulary size including special symbols.
        output_dir: Directory that will contain `tokenizer.json`.
        min_frequency: Minimal merge frequency kept by the BPE trainer.

    Returns:
        Tokenizer: Trained tokenizer ready for encoding and decoding.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_byte_level_bpe_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[EOS]", "[UNK]"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    tokenizer.save(str(output_dir / "tokenizer.json"), pretty=False)
    return tokenizer


class ByteBPETokenizerWrapper:
    """Thin convenience wrapper over `tokenizers.Tokenizer` for integer ids."""

    def __init__(self, tok: Tokenizer) -> None:
        self._tok = tok
        self.pad_id = int(self._tok.token_to_id("[PAD]"))
        self.eos_id = int(self._tok.token_to_id("[EOS]"))
        self.unk_id = int(self._tok.token_to_id("[UNK]"))

    @property
    def vocab_size(self) -> int:
        """Trained vocabulary size including specials."""

        return int(self._tok.get_vocab_size())

    @classmethod
    def from_file(cls, path: Path) -> ByteBPETokenizerWrapper:
        """Loads a trained tokenizer from `tokenizer.json`.

        Args:
            path: Path to `tokenizer.json`.

        Returns:
            ByteBPETokenizerWrapper: Ready wrapper with special token ids resolved.
        """

        tok = Tokenizer.from_file(str(path))
        return cls(tok)

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        """Encodes text to token ids with optional EOS.

        Args:
            text: Raw document string.
            add_eos: When `True`, appends the EOS id.

        Returns:
            List of integer token ids.
        """

        enc = self._tok.encode(text)
        ids = [int(i) for i in enc.ids]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decodes token ids to a string.

        Args:
            ids: Token indices.
            skip_special: When `True`, drops special tokens in the string output.

        Returns:
            Reconstructed text.
        """

        return self._tok.decode(ids, skip_special_tokens=skip_special)
