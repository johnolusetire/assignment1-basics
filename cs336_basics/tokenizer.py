from typing import Iterable, Iterator
import pickle
import os

from zmq import has

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.vocab_tok_to_idx = {tok:idx for idx, tok in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self._add_special_tokens()
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merge_filepath: str, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges and (optionally) a list of special tokens
        """
        if not os.path.exists(vocab_filepath):
            raise FileNotFoundError(f"Vocab file not found at path {vocab_filepath}")
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"Merge file not found at {merge_filepath}")
        
        try:
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)
            with open(merge_filepath, "rb") as f:
                merges = pickle.load(f)
        except pickle.UnpicklingError as e:
            raise RuntimeError(f"Failed to unpickle vocab or merges: {e}")
        
        if not isinstance(vocab, dict) or not isinstance(merges, list):
            raise TypeError("Vocab must be a dict and merges must be a list")
            
        return cls(vocab, merges, special_tokens)
    
    def _add_special_tokens(self):
        """ handle user-defined special tokens when encoding text (provided when constructing the tokenizer)."""
        if self.special_tokens is not None:
            new_id = max(self.vocab.keys())
            for tok in self.special_tokens:
                encoded_tok = tok.encode("utf-8", errors="replace")
                if encoded_tok not in self.vocab_tok_to_idx:
                    new_id += 1
                    self.vocab[new_id] = encoded_tok
                    self.vocab_tok_to_idx[encoded_tok] = new_id

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass