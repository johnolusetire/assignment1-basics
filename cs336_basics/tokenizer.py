from typing import Iterable, Iterator
import pickle
import os
import regex as re
from itertools import pairwise

from zmq import has

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.vocab_tok_to_idx = {tok:idx for idx, tok in self.vocab.items()}
        self._validate_byte_coverage()
        self.merges = merges
        self.special_tokens = special_tokens
        self.pattern = self._create_pattern(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self._cache: dict[bytes, list[int]] = {}
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

    def __repr__(self) -> str:
        return (f"Tokenizer(vocab_size={len(self.vocab)}, "
                f"merges={len(self.merges)}, "
                f"special_tokens={self.special_tokens})")
    
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
    
    def _create_pattern(self, pattern: str):
        special_pattern = ("|".join(map(re.escape, self.special_tokens)) if self.special_tokens else "")
        master_pattern = special_pattern + "|" + pattern if special_pattern else pattern
        return re.compile(master_pattern)

    def encode(self, text: str) -> list[int]:
        ids = []
        for match in self.pattern.finditer(text):
            token = match.group().encode("utf-8", errors="replace")
            if token in self.vocab_tok_to_idx:
                # check if the token is already in the vocab. case for completely merged words and special tokens
                ids.append(self.vocab_tok_to_idx[token])
            elif token in self._cache:
                ids.extend(self._cache[token])
            else:
                merged_ids = self._apply_merge(token) # might add a cache to store tokenized words for repeat words 
                ids.extend(merged_ids)
        
        return ids
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into memory."""
        pass

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")

    def _apply_merge(self, token: bytes) -> list[int]:
        toks = [bytes([b]) for b in token]        

        for merge in self.merges:
            i, j = 0, len(toks)
            merged_toks = []
            while (i < j):
                if i < j - 1 and toks[i] == merge[0] and toks[i + 1] == merge[1]:
                    merged_toks.append(merge[0] + merge[1])
                    i += 2
                else:
                    merged_toks.append(toks[i])
                    i += 1
            toks = merged_toks
            
        result = [self.vocab_tok_to_idx[tok] for tok in toks]
        self._cache[token] = result
        return result
    
    def _validate_byte_coverage(self):
        missing_bytes = []
        for i in range(256):
            if bytes([i]) not in self.vocab_tok_to_idx:
                missing_bytes.append(i)
        if missing_bytes:
            raise ValueError(f"Vocab missing byte values: {missing_bytes[:10]}...")