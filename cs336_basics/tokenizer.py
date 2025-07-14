import token
from typing import Iterable, Iterator
import pickle
import os
import regex as re
from itertools import islice

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.inv_vocab = {tok:idx for idx, tok in self.vocab.items()}
        self._validate_byte_coverage()
        self.merges = merges        
        self.pretok_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self._cache: dict[bytes, list[int]] = {}
        if special_tokens is not None:
            self.special_tokens = self._add_special_tokens(special_tokens)
            self.special_pattern = re.compile("|".join(map(re.escape, self.special_tokens.keys())))
        else:
            self.special_tokens = None
            self.special_pattern = None
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merge_filepath: str, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges and (optionally) a list of special tokens
        """
        vocab = cls.load_vocab(vocab_filepath)
        merges = cls.load_merges(merge_filepath)

        if not isinstance(vocab, dict) or not isinstance(merges, list):
            raise TypeError("Vocab must be a dict and merges must be a list")    
        return cls(vocab, merges, special_tokens)
    
    @classmethod
    def load_vocab(cls, vocab_filepath: str):
        if not os.path.exists(vocab_filepath):
            raise FileNotFoundError(f"Vocab file not found at path {vocab_filepath}")
        try:
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)
        except pickle.UnpicklingError as e:
            raise RuntimeError(f"Failed to unpickle vocab: {e}")
        return vocab

    @classmethod
    def load_merges(cls, merge_filepath: str):
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"Vocab file not found at path {merge_filepath}")
        try:
            with open(merge_filepath, "rb") as f:
                merges = pickle.load(f)
        except pickle.UnpicklingError as e:
            raise RuntimeError(f"Failed to unpickle provided merges file: {e}")
        return merges

    def __repr__(self) -> str:
        return (f"Tokenizer(vocab_size={len(self.vocab)}, "
                f"merges={len(self.merges)}, "
                f"special_tokens={self.special_tokens})")
    
    def _add_special_tokens(self, special_tokens: list[str]) -> dict[str, int]:
        """ handle user-defined special tokens when encoding text (provided when constructing the tokenizer)."""
        temp = {}
        new_id = max(self.vocab.keys())
        for tok in special_tokens:
            if tok:
                encoded_tok = tok.encode("utf-8", errors="replace")
                if encoded_tok not in self.inv_vocab:
                    new_id += 1
                    self.vocab[new_id] = encoded_tok
                    self.inv_vocab[encoded_tok] = new_id
                temp[tok] = self.inv_vocab[encoded_tok]
        return temp

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        special_token = None
        start = 0
        while start < len(text):
            if self.special_pattern is not None:
                special_token = self.special_pattern.search(text[start:])
            
            end = len(text) if special_token is None else special_token.span()[0] + start

            parts = self.pretok_pattern.finditer(text[start:end])

            for match in parts:
                token = match.group().encode("utf-8", errors="replace")
                if token in self.inv_vocab:
                    # check if the token is already in the vocab. case for completely merged words and special tokens
                    ids.append(self.inv_vocab[token])
                elif token in self._cache:
                    # check if that particular string is in the cache
                    ids.extend(self._cache[token])
                else:
                    # if it is a new token, then we apply merges on its bytes characters to get the token sequence
                    merged_ids = self._apply_merge(token)
                    ids.extend(merged_ids)
            
            if special_token is not None:
                ids.append(self.inv_vocab[special_token.group().encode("utf-8", errors="replace")]) # get the id for the special token
                start += special_token.span()[-1]
            else:
                start += end
        
        return ids       

    
    def _normalize_string(self, iterable: Iterable[str], max_len: int = 256) -> Iterator[str]:
        """ Yield at most 256 chars from a given line"""
        for line in iterable:
            if len(line) <= max_len:
                yield line
            else:
                yield from (line[i : i+max_len] for i in range(0, len(line), max_len))

    def encode_iterable(self, iterable: Iterable[str], max_str_length: int = 256, max_num_lines: int = 256) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into memory."""
        buffer = ""
        normalize_string = self._normalize_string(iterable, max_str_length)
        while True:
            chunk = "".join(list(islice(normalize_string, max_num_lines)))
            if chunk == "":
                break
            chunk = buffer + chunk
            last_valid_idx = self._get_chunk_boundary(chunk)
            buffer = chunk[last_valid_idx:]
            
            yield from self.encode(chunk[:last_valid_idx])
        yield from self.encode(buffer) 


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
            
        result = [self.inv_vocab[tok] for tok in toks]
        self._cache[token] = result
        return result
    
    def _validate_byte_coverage(self):
        missing_bytes = []
        for i in range(256):
            if bytes([i]) not in self.inv_vocab:
                missing_bytes.append(i)
        if missing_bytes:
            raise ValueError(f"Vocab missing byte values: {missing_bytes[:10]}...")
        
    def _get_chunk_boundary(self, chunk: str):
        """
        Returns the start of the last match found based on our pattern
        If there are special tokens, find the last index that points to the end of the last special token match.end()
        Search the remaining chunk from that index, find the start index of the last match. We encode every match except the last match.
        """
        cutoff_point = 0
        if self.special_pattern is not None:
            cutoff_point = max((match.end() for match in self.special_pattern.finditer(chunk)), default=cutoff_point)
        pretok_matches = self.pretok_pattern.finditer(chunk[cutoff_point:])
        cutoff_point = max(cutoff_point, max((match.start() for match in pretok_matches), default=cutoff_point))
        return cutoff_point