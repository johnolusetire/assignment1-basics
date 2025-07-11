from ast import pattern
import os
from numpy import size
import regex as re
from typing import BinaryIO
from collections import defaultdict, Counter
from itertools import pairwise
import multiprocessing
import time

def find_chunk_boundaries(file: BinaryIO, 
                          desired_num_chunks: int, 
                          split_special_token: bytes ) -> list[int]:
    
    """
    This function divides the whole file into different chunks
    It finds suitable starting points for each chunk based on the split special token.
    It returns a unique list of pointers to a valid start for a chunk
    """

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # reset file pointer

    chunk_size = file_size // desired_num_chunks

    # initial guesses for the chunk boundary
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] # [0, .....]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bound_pointer in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bound_pointer]
        file.seek(initial_position) # start at a boundary guess        

        while True:
            mini_chunk = file.read(mini_chunk_size)  # read a small chunk

            if mini_chunk == b"":
                chunk_boundaries[bound_pointer] = file_size
                break

            found_idx = mini_chunk.find(split_special_token)
            if found_idx != -1:
                # if the special split token was found in the minichunk. set the boundary to the idx it was found at
                chunk_boundaries[bound_pointer] = initial_position + found_idx
                break
            initial_position += mini_chunk_size # if nothing was found. add the read part to the initial position
    
    return sorted(set(chunk_boundaries)) # make sure chunk boundaries has unique pointers
 
def _process_chunk(chunk: str,
                   split_pattern: re.Pattern,
                   token_pattern: re.Pattern,
                   word_count: Counter[tuple[int, ...]] | None = None) -> Counter[tuple[int, ...]]:
    
    """Processes a string chunk and returns tokenization results."""
    if word_count is None:
        word_count = Counter()

    parts = re.split(split_pattern, chunk) # split data into parts using special_tokens as a delimiter
       
    # parts is a list of the split sections. iterate through each doc
    for doc in parts:
        if not doc:
            continue
        for match in token_pattern.finditer(string=doc):
            word = tuple(match.group().encode("utf-8", errors="ignore"))  # get the matched token
            word_count[word] += 1  # count the number of times a token appears in the corpus
    
    return word_count

def _pretokenize_worker(file_path: str, start: int, size: int, split_pattern: re.Pattern, token_pattern: re.Pattern):
    """Worker for multiprocessing: opens file, reads chunk, and processes it."""
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(size).decode(encoding="utf-8", errors="ignore")
    return _process_chunk(chunk, split_pattern, token_pattern)

def pretokenize_text(file_path: str | os.PathLike,
                     token_pattern: str,
                     special_tokens: list[str] = ["<|endoftext|>"],
                     mode: str = "sequential",
                     num_processes: int = 8) -> tuple[Counter, Counter, dict]:
        
    global_word_count = Counter()
    pair_counts = Counter()
    pair_to_word = defaultdict(list)
    
    if mode == "multi":
        num_processes = max(num_processes, len(os.sched_getaffinity(0)))  # better than .cpu_count() for cluster envs

    # pre tokenization split pattern
    token_pat = re.compile(pattern=token_pattern)
    # split corpus on special tokens
    special_pat = "|".join(map(re.escape, special_tokens))
    special_pat = re.compile(special_pat)

    with open(file_path, 'rb') as file:
        boundaries = find_chunk_boundaries(file, num_processes, "<|endoftext|>".encode("utf-8"))
        chunk_sizes = [end - start for start, end in pairwise(boundaries)]
    
    if mode == "sequential":
        with open(file_path, "rb") as file:
            for start, chunk_size in zip(boundaries[:-1], chunk_sizes):
                file.seek(start)
                chunk = file.read(chunk_size).decode(encoding="utf-8", errors="ignore")
                global_word_count = _process_chunk(chunk=chunk, split_pattern=special_pat, token_pattern=token_pat, word_count=global_word_count)
       
    elif mode == "multi":        
        tasks_args = [(file_path, start, chunk_size, special_pat, token_pat) for start, chunk_size in zip(boundaries[:-1], chunk_sizes)]

        with multiprocessing.Pool(processes=num_processes) as pool:
            dict_results = pool.starmap(_pretokenize_worker, tasks_args)
                      
        for word_counter in dict_results:
            global_word_count.update(word_counter)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'sequential' or 'multi'.")

    for word, count in global_word_count.items():
        for pair in pairwise(word):
            pair_counts[pair] += count
            pair_to_word[pair].append(word)
    
    return global_word_count, pair_counts, pair_to_word