from ast import pattern
import os
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
        initial_position = bound_pointer
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
                    word_count: dict[tuple, int] | None = None,
                    pair_counts: dict[tuple, int] | None = None):
    
    """Processes a string chunk and returns tokenization results."""

    if word_count is None or pair_counts is None:
        # Initialize if not provided
        word_count = defaultdict(int)
        pair_counts = Counter()

    parts = re.split(split_pattern, chunk) # split data into parts using special_tokens as a delimiter
       
    # parts is a list of the split sections. iterate through each doc
    for doc in parts:
        if not doc:
            continue
        for match in token_pattern.finditer(string=doc):
            word = tuple(match.group().encode("utf-8", errors="ignore"))  # get the matched token
            word_count[word] += 1  # count the number of times a token appears in the corpus

            
    for word, count in word_count.items():
        for pair in pairwise(word):
            pair_counts[pair] += count 
    
    return word_count, pair_counts

def _pretokenize_worker(file_path: str, start: int, size: int, split_pattern, token_pattern):
    """Worker for multiprocessing: opens file, reads chunk, and processes it."""
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(size).decode(encoding="utf-8", errors="ignore")
    return _process_chunk(chunk, split_pattern, token_pattern)


def pretokenize_text(file_path: str,
                     token_pattern: str,
                     special_tokens: list[str] = ["<|endoftext|>"],
                     mode: str = "sequential",
                     num_processes: int = 8) -> tuple[list[int], dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], dict[int, int]]]:
        
    global_word_count = defaultdict(int)
    global_pair_counts = Counter()
    
    if mode == "multi":
        num_processes = min(num_processes, len(os.sched_getaffinity(0)))  # better than .cpu_count() for cluster envs
    else:
        num_processes = 1

    # pre tokenization split pattern
    token_pat = re.compile(pattern=token_pattern)
    # split corpus on special tokens
    special_pat = "|".join(map(re.escape, special_tokens))
    
    
    with open(file_path, 'rb') as file:
        boundaries = find_chunk_boundaries(file, num_processes, special_tokens[-1].encode())
        chunk_sizes = [end - start for start, end in pairwise(boundaries)]

    if mode == "sequential":
        with open(file_path, 'rb') as file:          
            for start, chunk_size in zip(boundaries[:-1], chunk_sizes):
                file.seek(start)
                # read the chunk of data from the file
                chunk = file.read(chunk_size).decode(encoding="utf-8", errors="ignore")
                _, _, _ =   _process_chunk(chunk=chunk,
                                            split_pattern=special_pat,
                                            token_pattern=token_pat,
                                            words=word_list,
                                            pair_counts=global_pair_counts,
                                            pair_positions=global_pair_positions)

        return word_list, global_pair_counts, global_pair_positions
    
    elif mode == "multi":
        tasks_args = [(file_path, start, chunk_size, special_pat, token_pat) for start, chunk_size in zip(boundaries[:-1], chunk_sizes)]

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(_pretokenize_worker, tasks_args)
        
        start_time = time.time()
        for word_ids, local_pair_counts, local_pair_positions in results:
            cur_sz = len(word_list)
            word_list.extend(word_ids)

            global_pair_counts.update(local_pair_counts)

            for pair, count_object in local_pair_positions.items():
                for location_id, num_occurences in count_object.items():
                    global_pair_positions[pair][location_id + cur_sz] = num_occurences

        return word_list, global_pair_counts, global_pair_positions
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'sequential' or 'multi'.")


