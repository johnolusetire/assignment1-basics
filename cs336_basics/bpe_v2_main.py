import time
import regex as re
from collections import Counter, defaultdict
import heapq
import os
from itertools import pairwise
import psutil, sys
from memory_profiler import profile

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@profile
def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str], block_size: int = 64, verbose: bool = False) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    """
    In this version, i just keep track of the list locations for each pair. so i know what pairs to jump directly to when merging.

    Every list of ids is stored in a list called words.
    Basically use pair_positions to keep track of the pairs, their locations in the words list and how many times they occur.
    The pair_positions is a defaultdict of Counters, where each key is a pair and the value is a Counter mapping word_id to the number of occurrences of that pair in that word.

    During merge, I can just loop through the pair_positions for the pair being merged, and update the words list directly.
    This way, I don't have to loop through the entire words list to find the pairs.

    Downside: The heap is still polluted with stale pairs.
    """
    process = psutil.Process(os.getpid())
    if verbose:        
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # vocab: dict[int, bytes] = {idx : bytes([idx]) for idx in range(256)} #initial vocab
    vocab = {idx : bytes([idx]) for idx in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # ensure file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at {input_path}")
    
    # pre tokenization split pattern
    tok_pat = re.compile(pattern=PAT)

    # split corpus on special tokens
    special_pat = "|".join(map(re.escape, special_tokens))
    
    # pretokenize text
    BLOCK = block_size * 1024 * 1024 # chunk block size
    pair_counts = Counter()
    words = []
    pair_positions = defaultdict(Counter) # map of pair -> Counter(index for word, num_occurences)
    #pair_positions = dict()
    
    start_time = None

    if verbose:
        print(f"Starting pretokenization stage")
        print(f"Reading file in blocks of {BLOCK} bytes...")
        start_time = time.time()

    with open(input_path, "r", encoding="utf-8") as f:
        buffer = ""

        while True: 
            chunk = f.read(BLOCK) # read a chunk/block of data into memory
            
            data = buffer + chunk if chunk else buffer # prepend buffer to the current chunk
            # if the chunk is empty (reached EOF) and buffer is empty. break
            if not data:
                break

            parts = re.split(special_pat, data) # split data into parts using special_tokens as a delimiter
            
            if chunk:
                buffer = parts.pop()  # move last part to buffer if nothing was read from the file (EOF reached)
            else:
                buffer = ""

            # parts is a list of the split sections. 
            # iterate through each doc
            for doc in parts:
                # iterate through every match found using the pretokenization pattern
                for match in re.finditer(pattern=tok_pat, string=doc):
                    # prev_id = None
                    ids = [bytes([b]) for b in match.group().encode("utf-8")]  # list of bytes in the encoded token 'hello' -> [b'h',b'e',b'l',b'l',b'o']
                    words.append(ids)  # append each list of bytes (ids) to the words list. the words list holds all the splits from the regex pattern
                    
                    word_id = len(words) - 1
                    
                    for pair in pairwise(ids):
                        pair_counts[pair] += 1
                        pair_positions[pair][word_id] += 1
                        # pair_positions[pair] = pair_positions.setdefault(pair, {})
                        # pair_positions[pair][word_id] = pair_positions[pair].get(word_id, 0) + 1
                if verbose:
                    if len(words) % 10000 == 0:
                        print(f"Processed {len(words)} words so far...")

    if verbose:
        print(f"Finished pretokenization stage")
        if start_time is not None:
            print(f"Processed {len(words)} words in {time.time() - start_time:.2f} seconds")
        else:
            print(f"Processed {len(words)} words")
        
        print(f"Found {len(pair_counts)} unique pairs")
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after pretokenization: {current_memory:.2f} MB")
        print("-"*50)
        print(f"Starting heap build stage...")

        
    # use a priority queue/heap to keep track of maximum pairs instead of using the max function
    # first build the initial heap
    count_heap = []
    for pair, count in pair_counts.items():
        heapq.heappush(count_heap, HeapItem(count, pair))
    
    if verbose:
        MB = 1024 * 1024
        print("Initial Memory usage breakdown:")
        print(f"Size of pair_counts: {deep_getsizeof(pair_counts) / MB:.2f} MB")
        print(f"Size of pair_positions: {deep_getsizeof(pair_positions) / MB:.2f} MB")
        print(f"Size of words: {deep_getsizeof(words) / MB:.2f} MB")
        print(f"Size of count_heap: {deep_getsizeof(count_heap) / MB:.2f} MB")
        print("-"*50)

    mid_time = None
    if verbose:
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory at merge start: {current_memory:.2f} MB")
        print(f"Heap size: {len(count_heap)}")
        print(f"Finished heap build stage")
        print(f"Starting merge stage...")
        print("-"*50)
        start_time = time.time()
        mid_time = time.time()

    # get num of merges
    num_merges = vocab_size - 256 - len(special_tokens)
    i = 0

    while i < num_merges:       
        # # pop from the heap until the popped pair maatches the updated count
        while True:
            heap_item: HeapItem = heapq.heappop(count_heap)
            max_count, max_pair = heap_item.count, heap_item.pair
            
            if pair_counts[max_pair] == max_count :
                break
        
        # get new_id and update vocab with combination of pair and new_id
        new_id = len(vocab)
        vocab[new_id] = b"".join(max_pair)
        merges.append(max_pair)

        #print(f"merge {i+1}/{num_merges}: {max_pair} -> {vocab[new_id]} index {new_id} had {max_count} occurrences")
        
        # a dictionary/counter to hold the changes made to pair counts during the merge. 
        # This will be used to update the heap and the pair_counts counter 
        delta_count = Counter()

        # get all words that contain the max_pair
        words_with_pair: dict[int, int] = pair_positions[max_pair]  

        # if the pair still exists in the pair_positions, we can merge it
        for word_id, num_occurence in words_with_pair.items():
            if num_occurence > 0:
                word = words[word_id]

                ids, delta_count, pair_positions = merge(ids=word, pair=max_pair, word_id=word_id, 
                                                    num_occurence = num_occurence, 
                                                    local_count_delta=delta_count, 
                                                    delta_pos=pair_positions)


        del pair_positions[max_pair]
        del pair_counts[max_pair]

        # update heap and pair counter with only values that changed during merge
        # pair_counts.update(delta_count)
        for pair, count_delta in delta_count.items():
            if count_delta != 0:
                curr_count = pair_counts[pair] + count_delta
                if curr_count > 0:
                    pair_counts[pair] = curr_count
                    heapq.heappush(count_heap, HeapItem(pair_counts[pair], pair)) # only push to heap if its count is greater than zero
                else:
                    del pair_counts[pair]       

        i += 1

        if verbose:
            if i % 100 == 0 and mid_time is not None:
                print(f"Completed {i}/{num_merges} merges in {time.time() - mid_time:.2f} seconds")
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Memory at merge {i}: {current_memory:.2f} MB")
                print(f"Heap size: {len(count_heap)}")
                mid_time = time.time()
                print("-"*50)

    if verbose and start_time is not None:
        print("-"*50)
        print(f"Finished merge stage")
        print(f"Completed {i}/{num_merges} merges in {time.time() - start_time:.2f} seconds")
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory at merge end: {current_memory:.2f} MB")
        print("-"*50)
        print(f"Final heap size: {len(count_heap)}")
        print(f"Words loaded: {len(words)}")
        print(f"Unique pairs: {len(pair_counts)}")
        print(f"Pair positions entries: {sum(len(counter) for counter in pair_positions.values())}")
        print("-"*50)

        MB = 1024 * 1024
        print(f"Final Memory usage breakdown:")
        print(f"Size of pair_counts: {deep_getsizeof(pair_counts) / MB:.2f} MB")
        print(f"Size of pair_positions: {deep_getsizeof(pair_positions) / MB:.2f} MB")
        print(f"Size of words: {deep_getsizeof(words) / MB:.2f} MB")
        print(f"Size of count_heap: {deep_getsizeof(count_heap) / MB:.2f} MB")
        print("-"*50)

    for special in special_tokens:
        if len(vocab) >= vocab_size:
            break
        vocab[len(vocab)] = special.encode("utf-8")

    if verbose:
        print(f"Final vocabulary size: {len(vocab)}")
        print(f"Total merges performed: {len(merges)}")

    return vocab, merges


def merge(ids: list[bytes],  
          pair: tuple[bytes, bytes],
          word_id: int,
          num_occurence: int, 
          local_count_delta: Counter[tuple[bytes, bytes]],
          delta_pos) -> tuple[list[bytes], Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], dict]]:
    
    A, B = pair[0], pair[1]
    C = A + B

    idx = merges_done = 0
    

    while idx < len(ids):
        if idx < len(ids) - 1 and ids[idx] == A and ids[idx + 1] == B:
            merges_done += 1

            if idx > 0:
                old_left_pair = (ids[idx - 1], ids[idx])
                new_left_pair = (ids[idx - 1], C)

                local_count_delta[old_left_pair] -= 1
                local_count_delta[new_left_pair] += 1

                delta_pos[old_left_pair][word_id] -= 1
                delta_pos[new_left_pair][word_id] += 1

                # delta_pos[old_left_pair][word_id] -= 1
                # delta_pos[new_left_pair] = delta_pos.setdefault(new_left_pair, {})
                # delta_pos[new_left_pair][word_id] = delta_pos[new_left_pair].get(word_id, 0) + 1
                
                
            
            ids[idx] = C
            del ids[idx + 1]

            if idx + 1 < len(ids):
                old_right_pair = (B, ids[idx + 1])
                new_right_pair = (C, ids[idx + 1])
                
                local_count_delta[old_right_pair] -= 1
                local_count_delta[new_right_pair] += 1

                delta_pos[old_right_pair][word_id] -= 1
                delta_pos[new_right_pair][word_id] += 1
                
                # delta_pos[old_right_pair][word_id] -= 1
                # delta_pos[new_right_pair] = delta_pos.setdefault(new_right_pair, {})
                # delta_pos[new_right_pair][word_id] = delta_pos[new_right_pair].get(word_id, 0) + 1
        
        idx += 1
        if merges_done >= num_occurence:
            break
    
    return ids, local_count_delta, delta_pos

# 1. Create a wrapper class to define custom sorting logic
class HeapItem:
    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        """
        Custom comparison for the min-heap.
        __lt__ means "less than". The item that is "less than" another
        will have higher priority and be popped first.
        """
        # If counts are different, the one with the HIGHER count is "less than" (higher priority)
        if self.count != other.count:
            return self.count > other.count

        # If counts are tied, the one with the lexicographically LARGER pair is "less than"
        # In Python, ('s', 't') > ('a', 't'), so this works directly.
        return self.pair > other.pair

    def __repr__(self):
        """A nice representation for printing."""
        return f"HeapItem(count={self.count}, pair={self.pair})"

# Helper function to calculate deep size of an object
def deep_getsizeof(o, ids=None):
    """
    Find the memory footprint of a Python object.
    This is a recursive function that drills down a Python object graph
    to determine the total memory usage of the object and all of its contents.
    """
    if ids is None:
        ids = set()
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = sys.getsizeof(o)
    ids.add(id(o))

    if isinstance(o, (str, bytes)):
        return r

    if isinstance(o, dict):
        r += sum(d(k, ids) + d(v, ids) for k, v in o.items())
    elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes)):
        r += sum(d(x, ids) for x in o)
    
    return r

if __name__ == "__main__":
    _, _ = train_bpe(input_path="tests/fixtures/tinystories_sample_5M.txt",
            vocab_size=1000,
            special_tokens=["<|endoftext|>"],
            verbose=False)