from collections import Counter, defaultdict
from heapq import heappop, heappush
import os
from itertools import pairwise
from cs336_basics.pretokenization import pretokenize_text

# 1. Create a wrapper class to define custom sorting logic
class HeapItem:
    def __init__(self, count, lex_pair, int_pair):
        self.count = count
        self.lex_pair = lex_pair
        self.int_pair = int_pair

    def __lt__(self, other):
        """
        Custom comparison for the min-heap. __lt__ means "less than". The item that is "less than" another will have higher priority and be popped first.
        """
        # If counts are different, the one with the HIGHER count is "less than" (higher priority)
        if self.count != other.count:
            return self.count > other.count
        # If counts are tied, the one with the lexicographically LARGER pair is "less than"
        return self.lex_pair > other.lex_pair

    def __repr__(self):
        return f"HeapItem(count={self.count}, lex_pair={self.lex_pair}, int_pair={self.int_pair})"


def make_updates(old_word, new_word, word_count, word_to_pair) -> None:
    word_count[new_word] += word_count[old_word]
    del word_count[old_word]
    for pair in pairwise(new_word):
        word_to_pair[pair].append(new_word)

def merge(ids: tuple[int,...],  
          pair: tuple[int, int],
          rank: int,
          local_count: Counter[tuple[int, int]],
          word_count: int) -> tuple[tuple[int, ...], Counter[tuple[int, int]]]:
    
    A, B = pair[0], pair[1]
    C = rank
    idx =  0
    new_word: list[int] = []  

    while idx < len(ids):
        if idx < len(ids) - 1 and ids[idx] == A and ids[idx + 1] == B:

            if idx > 0:
                old_left_pair = (new_word[-1], ids[idx])
                new_left_pair = (new_word[-1], C)

                local_count[old_left_pair] -= (1 * word_count) 
                local_count[new_left_pair] += (1 * word_count)

            local_count[pair] -= (1 * word_count)
            idx += 1
            new_word.append(C)

            if idx + 1 < len(ids):
                old_right_pair = (ids[idx], ids[idx + 1])
                new_right_pair = (C, ids[idx + 1])
                
                local_count[old_right_pair] -= (1 * word_count)
                local_count[new_right_pair] += (1 * word_count)
            
        else:
            new_word.append(ids[idx])
        
        idx += 1
    
    return tuple(new_word), local_count



def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str], 
              mode: str = "sequential", 
              verbose: bool = False) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    """
    Final version:
    - Count unique words during pretokinization stage
    - Support for multiprocessing in pretokenization
    - Use heap to keep track of max count pair
    - Keep mapping of "pair" to a list of words it appears inside
    - when merging update all three data structures
    """
    
    # vocab: dict[int, bytes] = {idx : bytes([idx]) for idx in range(256)} #initial vocab
    vocab = {idx : bytes([idx]) for idx in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # ensure file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at {input_path}")
    
    word_count, pair_counts, pair_to_word = pretokenize_text(file_path=input_path, token_pattern=PAT, special_tokens=special_tokens, mode=mode)

    # use a priority queue/heap to keep track of maximum pairs instead of using the max function
    # first build the initial heap
    count_heap = []
    for pair, count in pair_counts.items():
        lex_pair = (vocab[pair[0]], vocab[pair[1]])
        heappush(count_heap, HeapItem(count, lex_pair, pair))    

    # get num of merges
    num_merges = vocab_size - 256 - len(special_tokens)
    i = 0

    for special in special_tokens:
        if len(vocab) >= vocab_size:
            break
        vocab[len(vocab)] = special.encode("utf-8")
        
    while i < num_merges:       
        # # pop from the heap until the popped pair maatches the updated count
        while True:
            heap_item: HeapItem = heappop(count_heap)
            max_count, max_pair = heap_item.count, heap_item.int_pair
            
            if pair_counts[max_pair] == max_count :
                break
        
        # get new_id and update vocab with combination of pair and new_id
        new_rank = len(vocab)
        pair_0, pair_1 = max_pair[0], max_pair[1]
        vocab[new_rank] = vocab[pair_0] + vocab[pair_1]
        merges.append((vocab[pair_0], vocab[pair_1]))

        if verbose:
            print(f"merge {i+1}/{num_merges}: {max_pair} -> {vocab[new_rank]} index {new_rank} had {max_count} occurrences")
        
        # a dictionary/counter to hold the changes made to pair counts during the merge. 
        # This will be used to update the heap and the pair_counts counter 
        delta_count = Counter()

        # get all words that contain the max_pair
        words_with_pair = pair_to_word[max_pair]  

        for old_word in words_with_pair:
            if old_word not in word_count:
                continue
            
            count = word_count[old_word]
            new_word, delta_count = merge(ids=old_word, pair=max_pair, rank=new_rank, local_count=delta_count, word_count=count)

            make_updates(old_word, new_word, word_count, pair_to_word)  
               
        del pair_counts[max_pair]
        del pair_to_word[max_pair]

        # update heap and pair counter with only values that changed during merge
        # pair_counts.update(delta_count)
        for pair, count_delta in delta_count.items():
            if count_delta != 0:
                curr_count = pair_counts[pair] + count_delta
                if curr_count > 0:
                    pair_counts[pair] = curr_count
                    lex_pair = (vocab[pair[0]], vocab[pair[1]])
                    heappush(count_heap, HeapItem(curr_count, lex_pair, pair)) # only push to heap if its count is greater than zero
                else:
                    del pair_counts[pair]       

        i += 1


    # for special in special_tokens:
    #     if len(vocab) >= vocab_size:
    #         break
    #     vocab[len(vocab)] = special.encode("utf-8")

    return vocab, merges



# Helper function to calculate deep size of an object
# def deep_getsizeof(o, ids=None):
#     """
#     Find the memory footprint of a Python object.
#     This is a recursive function that drills down a Python object graph
#     to determine the total memory usage of the object and all of its contents.
#     """
#     if ids is None:
#         ids = set()
#     d = deep_getsizeof
#     if id(o) in ids:
#         return 0

#     r = sys.getsizeof(o)
#     ids.add(id(o))

#     if isinstance(o, (str, bytes)):
#         return r

#     if isinstance(o, dict):
#         r += sum(d(k, ids) + d(v, ids) for k, v in o.items())
#     elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes)):
#         r += sum(d(x, ids) for x in o)
    
#     return r

# if __name__ == "__main__":
#     _, _ = train_bpe(input_path="tests/fixtures/tinystories_sample_5M.txt",
#             vocab_size=1000,
#             special_tokens=["<|endoftext|>"],
#             verbose=False)