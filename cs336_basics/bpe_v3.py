import regex as re
from collections import Counter
import heapq
import os
from itertools import pairwise
from line_profiler import profile

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class TokenNode:
    def __init__(self, value, next_node=None, prev=None) -> None:
        self.val: bytes = value
        self.next_node: TokenNode | None = next_node
        self.prev: TokenNode | None = prev

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

@profile    
def process_corpus(input_path: str | os.PathLike, 
                   special_pattern: str,
                   token_pattern:  str,
                   block_size: int, 
                   pair_counter: Counter[tuple[bytes, bytes]], 
                   pair_location: dict[tuple[bytes, bytes], list[TokenNode]]) -> tuple[Counter, dict[tuple[bytes, bytes], list[TokenNode]] ]:
    
    # pre tokenization split pattern
    token_pat = re.compile(pattern=token_pattern)

    with open(input_path, "r") as file:
        buffer = ""

        while True:
            chunk = file.read(block_size)
            data = buffer + chunk if chunk else buffer

            if not data:
                break

            parts = re.split(special_pattern, data)

            buffer = parts.pop() if chunk else ""

            for doc in parts:
                for match in re.finditer(pattern=token_pat, string=doc):
                    ids = [bytes([b]) for b in match.group().encode("utf-8")]

                    prev = None
                    curr = None

                    for pair in pairwise(ids):
                        pair_counter[pair] += 1

                        if pair not in pair_location:
                            pair_location[pair] = []
                        
                        if prev is None:
                            prev = TokenNode(value=pair[0])
                        pair_location[pair].append(prev)
                        
                        curr = TokenNode(value=pair[1], prev=prev)
                        prev.next_node = curr
                        prev = curr
    
    return pair_counter, pair_location

@profile                           
def merge(pair: tuple[bytes, bytes], 
          token_node: TokenNode,
          delta_counter: Counter[tuple[bytes, bytes]],
          pair_location: dict[tuple[bytes, bytes], list[TokenNode]]):
    
    A, B = pair[0], pair[1]
    C = A + B

    temp: TokenNode | None = token_node.next_node

    if token_node.prev:
        old_left_pair = (token_node.prev.val, A)
        new_left_pair = (token_node.prev.val, C)

        delta_counter[old_left_pair] -= 1
        delta_counter[new_left_pair] += 1

        if new_left_pair not in pair_location:
            pair_location[new_left_pair] = []
        pair_location[new_left_pair].append(token_node.prev)
    
    token_node.val = C

    if temp and temp.next_node:
        old_right_pair = (B, temp.next_node.val)
        new_right_pair = (C, temp.next_node.val)

        delta_counter[old_right_pair] -= 1
        delta_counter[new_right_pair] += 1

        if new_right_pair not in pair_location:
            pair_location[new_right_pair] = []
        pair_location[new_right_pair].append(token_node)
    
    if temp:
        token_node.next_node = temp.next_node
        temp.next_node = temp.prev = None

    if token_node.next_node:
        token_node.next_node.prev = token_node

    return delta_counter, pair_location

@profile
def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # vocab: dict[int, bytes] = {idx : bytes([idx]) for idx in range(256)} #initial vocab
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # ensure file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at {input_path}")
    
    # get regex pattern to split text corpus on specila tokens
    special_pat = "|".join(map(re.escape, special_tokens))

    BLOCK = 64 * 1024 * 1024
    pair_counter = Counter()
    pair_location: dict[tuple[bytes, bytes], list[TokenNode]] = {}

    pair_counter, pair_location = process_corpus(input_path=input_path,
                                                 special_pattern=special_pat,
                                                 token_pattern= PAT, 
                                                 block_size=BLOCK,
                                                 pair_counter=pair_counter,
                                                 pair_location=pair_location)

    # use a priority queue/heap to keep track of maximum pairs instead of using the max function
    # first build the initial heap
    count_heap = []
    for pair, count in pair_counter.items():
        heapq.heappush(count_heap, HeapItem(count, pair))

    # get num of merges
    num_merges = vocab_size - 256 - len(special_tokens)
    i = 0

    while i < num_merges:       
        # # pop from the heap until the popped pair maatches the updated count
        while True:
            heap_item: HeapItem = heapq.heappop(count_heap)
            max_count, max_pair = heap_item.count, heap_item.pair
            
            if pair_counter[max_pair] == max_count :
                break
        
        # get new_id and update vocab with combination of pair and new_id
        new_id = len(vocab)
        vocab[new_id] = b"".join(max_pair)
        merges.append(max_pair)

        #print(f"merge {i+1}/{num_merges}: {max_pair} -> {vocab[new_id]} index {new_id} had {max_count} occurrences")
        
        # a dictionary/counter to hold the changes made to pair counts during the merge. 
        # This will be used to update the heap and the pair_counts counter 
        delta_counter = Counter()

        words_with_pair = pair_location[max_pair]
        
        for word in words_with_pair:

            if word and word.next_node and (word.val, word.next_node.val) == max_pair:
                delta_counter, pair_location = merge(pair=max_pair, token_node=word,
                                                     delta_counter=delta_counter,
                                                     pair_location=pair_location)
        
        del pair_counter[max_pair]
        del pair_location[max_pair]

        # update heap and pair counter with only values that changed during merge
        # pair_counts.update(delta_count)
        for pair, count in delta_counter.items():
            if count != 0:
                curr_count = pair_counter[pair] + count
                if curr_count > 0:
                    pair_counter[pair] = curr_count
                    heapq.heappush(count_heap, HeapItem(pair_counter[pair], pair)) # only push to heap if its count is greater than zero
                else:
                    del pair_counter[pair]
        
        i += 1

    for special in special_tokens:
        if len(vocab) >= vocab_size:
            break
        vocab[len(vocab)] = special.encode("utf-8")

    return vocab, merges

@profile
def main():
    path = "tests/fixtures/tinystories_sample_5M.txt"
    return train_bpe(input_path=path, vocab_size=1000, special_tokens=["<|endoftext|>"])

if __name__ == "__main__":
    vocab, merges = main()
        