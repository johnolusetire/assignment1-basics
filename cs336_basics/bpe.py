import regex as re
from collections import Counter, defaultdict
import heapq
import os
from itertools import pairwise

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
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
    BLOCK = 64 * 1024 * 1024 # chunk block size
    pair_counts = Counter()
    words = []
    #pair_positions = defaultdict(list)


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
                    pair_counts.update(pairwise(ids))  # update counter with pairs from each match. same as looping through with zip(ids, ids[1:])

    
    # use a priority queue/heap to keep track of maximum pairs instead of using the max function
    # first build the initial heap
    count_heap = []
    for pair, count in pair_counts.items():
        heapq.heappush(count_heap, HeapItem(count, pair))
    
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

        for ids in words:
            if max_pair[0] in ids:
                ids, delta_count = merge(ids=ids, pair=max_pair, local_delta=delta_count)


        # update heap and pair counter with only values that changed during merge
        # pair_counts.update(delta_count)
        for pair, count_delta in delta_count.items():
            if count_delta != 0:
                curr_count = pair_counts.get(pair, 0) + count_delta
                if curr_count > 0:
                    pair_counts[pair] = curr_count
                    heapq.heappush(count_heap, HeapItem(pair_counts[pair], pair)) # only push to heap if its count is greater than zero
                else:
                    del pair_counts[pair]       

        i += 1   
    
    for special in special_tokens:
        if len(vocab) >= vocab_size:
            break
        vocab[len(vocab)] = special.encode("utf-8")

    return vocab, merges



def merge(ids: list[bytes], pair: tuple[bytes, bytes], local_delta: Counter[tuple[bytes, bytes]]) -> tuple[list[bytes], Counter[tuple[bytes, bytes]]]:
    A, B = pair[0], pair[1]  # pair is e.g (b'h', b'e')
    new_val = A + B  # new_val wil be b'he'
    
    # two pointer in place merge
    write = read = 0
    while read < len(ids):
        if read < len(ids) - 1 and ids[read] == A and ids[read + 1] == B:
            # pair found. record the change for this list of ids by removing that pair from the count
            local_delta[pair] -= 1

            # remove pair to the left of found pair
            if write > 0:
                local_delta[(ids[write - 1], ids[read])] -= 1
                local_delta[(ids[write - 1], new_val)] += 1
            
            ids[write] = new_val # write new_val into list of bytes
            read += 2 # skip the next element

            if read < len(ids):
                local_delta[(ids[read - 1], ids[read])] -= 1
                local_delta[(new_val, ids[read])] += 1
        else:
            ids[write] = ids[read]
            read += 1
        write += 1

    del ids[write:]
   
    return ids, local_delta


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