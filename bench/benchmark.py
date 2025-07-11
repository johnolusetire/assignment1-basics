#from cs336_basics.bpe import train_bpe
from cs336_basics.bpe_v3 import train_bpe
import time

#path = "tests/fixtures/corpus.en"
path = "tests/fixtures/tinystories_sample_5M.txt"
times = []
NUM_TESTS = 1
for _ in range(NUM_TESTS):
    start_time = time.time()
    vocab, merges = train_bpe(path, 1000, ["<|endoftext|>"])
    end_time = time.time()
    times.append(end_time-start_time)

print(f"It took {sum(times) / NUM_TESTS} to train (avg)")
print(f"trial times: {[f'{t:0.2f}' for t in times]}")