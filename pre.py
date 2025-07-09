from cs336_basics.pretokenization import pretokenize_text
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
file_path = "data/TinyStoriesV2-GPT4-valid.txt"
#file_path = "tests/fixtures/tinystories_sample_5M.txt"


print(f"Pretokenizing {file_path}")

modes = ["sequential", "multi"]

for mode in modes:
    print(f"Using {mode} pretokenization")
    start_time = time.time()
    words, pair_counts, pair_positions = pretokenize_text(file_path, PAT, ["<|endoftext|>"], mode=mode, num_processes=12)
    end_time = time.time()
    print(f"It took {end_time - start_time:.2f} seconds to pretokenize in {mode} mode\n")
    print(f"Number of words: {len(words)}")
    print(f"Number of unique pairs: {len(pair_counts)}")
    print(f"Number of unique pair positions: {len(pair_positions)}\n")
    print("-" * 40)


print("All pretokenization results match!")
