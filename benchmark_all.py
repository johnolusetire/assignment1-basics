

import time
import os

# Import all the different BPE training functions
from cs336_basics.bpe import train_bpe as train_bpe_v1
from cs336_basics.bpe_v2 import train_bpe as train_bpe_v2
from cs336_basics.bpe_v2_2 import train_bpe as train_bpe_v2_2
from cs336_basics.bpe_v2_main import train_bpe as train_bpe_v2_main
from cs336_basics.bpe_v3 import train_bpe as train_bpe_v3
from cs336_basics.bpe_v6 import train_bpe as train_bpe_v6

def run_benchmark():
    """
    Benchmarks the different BPE implementations.
    """
    dataset_path = "tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    implementations = {
        "v2 (Indexed)": train_bpe_v2,
        "v2.2 (Indexed Set)": train_bpe_v2_2,
        "v2_main (Indexed Refined)": train_bpe_v2_main,
        "v3 (Linked List)": train_bpe_v3,
        "v6 (Corrected Two-Pointer)": train_bpe_v6,
    }

    results = {}

    print(f"Starting benchmark on dataset: {dataset_path}")
    print("-" * 30)

    for name, train_func in implementations.items():
        print(f"Running {name}...")
        start_time = time.time()
        try:
            train_func(dataset_path, vocab_size, special_tokens)
            end_time = time.time()
            duration = end_time - start_time
            results[name] = duration
            print(f"Finished in {duration:.4f} seconds.")
        except Exception as e:
            results[name] = float('inf') # Indicate failure
            print(f"Failed with error: {e}")
        print("-" * 30)


    # Sort results from fastest to slowest
    sorted_results = sorted(results.items(), key=lambda item: item[1])

    print("\n--- Benchmark Results ---")
    print("Ranked from fastest to slowest:")
    for i, (name, duration) in enumerate(sorted_results):
        if duration == float('inf'):
            print(f"{i+1}. {name}: FAILED")
        else:
            print(f"{i+1}. {name}: {duration:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
