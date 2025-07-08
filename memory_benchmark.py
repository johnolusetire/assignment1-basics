import psutil
import os
import time
from cs336_basics.bpe_v2_main import train_bpe

def benchmark_memory():
    process = psutil.Process(os.getpid())
    
    # Memory before training
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Start training with memory monitoring
    start_time = time.time()
    
    try:
        vocab, merges = train_bpe(
            input_path="data/TinyStoriesV2-GPT4-valid.txt",
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
            ##verbose=False
        )
        
        # Peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        print(f"Memory increase: {peak_memory - initial_memory:.2f} MB")
        
    except MemoryError:
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"MemoryError at {current_memory:.2f} MB")
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    benchmark_memory()