import psutil
import os
import time
import cProfile
import pstats
import io
from cs336_basics.bpe_v6 import train_bpe

def profile_train_bpe():
    """Comprehensive profiling of train_bpe function"""
    
    # Create profiler
    profiler = cProfile.Profile()
    
    print("Starting detailed profiling...")
    
    # Profile the training
    profiler.enable()
    
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        verbose=False,
        mode="multi"
    )
    
    profiler.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    print("\n" + "="*80)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80)
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)
    
    print("\n" + "="*80)
    print("TOP 20 FUNCTIONS BY TOTAL TIME (excluding subcalls)")
    print("="*80)
    ps.sort_stats(pstats.SortKey.TIME)
    ps.print_stats(20)
    
    print("\n" + "="*80)
    print("MOST CALLED FUNCTIONS")
    print("="*80)
    ps.sort_stats(pstats.SortKey.CALLS)
    ps.print_stats(20)
    
    # Save detailed report
    with open("bpe_profile_report.txt", "w") as f:
        f.write(s.getvalue())
    
    # Save binary profile for external tools
    profiler.dump_stats("bpe_profile.prof")
    
    print(f"\nDetailed reports saved to:")
    print(f"  - bpe_profile_report.txt")
    print(f"  - bpe_profile.prof (use with snakeviz: pip install snakeviz && snakeviz bpe_profile.prof)")
    
    return vocab, merges

if __name__ == "__main__":
    profile_train_bpe()