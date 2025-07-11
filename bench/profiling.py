from cs336_basics.bpe_v2_2 import train_bpe as train_bpe_v2_1

import time
import cProfile
import pstats
import io

#path = "tests/fixtures/corpus.en"
path = "tests/fixtures/tinystories_sample_5M.txt"
VOCAB_SIZE = 1000
SPECIAL_TOKENS = ["<|endoftext|>"]

# Create a profiler object
profiler = cProfile.Profile()

# Enable profiling
profiler.enable()

# Run the function you want to profile
start_time = time.time()
vocab, merges = train_bpe_v2_1(path, VOCAB_SIZE, SPECIAL_TOKENS) # Call the specific version
end_time = time.time()

# Disable profiling
profiler.disable()

print(f"It took {end_time - start_time:.2f}s to train")

# Analyze the results
s = io.StringIO()
# Sort stats by cumulative time spent in the function
sortby = pstats.SortKey.CUMULATIVE 
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
# ps.print_stats() # Print all stats

# You can also print specific numbers of lines, e.g., top 20
ps.print_stats(20) 

print(s.getvalue())

# To save to a file for more detailed analysis (e.g., with snakeviz):
profiler.dump_stats("train_bpe_v2_2_profile.prof")