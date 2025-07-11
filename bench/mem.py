import sys
import array
import random
import time

# Simulate a large list of integer word IDs
N = 10_000_000
words_list = [random.randint(0, 100_000) for _ in range(N)]

# Store the same data in an array of unsigned integers
words_array = array.array('I', words_list)

print(f"List of {N} integers: {sys.getsizeof(words_list)/1024/1024:.2f} MB (container only)")
print(f"Array of {N} integers: {sys.getsizeof(words_array)/1024/1024:.2f} MB (container only)")

# Add up the memory used by the elements themselves
list_total = sum(sys.getsizeof(x) for x in words_list)
array_total = sys.getsizeof(words_array)

print(f"Total memory for list: {list_total/1024/1024:.2f} MB")
print(f"Total memory for array: {array_total/1024/1024:.2f} MB")

# Optional: time access speed
start = time.time()
s = sum(words_list)
print(f"Sum of list: {s} (time: {time.time() - start:.2f}s)")

start = time.time()
s = sum(words_array)
print(f"Sum of array: {s} (time: {time.time() - start:.2f}s)")