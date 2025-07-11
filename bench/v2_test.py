#from cs336_basics.bpe import train_bpe
#from cs336_basics.bpe_v2_2 import train_bpe
from cs336_basics.bpe_v6 import train_bpe
from cs336_basics.damekbpe import BPETrainer
import time

#path = "tests/fixtures/corpus.en"
#path = "tests/fixtures/tinystories_sample_5M.txt"
path = "data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]


start_time = time.time()
vocab1, merges1 = train_bpe(path, vocab_size, special_tokens, mode="multi", verbose=False)
end_time = time.time()
print(f"It took {end_time-start_time} to train")
print("-" * 50)
print(f"Vocab length: {len(vocab1)}")
print(f"Merge length: {len(merges1)}")

start_time = time.time()
bpe = BPETrainer()
vocab2, merges2 = bpe.train(path, vocab_size, special_tokens, verbose=False)
end_time = time.time()
print(f"It took {end_time-start_time} to train new")
print("-" * 50)

for i in range(256,len(vocab1)):
    if vocab1[i] != vocab2[i]:
        print(f"{vocab1[i]}     {vocab2[i]}")

if merges1 == merges2:
    print("merges match")
else:
    print("merges don't match")

# import pickle
# save_path = "v2.pkl"


# save_data = {
#         "vocab_keys": set(vocab.keys()),
#         "vocab_values": set(vocab.values()),
#         "merges": merges }

# with open(save_path, "wb") as f:
#     pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)



