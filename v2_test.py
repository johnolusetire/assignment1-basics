#from cs336_basics.bpe import train_bpe
#from cs336_basics.bpe_v2_2 import train_bpe
from cs336_basics.bpe_v6 import train_bpe
from cs336_basics.damekbpe import BPETrainer
import time

#path = "tests/fixtures/corpus.en"
#path = "tests/fixtures/tinystories_sample_5M.txt"
path = "data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10
special_tokens = ["<|endoftext|>"]


# start_time = time.time()
# vocab1, merges1 = train_bpe(path, vocab_size, special_tokens, mode="multi", verbose=True)
# end_time = time.time()
# print(f"It took {end_time-start_time} to train old")
# print("-" * 50)

start_time = time.time()
bpe = BPETrainer()
vocab2, merges2 = bpe.train(path, vocab_size, special_tokens)
end_time = time.time()
print(f"It took {end_time-start_time} to train new")
print("-" * 50)

# if vocab1 == vocab2:
#     print("vocabs match")
# else:
#     print("vocabs don't match")

# if merges1 == merges2:
#     print("merges match")
# else:
#     print("merges don't match")

# import pickle
# save_path = "v2.pkl"


# save_data = {
#         "vocab_keys": set(vocab.keys()),
#         "vocab_values": set(vocab.values()),
#         "merges": merges }

# with open(save_path, "wb") as f:
#     pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)



