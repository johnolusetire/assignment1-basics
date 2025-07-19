#from cs336_basics.bpe import train_bpe
from cs336_basics.bpe import train_bpe
import time

#path = "tests/fixtures/corpus.en"
path = "tests/fixtures/tinystories_sample_5M.txt"
start_time = time.time()
vocab, merges = train_bpe(path, 1000, ["<|endoftext|>"])
end_time = time.time()
print(f"It took {end_time-start_time} to train")

import pickle
save_path = "v1.pkl"


save_data = {
        "vocab_keys": set(vocab.keys()),
        "vocab_values": set(vocab.values()),
        "merges": merges }

with open(save_path, "wb") as f:
    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)



