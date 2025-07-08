from cs336_basics.bpe_v2_main import train_bpe
import time

#path = "tests/fixtures/corpus.en"
#path = "data/TinyStoriesV2-GPT4-valid.txt"
path = "data/TinyStoriesV2-GPT4-train.txt"
start_time = time.time()
vocab, merges = train_bpe(path, 10000, ["<|endoftext|>"], block_size=64, verbose=True)
end_time = time.time()
print(f"It took {end_time-start_time} to train")

import pickle
save_path = "results/tiny_stories_valid.pkl"


save_data = {
        "vocab_keys": set(vocab.keys()),
        "vocab_values": set(vocab.values()),
        "merges": merges }

with open(save_path, "wb") as f:
    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)