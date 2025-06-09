from cs336_basics.bpe import train_bpe
import time

path = "tests/fixtures/corpus.en"
start_time = time.time()
vocab, merges = train_bpe(path, 500, ["<|endoftext|>"])
end_time = time.time()
print(f"It took {end_time-start_time} to train")

# import pickle
# save_path = "new8.pkl"


# save_data = {
#         "vocab_keys": set(vocab.keys()),
#         "vocab_values": set(vocab.values()),
#         "merges": merges }

# with open(save_path, "wb") as f:
#     pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)



