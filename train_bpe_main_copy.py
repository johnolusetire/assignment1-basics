import os
from cs336_basics.damekbpe import BPETrainer
import time
import pickle
import argparse

def main():
    #path = "tests/fixtures/corpus.en"
    #path = "data/TinyStoriesV2-GPT4-valid.txt"
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, default="data/TinyStoriesV2-GPT4-train.txt", help="Path to the training data file")
    args.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary to be created")
    #args.add_argument("--mode", type=str, default="multi", choices=["sequential", "multi"], help="Mode of BPE training: 'sequential' for single token, 'multi' for multiple tokens")
    args.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"], help="List of special tokens to be added to the vocabulary")
    args.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging")
    args.add_argument("--output_file", type=str, default="tiny_stories", help="Directory to save the output vocabulary and merges")

    args = args.parse_args()
    path = args.path
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens
    #mode = args.mode
    verbose = args.verbose

    print(f"Training BPE with path: {path}, vocab_size: {vocab_size}, special_tokens: {special_tokens}, verbose: {verbose}")
    start_time = time.time()
    bpe = BPETrainer()
    vocab, merges = bpe.train(path, vocab_size, special_tokens, verbose)
    end_time = time.time()
    print(f"It took {end_time-start_time} to train")

    result_dir = "results/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    vocab_path = os.path.join(result_dir, args.output_file + "_vocab.pkl")    
    print(f"Saving vocab to {vocab_path}")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    merge_path = os.path.join(result_dir, args.output_file + "_merge.pkl")    
    print(f"Saving vocab to {merge_path}")

    with open(merge_path, "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Vocab length: {len(vocab)}")
    print(f"Merge length: {len(merges)}")

if __name__ == "__main__":        
    main()
