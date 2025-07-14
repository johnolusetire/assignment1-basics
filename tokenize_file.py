from cs336_basics.tokenizer import Tokenizer
import os
import time
import numpy as np

def tokenize_to_file(file_path: str, tokenizer: Tokenizer, save_path: str, *, flush_size: int = 1_000_000):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    buffer: list[int] = []
    total_tokens = 0
    
    with open(save_path, "wb") as out_file:
        with open(file_path, "r", encoding="utf-8", errors="replace") as in_file:
            # Pass file handle directly - encode_iterable reads line by line
            for token_id in tokenizer.encode_iterable(in_file):
                buffer.append(token_id)
                total_tokens += 1
                
                if len(buffer) >= flush_size:
                    print(f"Flushing buffer of size {len(buffer):,} tokens")
                    np.array(buffer, dtype=np.uint16).tofile(out_file)
                    buffer.clear()
        
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(out_file)
    
    return total_tokens

def main():
    special_tokens = ["<|endoftext|>"]

    # Tokenize tinystories file
    print("="*50)
    print("TOKENIZING TINY STORIES")
    print("="*50)

    # Tokenize tinystories file
    tn_filepath = "data/TinyStoriesV2-GPT4-train.txt"
    output_file = "results/tiny_stories_tokens.bin"
    vocab_tn = "results/tiny_stories/tiny_stories_vocab.pkl"
    merges_tn = "results/tiny_stories/tiny_stories_merges.pkl"
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_tn,
                                     merge_filepath=merges_tn,
                                     special_tokens=special_tokens)
    
    print(f"Starting tokenization of tiny stories")
    start_time = time.time()
    total_tokens_tn = tokenize_to_file(tn_filepath, tokenizer, output_file)
    elapsed_tn = time.time() - start_time
    
    print(f"Tiny Stories completed in {elapsed_tn:.2f} seconds")
    print(f"Output saved to: {output_file}")
    print(f"Throughput: {total_tokens_tn/elapsed_tn:,.0f} tokens/second")


    # Tokenize OpenWebText file  
    print("\n" + "="*50)
    print("TOKENIZING OPEN WEB TEXT")
    print("="*50)
    
    owt_filepath = "data/owt_train.txt"
    output_file = "results/open_web_text_tokens.bin"
    vocab_owt = "results/owt/owt_train_vocab.pkl"
    merges_owt = "results/owt/owt_train_merge.pkl"

    tokenizer_owt = Tokenizer.from_files(vocab_filepath=vocab_owt,
                                        merge_filepath=merges_owt,
                                        special_tokens=special_tokens)
    
    start_time = time.time()
    total_tokens_owt = tokenize_to_file(owt_filepath, tokenizer_owt, output_file)
    elapsed_owt = time.time() - start_time
    
    print(f"OpenWebText completed in {elapsed_owt:.2f} seconds")
    print(f"Output saved to: {output_file}")
    print(f"Throughput: {total_tokens_owt/elapsed_owt:,.0f} tokens/second")

if __name__ == "__main__":
    main()