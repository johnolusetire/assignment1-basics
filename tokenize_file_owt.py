from cs336_basics.tokenizer import Tokenizer
import os
import time
import numpy as np

def tokenize_to_file(file_path: str, tokenizer: Tokenizer, save_path: str, *, flush_size: int = 25_000_000):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    buffer: list[int] = []
    total_tokens = 0
    
    with open(save_path, "wb") as out_file:
        with open(file_path, "r", encoding="utf-8", errors="replace") as in_file:
            # Pass file handle directly - encode_iterable reads line by line
            for token_id in tokenizer.encode_iterable(in_file, 1000, 1000):
                buffer.append(token_id)
                total_tokens += 1
                
                if len(buffer) >= flush_size:
                    print(f"Flushing buffer of size {len(buffer):,} tokens ({len(buffer) * 2 / 1024 / 1024:.1f} MB)")
                    np.array(buffer, dtype=np.uint16).tofile(out_file)
                    buffer.clear()
                if total_tokens % 15_000_000 == 0:
                    print("="*50)
                    print(f"Processed {total_tokens:,} tokens")
                    print("="*50)        
        if buffer:
            print(f"Final flush: {len(buffer):,} tokens ({len(buffer) * 2 / 1024 / 1024:.1f} MB)")
            np.array(buffer, dtype=np.uint16).tofile(out_file)
    print(f"Done tokenizing file. Total tokens: {total_tokens:,}")
    return total_tokens

def main():
    special_tokens = ["<|endoftext|>"]

    # Tokenize OpenWebText file  
    print("="*50)
    print("TOKENIZING OPEN WEB TEXT")
    print("="*50)
    
    vocab_owt = "results/owt/owt_train_vocab.pkl"
    merges_owt = "results/owt/owt_train_merge.pkl"

    tokenizer_owt = Tokenizer.from_files(vocab_filepath=vocab_owt,
                                        merge_filepath=merges_owt,
                                        special_tokens=special_tokens)
    
    owt_filepath = "data/owt_train.txt"
    output_file = "results/open_web_text_train_tokens.bin"
    print(f"Starting tokenization of Open web text train set")
    start_time = time.time()
    total_tokens_owt = tokenize_to_file(owt_filepath, tokenizer_owt, output_file)
    elapsed_owt = time.time() - start_time
    print(f"OpenWebText train set completed in {elapsed_owt:.2f} seconds")
    print(f"Output saved to: {output_file}")
    print(f"Throughput: {total_tokens_owt/elapsed_owt:,.0f} tokens/second")

    owt_valid_filepath = "data/owt_valid.txt"
    output_file = "results/open_web_text_valid_tokens.bin"
    print(f"Starting tokenization of Open web text validation")
    start_time = time.time()
    total_tokens_owt = tokenize_to_file(owt_valid_filepath, tokenizer_owt, output_file)
    elapsed_owt = time.time() - start_time
    print(f"OpenWebText val set completed in {elapsed_owt:.2f} seconds")
    print(f"Output saved to: {output_file}")
    print(f"Throughput: {total_tokens_owt/elapsed_owt:,.0f} tokens/second")

if __name__ == "__main__":
    main()