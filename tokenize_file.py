from cs336_basics.tokenizer import Tokenizer
import os
import time
import numpy as np
from typing import Iterator

class DocumentSampler:
    def __init__(self, file_path: str, delimiter: str = "<|endoftext|>", chunk_size: int = 1) -> None:
        self.file_path = file_path
        self.delimiter = delimiter
        self.chunk_size = chunk_size * 1024 *1024 # MB
    
    def sample_all(self) -> Iterator[str]:
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as file:
            while True:
                chunk = file.read(self.chunk_size)
                if chunk == "":
                    break
                yield chunk
    
    def sample(self, num_samples: int = 10):
        buffer = ""
        chunk = ""
        docs_sampled = 0
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as file:
            while docs_sampled < num_samples:
                chunk = file.read(self.chunk_size)
                if chunk == "":
                    break
                chunk = buffer + chunk
                parts = chunk.split(self.delimiter)

                buffer = parts[-1]
                docs = parts[:-1]

                for doc in docs:
                    if doc:
                        yield doc
                        docs_sampled += 1
                    if docs_sampled >= num_samples:
                        return

def tokenize_to_file(sampler: DocumentSampler, tokenizer: Tokenizer, save_path: str, *, flush_size: int = 1_000_000):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    buffer: list[int] = []
    with open(save_path, "wb") as file:
        for chunk in sampler.sample_all():
            buffer.extend(tokenizer.encode_iterable(chunk))
            if len(buffer) >= flush_size:
                print(f"Flushing buffer of size {len(buffer)}")
                np.array(buffer, dtype=np.uint16).tofile(file)
                buffer.clear()
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(file)
    print("Done tokenizing file")
    return

def main():
    special_tokens = ["<|endoftext|>"]

    # Tokenize tinystories file
    tn_filepath = "data/TinyStoriesV2-GPT4-train.txt"
    output_file = "results/tiny_stories_tokens.bin"
    vocab_tn = "results/tiny_stories/tiny_stories_vocab.pkl"
    merges_tn = "results/tiny_stories/tiny_stories_merges.pkl"
    tiny_stories_sampler = DocumentSampler(tn_filepath)
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_tn,
                                     merge_filepath=merges_tn,
                                     special_tokens=special_tokens)
    
    print(f"Starting tokenization of tiny stories")
    start_time = time.time()
    tokenize_to_file(tiny_stories_sampler, tokenizer, output_file)
    print(f"Done tokenizing tiny stories file in {time.time() - start_time:2f} seconds. File saved at {output_file}")


        # Tokenize tinystories file
    owt_filepath = "data/owt_train.txt"
    output_file = "results/open_web_text_tokens.bin"
    vocab_owt = "results/owt/owt_train_vocab.pkl"
    merges_owt = "results/owt/owt_train_merge.pkl"
    open_web_text_sampler = DocumentSampler(owt_filepath)
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_owt,
                                     merge_filepath=merges_owt,
                                     special_tokens=special_tokens)
    
    print(f"Starting tokenization of open web text")
    start_time = time.time()
    tokenize_to_file(open_web_text_sampler, tokenizer, output_file)
    print(f"Done tokenizing open web text file in {time.time() - start_time:2f} seconds")

if __name__ == "__main__":
    main()