import os
import argparse
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from data_common import write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "large_japanese_data")

tokenizer = AutoTokenizer.from_pretrained("ku-nlp/gpt2-small-japanese-char")

def encode(s: str):
    if s.startswith("\uEE00"):
        s = s + "</s>"
    else:
        s = "<s>" + s + "</s>"
    e = tokenizer.encode(s, max_length=192, truncation=True, padding='max_length', pad_to_max_length=True)
    return np.array(e, dtype=np.uint16)

def tokenize_document(doc):
    return encode(doc).astype(np.uint16)

def process_data(input_file, shard_size, val_filename, train_filename):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # valは10万行、trainは残り
    val_lines = lines[:100000]
    train_lines = lines[100000:]

    nprocs = max(1, os.cpu_count() - 2)
    
    def process_lines(lines, output_filename):
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        with mp.Pool(nprocs) as pool:
            for tokens in pool.imap(tokenize_document, lines, chunksize=16):
                if token_count + len(tokens) < shard_size:
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    filename = f"{output_filename}_{shard_index:06d}.bin"
                    remainder = shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens)-remainder
            
            if token_count != 0:
                filename = f"{output_filename}_{shard_index:06d}.bin"
                write_datafile(filename, all_tokens_np[:token_count])

    process_lines(val_lines, val_filename)
    process_lines(train_lines, train_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a text dataset using HuggingFace tokenizer for large datasets")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
    parser.add_argument("--val_output", type=str, default="large_japanese_data/japanese_data_val", help="Path to the validation output file (default: large_japanese_data/japanese_data_val)")
    parser.add_argument("--train_output", type=str, default="large_japanese_data/japanese_data_train", help="Path to the training output file (default: large_japanese_data/japanese_data_train)")

    args = parser.parse_args()

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    process_data(args.input_file, args.shard_size, args.val_output, args.train_output)