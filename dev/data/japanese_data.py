import os
import argparse
from transformers import AutoTokenizer
from data_common import write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "japanese_data")

tokenizer = AutoTokenizer.from_pretrained("ku-nlp/gpt2-small-japanese-char")
def encode(s: str):
    if s.startswith("\uEE00"):
        s = s + "</s>"
    else:
        s = "<s>" + s + "</s>"
    e = tokenizer.encode(s, max_length=192, truncation=True, padding='max_length', pad_to_max_length=True)
    return e

def tokenize(input_file, val_filename, train_filename):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    train_lines = lines[:1400000]
    val_lines = lines[1400000:]

    train_tokens = [encode(line) for line in train_lines]
    val_tokens = [encode(line) for line in val_lines]

    train_tokens_flat = [token for sublist in train_tokens for token in sublist]
    val_tokens_flat = [token for sublist in val_tokens for token in sublist]

    write_datafile(val_filename, val_tokens_flat)
    print(f"Saved {len(val_tokens_flat)} tokens to {val_filename}")
    write_datafile(train_filename, train_tokens_flat)
    print(f"Saved {len(train_tokens_flat)} tokens to {train_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize a text dataset using HuggingFace tokenizer"
    )
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument(
        "--val_output",
        type=str,
        default="japanese_data/japanese_data_val.bin",
        help="Path to the validation output file (default: japanese_data/japanese_data_val.bin)",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="japanese_data/japanese_data_train.bin",
        help="Path to the training output file (default: japanese_data/japanese_data_train.bin)",
    )

    args = parser.parse_args()

    tokenize(args.input_file, args.val_output, args.train_output)
