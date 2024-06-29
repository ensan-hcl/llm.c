"""
Tokenizes a text dataset using HuggingFace's AutoTokenizer.
- The tokenization is GPT-2 tokenizer with ku-nlp/gpt2-small-japanese-char.

The script prints:

Saved 32768 tokens to <val_output_file>
Saved <remaining_tokens> tokens to <train_output_file>

The .bin files are raw byte streams of int32 numbers indicating the token ids.
"""

import os
import argparse
from transformers import AutoTokenizer
import numpy as np
from data_common import write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "japanese_data")

tokenizer = AutoTokenizer.from_pretrained("ku-nlp/gpt2-small-japanese-char")
encode = lambda s: tokenizer.encode(s)


def tokenize(input_file, val_filename, train_filename):
    text = open(input_file, "r", encoding="utf-8").read()
    tokens = encode(text)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    write_datafile(val_filename, val_tokens)
    write_datafile(train_filename, train_tokens)
    print(f"Saved {len(val_tokens)} tokens to {val_filename}")
    print(f"Saved {len(train_tokens)} tokens to {train_filename}")


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
