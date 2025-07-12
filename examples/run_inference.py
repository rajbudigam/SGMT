#!/usr/bin/env python3
"""
examples/run_inference.py

Load the trained SGMT checkpoint and run inference on new
commands. Prints the model’s predicted action sequences.
"""

import os
import torch
import argparse

from sgmt.model import SGMT

def decode_sequence(seq_indices, vocab):
    """Convert list of token indices back into strings."""
    return " ".join(vocab[idx] for idx in seq_indices if idx != 0)

def main(checkpoint, input_cmds):
    # 1) Load checkpoint
    data = torch.load(checkpoint, map_location="cpu")
    src_vocab = data["src_vocab"]
    tgt_vocab = data["tgt_vocab"]
    model = SGMT(src_vocab, tgt_vocab)
    model.load_state_dict(data["model_state"])
    model.eval()

    # 2) Tokenize and run model
    for cmd in input_cmds:
        tokens = cmd.strip().split()
        src_idxs = [src_vocab.index(t) if t in src_vocab else 0 for t in tokens]
        src = torch.tensor([src_idxs], dtype=torch.long)
        # start with BOS (we use zero padding here)
        tgt_in = torch.zeros((1, len(src_idxs)), dtype=torch.long)

        with torch.no_grad():
            logits, _ = model(src, tgt_in)
        pred_idxs = logits.argmax(dim=-1)[0].tolist()
        # 3) Decode output
        pred_seq = decode_sequence(pred_idxs, tgt_vocab)
        print(f">>> Input : {cmd}")
        print(f"    Output: {pred_seq}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGMT Inference Example")
    parser.add_argument(
        "--checkpoint", type=str,
        default="examples/sgmt_scan.pt",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "cmds", nargs="+",
        help="One or more commands (e.g. \"jump twice and walk\")"
    )
    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    main(args.checkpoint, args.cmds)
