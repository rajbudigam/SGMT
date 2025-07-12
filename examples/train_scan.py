#!/usr/bin/env python3
"""
examples/train_scan.py

Generate a toy SCAN‐style dataset and train SGMT for a few epochs.
Saves a checkpoint to examples/sgmt_scan.pt
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from sgmt.model import SGMT

# 1) Define a tiny SCAN‐style data generator
VERBS      = ["walk", "jump", "run", "look", "zigzag", "twirl"]
MODIFIERS  = ["twice", "thrice"]
CONNECTORS = ["and", "after"]

def make_command():
    """Randomly sample a command like 'jump twice and walk thrice'."""
    verb1 = random.choice(VERBS)
    mod1  = random.choice(MODIFIERS) if random.random()<0.5 else None
    verb2 = random.choice(VERBS)
    mod2  = random.choice(MODIFIERS) if random.random()<0.5 else None
    conn  = random.choice(CONNECTORS)
    parts = []
    parts.append(verb1)
    if mod1: parts.append(mod1)
    parts.append(conn)
    parts.append(verb2)
    if mod2: parts.append(mod2)
    return parts

def compile_action_sequence(cmd_tokens):
    """
    Dummy action generator: maps each token to a single action symbol.
    In a real SCAN you’d expand 'jump twice' → 'JUMP JUMP', etc.
    Here we just uppercase tokens.
    """
    return [token.upper() for token in cmd_tokens]

# 2) Build vocabularies
SRC_VOCAB = sorted({t for _ in range(1) for t in VERBS + MODIFIERS + CONNECTORS})
TGT_VOCAB = sorted({w.upper() for w in SRC_VOCAB})

SRC2IDX = {tok:i for i,tok in enumerate(SRC_VOCAB)}
TGT2IDX = {tok:i for i,tok in enumerate(TGT_VOCAB)}

# 3) PyTorch Dataset
class ScanDataset(Dataset):
    def __init__(self, size=1000):
        self.commands = [make_command() for _ in range(size)]
        self.actions  = [compile_action_sequence(cmd) for cmd in self.commands]

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        src = torch.tensor([SRC2IDX[t] for t in self.commands[idx]], dtype=torch.long)
        tgt = torch.tensor([TGT2IDX[t] for t in self.actions[idx]], dtype=torch.long)
        return src, tgt

def collate_fn(batch):
    # Pads src and tgt to the same length per batch
    srcs, tgts = zip(*batch)
    srcs = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts = pad_sequence(tgts, batch_first=True, padding_value=0)
    return srcs, tgts

# 4) Hyperparameters
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 1e-3
CKPT_PATH  = os.path.join("examples", "sgmt_scan.pt")

# 5) DataLoaders
train_ds = ScanDataset(size=2000)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 6) Model, optimizer, loss
model = SGMT(SRC_VOCAB, TGT_VOCAB)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# 7) Training loop
model.train()
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0
    total_tok  = 0
    correct_tok= 0

    for src_batch, tgt_batch in train_dl:
        # shift tgt for teacher forcing
        tgt_in  = tgt_batch[:, :-1]
        tgt_out = tgt_batch[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(src_batch, tgt_in)
        # (B, L, V) → (B*L, V), tgt_out → (B*L,)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        # Logging token‐level accuracy
        total_loss += loss.item() * tgt_out.numel()
        preds = logits.argmax(dim=-1)
        mask  = (tgt_out != 0)
        correct_tok += ((preds == tgt_out) & mask).sum().item()
        total_tok   += mask.sum().item()

    avg_loss = total_loss / total_tok
    acc = correct_tok / total_tok * 100
    print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f} — Token Accuracy: {acc:.2f}%")

# 8) Save checkpoint
torch.save({
    "model_state": model.state_dict(),
    "src_vocab":   SRC_VOCAB,
    "tgt_vocab":   TGT_VOCAB
}, CKPT_PATH)
print(f"Checkpoint saved to {CKPT_PATH}")
