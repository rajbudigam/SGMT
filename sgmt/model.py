# sgmt/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_sigma = nn.Parameter(torch.randn(1, 1, dim))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = F.softplus(self.slot_sigma).expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        x = self.norm_inputs(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.q_proj(slots_norm)
            attn_logits = torch.einsum("bid,bjd->bij", q, k)
            attn = F.softmax(attn_logits, dim=1)
            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.view(B, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

class SGMT(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=256,
        num_heads=8,
        num_slots=6,
        num_modules=8
    ):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tok = nn.Embedding(len(src_vocab), d_model)
        self.tgt_tok = nn.Embedding(len(tgt_vocab), d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, d_model))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=4
        )
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=d_model)

        self.router = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, num_modules)
        )

        # Rename from `modules` to `expert_modules` to avoid name clash
        self.expert_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
            for _ in range(num_modules)
        ])

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=2
        )
        self.out_proj = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt_in):
        # Token embeddings + positional encoding
        src_embed = self.src_tok(src) + self.pos_enc[:, :src.size(1), :]
        tgt_embed = self.tgt_tok(tgt_in) + self.pos_enc[:, :tgt_in.size(1), :]

        # Encode source
        memory = self.encoder(src_embed.transpose(0, 1)).transpose(0, 1)

        # Slot attention
        slots = self.slot_attention(memory)

        # Routing
        B, S, D = slots.shape
        routing_logits = self.router(slots)
        routing_weights = F.gumbel_softmax(
            routing_logits, tau=1.0, hard=True, dim=-1
        )

        # Apply expert modules
        routed = torch.zeros_like(slots)
        for i, module in enumerate(self.expert_modules):
            mask = routing_weights[:, :, i].unsqueeze(-1)
            routed = routed + mask * module(slots)

        # Decode
        decoded = self.decoder(
            tgt_embed.transpose(0, 1),
            memory.transpose(0, 1)
        ).transpose(0, 1)

        out = self.out_proj(decoded)
        return out, routing_weights
