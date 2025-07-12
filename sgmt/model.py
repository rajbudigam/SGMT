import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters

        # Slot parameters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_sigma = nn.Parameter(torch.randn(1, 1, dim))

        # Projection layers
        self.project_q = nn.Linear(dim, dim)
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)

        # GRU & MLP for slot update
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Normalisation layers
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run slot attention over encoded inputs.

        Args:
            x: Tensor of shape (B, N, D)
        Returns:
            slots: Tensor of shape (B, num_slots, D)
        """
        B, N, D = x.shape

        # Initialise slots
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = torch.exp(self.slot_sigma).expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Pre-normalise inputs
        x = self.norm_inputs(x)
        k = self.project_k(x)  # (B, N, D)
        v = self.project_v(x)

        for _ in range(self.iters):
            # Compute attention
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)  # (B, S, D)
            attn_logits = torch.einsum("bsd,bnd->bsn", q, k)
            attn = F.softmax(attn_logits, dim=-1)  # (B, S, N)

            # Weighted sum
            updates = torch.einsum("bnd,bsn->bsd", v, attn)

            # Slot update via GRU
            slots_flat = slots.view(-1, D)
            updates_flat = updates.view(-1, D)
            slots_flat = self.gru(updates_flat, slots_flat)
            slots = slots_flat.view(B, self.num_slots, D)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class SGMT(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model: int = 256,
        num_heads: int = 8,
        num_slots: int = 6,
        num_modules: int = 8,
    ) -> None:
        super().__init__()

        # Embedding layers
        self.src_tok = nn.Embedding(len(src_vocab), d_model)
        self.tgt_tok = nn.Embedding(len(tgt_vocab), d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, d_model))

        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=2,
        )

        # Slot Attention
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=d_model)

        # Router
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modules),
        )

        # Expert modules
        self.expert_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_modules)
        ])

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=2,
        )
        self.out_proj = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor):
        # Embedding + positional encoding
        src_emb = self.src_tok(src) + self.pos_enc[:, : src.size(1), :]
        tgt_emb = self.tgt_tok(tgt_in) + self.pos_enc[:, : tgt_in.size(1), :]

        # Encode
        memory = self.encoder(src_emb.transpose(0, 1)).transpose(0, 1)

        # Slot attention
        slots = self.slot_attention(memory)

        # Routing
        routing_logits = self.router(slots)
        routing_weights = F.gumbel_softmax(routing_logits, tau=1.0, hard=True, dim=-1)

        routed = torch.zeros_like(slots)
        for i, module in enumerate(self.expert_modules):
            mask = routing_weights[:, :, i].unsqueeze(-1)
            routed += mask * module(slots)

        # Decode
        decoded = self.decoder(
            tgt_emb.transpose(0, 1),
            memory.transpose(0, 1),
        ).transpose(0, 1)

        logits = self.out_proj(decoded)
        return logits, routing_weights
