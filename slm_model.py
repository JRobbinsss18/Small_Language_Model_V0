import math, torch
import torch.nn as nn
import torch.nn.functional as F

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, n_layers=4, n_heads=4, emb_dim=256, ff_mult=4, dropout=0.2, pad_id=0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.blocks = nn.Sequential(*[self._block(emb_dim, n_heads, ff_mult, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
        self.context_length = context_length
        self.pad_id = pad_id
    def _block(self, C, H, M, d):
        head = C // H
        attn = nn.MultiheadAttention(C, H, dropout=d, batch_first=True)
        ff = nn.Sequential(nn.Linear(C, C*M), nn.ReLU(), nn.Dropout(d), nn.Linear(C*M, C))
        ln1 = nn.LayerNorm(C)
        ln2 = nn.LayerNorm(C)
        return nn.Sequential(nn.Identity()) if False else nn.ModuleDict({
            'attn': attn,'ff': ff,'ln1': ln1,'ln2': ln2
        })
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for b in self.blocks:
            x = x + b['attn'](b['ln1'](x), b['ln1'](x), b['ln1'](x), attn_mask=torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool())[0]
            x = x + b['ff'](b['ln2'](x))
        x = self.ln(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T), ignore_index=self.pad_id)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.5, top_p=0.9, stop_ids=()):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / max(1e-8, temperature)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            cutoff = (cum > top_p).float().argmax(dim=-1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            for b in range(logits.size(0)):
                k = cutoff[b].item()
                mask[b, sorted_indices[b, k+1:]] = False
            logits = torch.where(mask, logits, torch.full_like(logits, -float('inf')))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            if next_id.item() in stop_ids:
                idx = torch.cat([idx, next_id], dim=1)
                break
            idx = torch.cat([idx, next_id], dim=1)
        return idx