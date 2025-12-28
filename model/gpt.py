"""
SpacingGPT: Transformer for zeta zero spacing prediction.
Adapted from nanoGPT for spectral sequence modeling.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Model configuration."""
    vocab_size: int = 256      # number of spacing bins
    seq_len: int = 256         # context length
    n_layer: int = 4           # transformer layers
    n_head: int = 4            # attention heads
    n_embd: int = 128          # embedding dimension
    dropout: float = 0.1
    bias: bool = False         # use bias in Linear and LayerNorm


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len))
            .view(1, 1, config.seq_len, config.seq_len)
        )

    def forward(self, x, return_attention=False):
        B, T, C = x.size()

        # Calculate query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Handle dynamic sequence lengths (for RMT with memory tokens)
        if T <= self.mask.size(2):
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        else:
            # Create dynamic mask for longer sequences
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask.view(1, 1, T, T), float('-inf'))

        att = F.softmax(att, dim=-1)
        att_weights = att  # Save before dropout for analysis
        att = self.attn_dropout(att)
        y = att @ v

        # Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if return_attention:
            return y, att_weights
        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class SpacingGPT(nn.Module):
    """
    GPT for spacing sequence prediction.

    Input: sequence of bin indices (B, T) where each index is 0..vocab_size-1
    Output: logits (B, T, vocab_size) for next token prediction
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.seq_len, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SpacingGPT: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attention=False):
        """
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target indices (optional)
            return_attention: if True, return attention weights from all layers

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided
            attentions: list of attention weights if return_attention=True
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.seq_len, f"Sequence length {T} > max {self.config.seq_len}"

        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        attentions = []
        for block in self.transformer.h:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Shift: predict next token from current
            # logits[:, :-1] predicts targets[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                targets[:, 1:].contiguous().view(-1),
                ignore_index=-1
            )

        if return_attention:
            return logits, loss, attentions
        return logits, loss

    def get_hidden_states(self, idx):
        """Extract hidden states from all layers (for analysis)."""
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        hidden_states = [x.detach()]
        for block in self.transformer.h:
            x = block(x)
            hidden_states.append(x.detach())

        return hidden_states

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# RMT (Recurrent Memory Transformer) Extension
# =============================================================================

@dataclass
class RMTConfig(GPTConfig):
    """RMT configuration extends GPT config."""
    n_mem_tokens: int = 1         # Number of memory tokens
    memory_alpha_init: float = 0.5  # Initial EMA weight for memory


class RMTSpacingGPT(SpacingGPT):
    """
    RMT-enhanced SpacingGPT with learnable memory.

    Key idea: Memory token carries compressed history across windows.
    The model learns to:
    1. Read from memory (attention to MEM token)
    2. Write to memory (update based on hidden states)

    Architecture:
    - Input: [MEM, x_1, x_2, ..., x_T]
    - Output: logits for x_2, ..., x_{T+1}
    - Memory update: MEM' = α·MEM + (1-α)·h_last
    """

    def __init__(self, config: RMTConfig):
        # Don't call super().__init__ yet - we need to adjust seq_len first
        nn.Module.__init__(self)
        self.config = config

        # Adjust effective sequence length (MEM tokens take space)
        effective_seq_len = config.seq_len + config.n_mem_tokens

        # Create transformer components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(effective_seq_len, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # === RMT-specific components ===
        # Learnable initial memory token(s)
        self.mem_init = nn.Parameter(
            torch.randn(1, config.n_mem_tokens, config.n_embd) * 0.02
        )

        # Memory update projection (learns optimal summary)
        self.mem_write = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Learnable EMA weight (constrained to [0, 1] via sigmoid)
        self.memory_alpha_logit = nn.Parameter(
            torch.tensor(math.log(config.memory_alpha_init / (1 - config.memory_alpha_init)))
        )

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"RMTSpacingGPT: {n_params/1e6:.2f}M parameters")

    @property
    def memory_alpha(self):
        """Learnable EMA weight constrained to [0, 1]."""
        return torch.sigmoid(self.memory_alpha_logit)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize memory state for a batch."""
        return self.mem_init.expand(batch_size, -1, -1).clone()

    def forward(self, idx, targets=None, memory=None, return_memory=False):
        """
        Forward pass with optional memory.

        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target indices (optional)
            memory: (B, n_mem, n_embd) memory state (optional)
            return_memory: if True, return updated memory

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided
            new_memory: (B, n_mem, n_embd) if return_memory=True
        """
        device = idx.device
        B, T = idx.size()
        n_mem = self.config.n_mem_tokens

        # Initialize memory if not provided
        if memory is None:
            memory = self.init_memory(B, device)

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        # Prepend memory to sequence
        # x = [MEM_1, ..., MEM_k, tok_1, ..., tok_T]
        x = torch.cat([memory, tok_emb], dim=1)  # (B, n_mem + T, n_embd)

        # Position embeddings for full sequence
        total_len = n_mem + T
        pos = torch.arange(0, total_len, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)  # (total_len, n_embd)
        x = self.transformer.drop(x + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Split output: memory positions vs token positions
        mem_hidden = x[:, :n_mem, :]  # (B, n_mem, n_embd)
        tok_hidden = x[:, n_mem:, :]  # (B, T, n_embd)

        # Logits for tokens only
        logits = self.lm_head(tok_hidden)  # (B, T, vocab_size)

        # Update memory using learned projection
        # Take last hidden state of token sequence
        last_hidden = tok_hidden[:, -1:, :]  # (B, 1, n_embd)
        summary = self.mem_write(last_hidden)  # (B, 1, n_embd)

        # EMA update
        alpha = self.memory_alpha
        new_memory = alpha * memory + (1 - alpha) * summary.expand_as(memory)

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                targets[:, 1:].contiguous().view(-1),
                ignore_index=-1
            )

        if return_memory:
            return logits, loss, new_memory
        return logits, loss

    def forward_sequence(self, windows: list, memory=None):
        """
        Process multiple windows with memory carry-over.

        Args:
            windows: list of (B, T) tensors
            memory: initial memory state (optional)

        Returns:
            all_logits: list of logit tensors
            all_losses: list of loss values
            final_memory: memory state after all windows
        """
        device = windows[0].device
        B = windows[0].size(0)

        if memory is None:
            memory = self.init_memory(B, device)

        all_logits = []
        all_losses = []

        for window in windows:
            logits, loss, memory = self.forward(
                window, targets=window, memory=memory, return_memory=True
            )
            all_logits.append(logits)
            if loss is not None:
                all_losses.append(loss)

        return all_logits, all_losses, memory

    @torch.no_grad()
    def generate_with_memory(self, idx, max_new_tokens, memory=None,
                              temperature=1.0, window_size=None):
        """
        Generate tokens with memory updates.

        Args:
            idx: (B, T) initial context
            max_new_tokens: number of tokens to generate
            memory: initial memory state
            temperature: sampling temperature
            window_size: update memory every N tokens (default: seq_len)
        """
        B = idx.size(0)
        device = idx.device

        if memory is None:
            memory = self.init_memory(B, device)
        if window_size is None:
            window_size = self.config.seq_len

        generated = idx.clone()

        for i in range(max_new_tokens):
            # Crop context to fit
            ctx_len = min(generated.size(1), self.config.seq_len)
            idx_cond = generated[:, -ctx_len:]

            # Forward with memory
            logits, _, new_memory = self.forward(
                idx_cond, memory=memory, return_memory=True
            )
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, idx_next], dim=1)

            # Update memory at window boundaries
            if (i + 1) % window_size == 0:
                memory = new_memory

        return generated, memory

    def get_hidden_states(self, idx, memory=None):
        """Extract hidden states from all layers."""
        device = idx.device
        B, T = idx.size()
        n_mem = self.config.n_mem_tokens

        if memory is None:
            memory = self.init_memory(B, device)

        tok_emb = self.transformer.wte(idx)
        x = torch.cat([memory, tok_emb], dim=1)

        total_len = n_mem + T
        pos = torch.arange(0, total_len, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(x + pos_emb)

        hidden_states = [x[:, n_mem:, :].detach()]  # Exclude memory positions
        for block in self.transformer.h:
            x = block(x)
            hidden_states.append(x[:, n_mem:, :].detach())

        return hidden_states
