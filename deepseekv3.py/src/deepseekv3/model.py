import math
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepSeekV3Config:
    vocab_size: int = 50259  # GPT-2 vocabulary size + <|story|> + </|story|> tokens
    n_layer: int = 6  # Number of transformer blocks
    n_head: int = 8  # Number of attention heads
    n_embd: int = 256  # Embedding dimension
    block_size: int = 1024  # Maximum context window
    dropout: float = 0.1  # Dropout rate
    bias: bool = True  # Use bias in linear layers
    # MLA (Multihead Latent Attention) config
    kv_lora_rank: int = 128  # LoRA rank for key-value projection
    q_lora_rank: int = 192  # LoRA rank for query projection
    nope_dim: int = 32  # Content (no-position) dim per head
    rope_dim: int = 64  # RoPE dim per head
    # MoE (Mixture of Experts) config
    n_experts: int = 4  # Number of experts
    n_experts_per_token: int = 2  # Number of experts per token (top-k)
    expert_intermediate_size: int = 512  # Expert hidden size
    shared_expert_intermediate_size: int = 768  # Shared expert hidden size
    use_shared_expert: bool = True  # Enable shared expert
    aux_loss_weight: float = 0.0  # Auxiliary loss weight (0.0 for aux-free)
    # Multi-token prediction
    multi_token_predict: int = 2  # Number of MTP modules (predict t+2, t+3, ...)
    mtp_loss_weight: float = 0.3  # Weight of MTP loss vs main LM loss


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        # x: (..., D), weight: (D,)
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)  # (..., 1)
        return self.weight * x / (norm + self.eps)  # (..., D)


class RoPE(torch.nn.Module):
    """Rotary Positional Embedding (RoPE) for better position understanding"""

    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        # x: (..., T, D), inv_freq: (D/2,)
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)  # (T,)
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        cos, sin = freqs.cos(), freqs.sin()  # each (T, D/2)
        return cos, sin

    @staticmethod
    def apply(x, cos, sin):
        """Apply rotary position embedding"""
        # x: (..., T, D), cos/sin: (T, D/2)
        x1, x2 = x.chunk(2, dim=-1)  # (..., T, D/2)
        return torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )  # (..., T, D)


class MultiheadLatentAttention(torch.nn.Module):
    """
    Multihead Latent Attention (MLA) - DeepSeek's efficient attention mechanism
    Key innovations:
    - Compression/decompression of queries and key-values
    - LoRA-style low-rank projections for efficiency
    - RoPE with separate content and positional components
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.nope_dim = config.nope_dim
        # Compression dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_dim = config.rope_dim

        # KV compression / decompression
        self.kv_proj = torch.nn.Linear(self.n_embd, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.k_decompress = torch.nn.Linear(
            self.kv_lora_rank, self.n_head * self.nope_dim, bias=False
        )
        self.v_decompress = torch.nn.Linear(
            self.kv_lora_rank, self.n_head * self.nope_dim, bias=False
        )
        # Query compression / decompression
        self.q_proj = torch.nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
        self.q_decompress = torch.nn.Linear(
            self.q_lora_rank, self.n_head * self.nope_dim, bias=False
        )
        # RoPE projections
        self.k_rope_proj = torch.nn.Linear(
            self.n_embd, self.n_head * self.rope_dim, bias=False
        )
        self.q_rope_proj = torch.nn.Linear(
            self.q_lora_rank, self.n_head * self.rope_dim, bias=False
        )
        # Output projection
        self.o_proj = torch.nn.Linear(
            self.n_head * self.nope_dim, self.n_embd, bias=config.bias
        )
        # Dropout
        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)
        # RoPE
        self.rope = RoPE(self.rope_dim, config.block_size)
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.size()
        # Compression phase
        kv_compressed = self.kv_norm(self.kv_proj(x))
        q_compressed = self.q_proj(x)
        # Decompression phase
        k_content = self.k_decompress(kv_compressed)
        v = self.v_decompress(kv_compressed)
        q_content = self.q_decompress(q_compressed)
        # RoPE components
        k_rope = self.k_rope_proj(x)
        q_rope = self.q_rope_proj(q_compressed)
        # Reshape [B, H, T, nope_dim] / [B, H, T, rope_dim] for multi-head attention
        k_content = k_content.view(B, T, self.n_head, self.nope_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.nope_dim).transpose(1, 2)
        q_content = q_content.view(B, T, self.n_head, self.nope_dim).transpose(1, 2)
        k_rope = k_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
        q_rope = q_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
        # Apply RoPE
        cos, sin = self.rope(x, T)
        q_rope = RoPE.apply(q_rope, cos, sin)
        k_rope = RoPE.apply(k_rope, cos, sin)
        # Concatenate content and rope parts
        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)

        # Attention computation
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        # Apply padding mask if provided
        if attention_mask is not None:
            padding_mask_additive = (1 - attention_mask).unsqueeze(1).unsqueeze(
                2
            ) * float("-inf")
            scores = scores + padding_mask_additive
        # Softmax and dropout
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        # attn_weights: (B, H, T, T), v: (B, H, T, nope_dim)
        out = torch.matmul(attn_weights, v)  # (B, H, T, nope_dim)
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.nope_dim)
        out = self.resid_dropout(self.o_proj(out))
        return out


class SwiGLU(torch.nn.Module):
    """SwiGLU activation function used in DeepSeek experts"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, bias: bool = True
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down_proj = torch.nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor):
        gate = torch.nn.functional.silu(self.gate_proj(x))  # SiLU activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Expert(torch.nn.Module):
    """Expert network for Mixture of Experts using SwiGLU"""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.expert_mlp = SwiGLU(
            config.n_embd, config.expert_intermediate_size, config.n_embd, config.bias
        )

    def forward(self, x: torch.Tensor):
        return self.expert_mlp(x)


class MixtureOfExperts(torch.nn.Module):
    """
    DeepSeek MoE layer with shared expert and auxiliary-loss-free load balancing

    Key features:
    - Shared expert that processes all tokens
    - Auxiliary-loss-free load balancing via bias updates
    - Top-k routing to selected experts
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.n_embd = config.n_embd
        # Router: learns which experts to use for each token
        self.router = torch.nn.Linear(config.n_embd, config.n_experts, bias=False)
        # Expert networks
        self.experts = torch.nn.ModuleList(
            [Expert(config) for _ in range(config.n_experts)]
        )
        # Shared expert (processes all tokens)
        if config.use_shared_expert:
            self.shared_expert = SwiGLU(
                config.n_embd,
                config.shared_expert_intermediate_size,
                config.n_embd,
                config.bias,
            )
        else:
            self.shared_expert = None
        # Auxiliary-loss-free load balancing
        self.register_buffer("expert_bias", torch.zeros(config.n_experts))
        self.bias_update_rate = 0.001
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.shape
        x = x.view(-1, n_embd)
        # Routing phase with bias for load balancing
        router_logits = self.router(x) + self.expert_bias
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = torch.zeros_like(router_logits)
        routing_weights.scatter_(
            -1, top_k_indices, torch.nn.functional.softmax(top_k_logits, dim=-1)
        )
        # Expert computation
        output = torch.zeros_like(x)
        expert_usage = torch.zeros(self.n_experts, device=x.device)

        # Process through selected experts
        for expert_idx in range(self.n_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (B*T,) bool
            expert_usage[expert_idx] = expert_mask.sum().float()
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                # Weight by routing probability
                weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                output[expert_mask] += expert_output * weights

        # Add shared expert output (processes all tokens)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            output += shared_output

        # Auxiliary-loss-free load balancing (update biases during training)
        if self.training:
            with torch.no_grad():
                avg_usage = expert_usage.mean()
                for i in range(self.n_experts):
                    if expert_usage[i] > avg_usage:
                        self.expert_bias[i] -= self.bias_update_rate
                    else:
                        self.expert_bias[i] += self.bias_update_rate

        output = self.dropout(output)
        return output.view(B, T, n_embd), router_logits.view(B, T, -1)

    def _complementary_sequence_aux_loss(self, router_logits, seq_mask=None):
        """
        router_logits: [B, T, num_experts]
            Raw logits from the router before softmax.
        seq_mask: optional mask for padding tokens.
        """
        # Convert to probabilities
        probs = torch.nn.functional.softmax(router_logits, dim=-1)  # [B, T, E]

        # Aggregate per-sequence expert usage
        if seq_mask is not None:
            probs = probs * seq_mask.unsqueeze(-1)  # mask padding

        seq_usage = probs.sum(dim=1)  # [B, E]
        # Normalize per sequence
        seq_usage = seq_usage / seq_usage.sum(dim=-1, keepdim=True)

        # Compute pairwise similarity between sequences
        sim_matrix = torch.matmul(seq_usage, seq_usage.transpose(0, 1))  # [B, B]

        # Encourage complementarity: minimize similarity off-diagonal
        batch_size = seq_usage.size(0)
        off_diag = sim_matrix - torch.eye(batch_size, device=sim_matrix.device)
        loss = off_diag.mean()
        return loss


class MultiTokenPredictionHead(torch.nn.Module):
    """
    Multi-Token Prediction Head
    Each head predicts a token at a specific future position.
    Combines previous hidden state with future token embedding.
    """

    def __init__(self, config: DeepSeekV3Config, depth: int):
        super().__init__()
        self.depth = depth
        self.n_embd = config.n_embd
        # Combine previous hidden state with future token embedding
        self.combine_proj = torch.nn.Linear(
            2 * config.n_embd, config.n_embd, bias=config.bias
        )
        # Normalization
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        # Transformer components (mini-transformer for each head)
        self.attn = MultiheadLatentAttention(config)
        self.mlp = MixtureOfExperts(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, prev_hidden, future_token_embed, attention_mask=None):
        # prev_hidden, future_token_embed: (B, T, n_embd)
        prev_norm = self.norm1(prev_hidden)
        future_norm = self.norm2(future_token_embed)
        combined = torch.cat([prev_norm, future_norm], dim=-1)  # (B, T, 2*n_embd)
        hidden = self.combine_proj(combined)  # (B, T, n_embd)
        hidden = hidden + self.attn(self.attn_norm(hidden), attention_mask)
        moe_out, _ = self.mlp(self.mlp_norm(hidden))
        hidden = hidden + moe_out
        return hidden


class TransformerBlock(torch.nn.Module):
    """Pre-norm block: RMSNorm → MLA → residual, RMSNorm → MoE → residual."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = MultiheadLatentAttention(config)
        self.moe_norm = RMSNorm(config.n_embd)
        self.moe = MixtureOfExperts(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x: (B, T, n_embd)
        x = x + self.attn(self.attn_norm(x), attention_mask)
        moe_out, router_logits = self.moe(
            self.moe_norm(x)
        )  # router_logits: (B, T, n_experts)
        x = x + moe_out
        return x, router_logits


class DeepSeekV3(torch.nn.Module):
    """
    Token embed → N × (MLA + MoE) → RMSNorm → LM head
    plus a chain of MTP modules on the backbone hidden states.
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = torch.nn.Dropout(config.dropout)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tied embeddings
        self.mtp_heads = torch.nn.ModuleList(
            [
                MultiTokenPredictionHead(config, depth=i + 1)
                for i in range(config.multi_token_predict)
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # tokens: (B, T) token ids
        B, T = tokens.size()
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        tok_emb = self.token_embedding(tokens)  # (B, T, n_embd)
        x = self.drop(tok_emb)

        router_logits_list = []
        for block in self.blocks:
            x, router_logits = block(x, attention_mask)
            router_logits_list.append(router_logits)

        hidden = self.norm(x)  # (B, T, n_embd)
        logits = self.lm_head(hidden)  # (B, T, vocab) — position t predicts token t+1

        if targets is None:
            return logits

        # Main next-token loss. Pass targets=tokens to use the usual shift.
        main_loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
            targets[:, 1:].contiguous().view(-1),
        )

        mtp_loss = torch.zeros((), device=tokens.device)
        h_mtp = x  # backbone residual, before final norm
        for depth, mtp_head in enumerate(self.mtp_heads, start=1):
            if h_mtp.size(1) < 2:
                break
            h_mtp = h_mtp[:, :-1, :]  # (B, T-depth, n_embd)
            fut = tok_emb[:, depth : depth + h_mtp.size(1), :]  # embed(x_{t+depth})
            mtp_mask = (
                attention_mask[:, : h_mtp.size(1)]
                if attention_mask is not None
                else None
            )
            h_mtp = mtp_head(h_mtp, fut, mtp_mask)
            mtp_logits = self.lm_head(
                self.norm(h_mtp)
            )  # position t predicts x_{t+depth+1}
            mtp_loss = mtp_loss + torch.nn.functional.cross_entropy(
                mtp_logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                targets[:, depth + 1 :].contiguous().view(-1),
            )

        n_mtp = max(len(self.mtp_heads), 1)
        loss = main_loss + self.config.mtp_loss_weight * mtp_loss / n_mtp

        if self.config.aux_loss_weight > 0:
            aux = torch.zeros((), device=tokens.device)
            for router_logits in router_logits_list:
                aux = aux + self.blocks[0].mlp._complementary_sequence_aux_loss(
                    router_logits, attention_mask
                )
            loss = loss + self.config.aux_loss_weight * aux / len(self.blocks)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """Autoregressive decode with the main LM head only."""
        for _ in range(max_new_tokens):
            logits = self(
                tokens
                if tokens.size(1) <= self.config.block_size
                else tokens[:, -self.config.block_size :]
            )
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
