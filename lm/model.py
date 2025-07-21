from matplotlib.style import context
import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Bool
from torch import Tensor

# Linear Layer
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Float[Tensor, "d_out d_in"] = self._initialize_weights()
    
    def _initialize_weights(self) -> torch.Tensor:
        weights = torch.empty((self.out_features, self.in_features))
        std = 2/(self.in_features+self.out_features)
        nn.init.trunc_normal_(weights, mean=0, std=std, a=-3*std, b=3*std)       
        return nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embedddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight: Float[Tensor, "vocab_size, d_model"] = self._initialize_embeddings()
   
    def _initialize_embeddings(self) -> torch.Tensor:
        embedding = torch.empty((self.num_embedddings, self.embedding_dim))
        nn.init.trunc_normal_(embedding, a=-3, b=3)       
        return nn.Parameter(embedding)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight: Float[Tensor, "d_model"] = nn.Parameter(torch.ones(d_model))
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x * self.weight / (self.eps + torch.sqrt_(torch.sum(x**2, dim=-1)/self.d_model).unsqueeze(-1))
        return result.to(in_dtype)

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = round(self.d_model / 24) * 64 if d_ff is None else d_ff # closest multiple of 64 to 8/3 d_model. simplified round((x * (8 / 3)) / 64) * 64
        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff)
    
    def forward(self, x: Tensor):
        x1 = self.w1(x)
        result = self.w2(self.w3(x) * torch.sigmoid(x1) * x1)
        return result

class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer(name="cos_thetas", tensor=torch.Tensor([]), persistent=False)
        self.register_buffer(name="sin_thetas", tensor=torch.Tensor([]), persistent=False)
        self._initialize_rope_values()            
    
    def _initialize_rope_values(self) -> None:
        positions = torch.arange(0, self.max_seq_len).reshape(-1, 1)
        dim_indices = torch.arange(0, self.d_k//2)
        inv_frequencies = 1 / self.theta**(2*dim_indices/self.d_k) # gives a row vector of 1/theta**(2k/d) 
        angles = positions * inv_frequencies
        
        cos_thetas: Float[Tensor, "max_seq_len, d_k//2"] = torch.cos(angles)
        sin_thetas: Float[Tensor, "max_seq_len, d_k//2"] = torch.sin(angles)

        self.cos_thetas = cos_thetas
        self.sin_thetas = sin_thetas
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.clamp(0, self.max_seq_len - 1)
        cos = self.cos_thetas[token_positions]
        sin = self.sin_thetas[token_positions]

        x_evens, x_odds = x[..., ::2], x[..., 1::2]
        rotated_evens = x_evens * cos - x_odds * sin
        rotated_odds = x_evens * sin + x_odds * cos

        output = torch.empty_like(x)
        output[...,::2] = rotated_evens
        output[..., 1::2] = rotated_odds

        return output
    
def softmax(x: torch.Tensor, dim: int=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True) + 1e-8

def scaled_dot_product_attention(queries: Float[Tensor, "... queries d_k"],
                                 keys: Float[Tensor, "... keys d_k"],
                                 values: Float[Tensor, "... values d_v"],
                                 mask: Bool[Tensor, "queries seq_len"] | None=None) -> Float[Tensor, "... d_v"]:
    
    d_k = keys.shape[-1]
    attention_weights = einsum(queries, keys, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k**0.5) # queries @ keys.transpose(-1,-2)
    if mask is not None:
        attention_weights = torch.masked_fill(attention_weights, ~mask, -torch.inf) # mask value that is False in attn_weights
    attention_scores = softmax(attention_weights, dim=-1)
    output = einsum(attention_scores, values, "... queries keys, ... keys d_v -> ... queries d_v") # @ attn_scores @ 
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float | None=None, max_seq_len: int | None = None, rope_object: ROPE | None = None ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, "d_model has to be divisible by numheads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(in_features=self.d_model, out_features=3*self.d_k*self.num_heads)
        self.output_proj = Linear(in_features=self.d_k*self.num_heads, out_features=self.d_model)
        
        self.rope = rope_object if rope_object is not None else None
        if rope_theta is not None or rope_object is not None:
            assert self.d_k % 2 == 0, "Head dimension (d_k) must be even for RoPE"
        if self.rope is None and rope_theta is not None:
            self.max_seq_len = max_seq_len if max_seq_len is not None else 64000
            self.rope = ROPE(rope_theta, d_k=self.d_k, max_seq_len=self.max_seq_len)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        _, seq_len, _ = x.shape
        qkv = self.qkv_proj(x) # [batch, seq_len, 3 * num_heads * d_k]
        q, k, v = rearrange(qkv, "batch seq_len (three num_heads d_k) -> three batch num_heads seq_len d_k", 
                             three=3, num_heads=self.num_heads, d_k=self.d_k)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        if not hasattr(self, '_mask') or self._mask.size(0) < seq_len:
            self._mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        mask = self._mask[:seq_len, :seq_len]
        attention_ouput_bhsd = scaled_dot_product_attention(queries=q, keys=k, values=v, mask=mask)
        attention_ouput_bsd = rearrange(attention_ouput_bhsd, "... num_heads seq_len d_model -> ... seq_len (num_heads d_model)")
        final_attention_output = self.output_proj(attention_ouput_bsd)

        return final_attention_output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, rope_object: ROPE | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, rope_theta=rope_theta, rope_object=rope_object)
        self.ln2 = RMSNorm(d_model=d_model)        
        self.ffn = SwiGLUFeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        if token_positions is None:
            token_positions = torch.arange(0, x.shape[-2])
        residual = x
        x = self.attn(self.ln1(x), token_positions) + residual
        residual = x
        output = self.ffn(self.ln2(x)) + residual
        return output

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = rope_theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        d_k = d_model // num_heads

        self.rope = ROPE(theta=self.theta, d_k=d_k, max_seq_len=context_length)
        self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, rope_theta=rope_theta, rope_object=self.rope) 
                                    for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=self.d_model, out_features=self.vocab_size)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        
        #x = x[..., -self.context_length::]
        x = self.token_embeddings(x)
        
        for layer in self.layers:
            x = layer(x, token_positions)    
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits