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
        self.device = device if device is not None and torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.weights: Float[Tensor, "d_out d_in"] = self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
    
    def _initialize_weights(self) -> torch.Tensor:
        weights = torch.empty((self.out_features, self.in_features), dtype=self.dtype, device=self.device)
        std = 2/(self.in_features+self.out_features)
        nn.init.trunc_normal_(weights, mean=0, std=std, a=-3*std, b=3*std)       
        return nn.Parameter(weights)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embedddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device if device is not None and torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.embedding: Float[Tensor, "vocab_size, d_model"] = self._initialize_embeddings()
   
    def _initialize_embeddings(self) -> torch.Tensor:
        embedding = torch.empty((self.num_embedddings, self.embedding_dim), dtype=self.dtype, device=self.device)
        nn.init.trunc_normal_(embedding, a=-3, b=3)       
        return nn.Parameter(embedding)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device if device is not None and torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.gain: Float[Tensor, "d_model"] = nn.Parameter(torch.ones(d_model, device=self.device))
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # reduce is much slower
        #result = x * self.gain / torch.sqrt_(self.eps + reduce(x**2, "b s d -> b s ()", reduction="sum") / self.d_model)
        result = x * self.gain / torch.sqrt_(self.eps + torch.sum(x**2, dim=-1)/self.d_model).unsqueeze(-1)
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
        self.device = device if device is not None and torch.cuda.is_available() else "cpu"
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

        self.cos_thetas = cos_thetas.to(self.device)
        self.sin_thetas = sin_thetas.to(self.device)
        
        # i could repeat and interleave it here or in the forward method storage vs speed tradeoff
        # cos_thetas: Float[Tensor, "max_seq_len, d_k"] = cos_thetas.repeat_interleave(repeats=2, dim=-1)
        # sin_thetas: Float[Tensor, "max_seq_len, d_k"] = sin_thetas.repeat_interleave(repeats=2, dim=-1)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.clamp(0, self.max_seq_len - 1)
        # _, seq_len, d_k = x.shape # x.shape -> b, seq_len, d_model
        # x2 = x.clone()
        # x2[..., 0::2] = -x[..., 1::2]
        # x2[..., 1::2] = x[..., 0::2]        
        # result = (x * self.cos_thetas[token_positions].repeat_interleave(repeats=2, dim=-1) + 
        #           x2 * self.sin_thetas[token_positions].repeat_interleave(repeats=2, dim=-1))

        x_evens, x_odds = x[..., ::2], x[..., 1::2]
        rotated_evens = x_evens * self.cos_thetas[token_positions] - x_odds * self.sin_thetas[token_positions]
        rotated_odds = x_evens * self.sin_thetas[token_positions] + x_odds * self.cos_thetas[token_positions]

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
    attention_weights = einsum(queries, keys, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k**0.5)
    if mask is not None:
        attention_weights = torch.masked_fill(attention_weights, ~mask, -torch.inf) # mask value that is False in attn_weights
    attention_scores = softmax(attention_weights, dim=-1)
    output = einsum(attention_scores, values, "... queries keys, ... keys d_v -> ... queries d_v")
    return output
    


    
