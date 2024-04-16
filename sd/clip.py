import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from sd.attention import SelfAttention

class CLIPEMbedding(nn.Module):

    def __init__(self, n_vocab, n_emb, n_tokens):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_emb)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_emb))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x
    
class CLIPLayer(nn.Module):
    def __init(self, n_head, n_emb):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_emb)
        self.attention = SelfAttention(n_head, n_emb)
        self.layernorm_2 = nn.LayerNorm(n_emb)
        self.linear_1 = nn.Linear(n_emb, 4* n_emb)
        self.linear_2 = nn.Linear(4 * n_emb, n_emb)

    def forward(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask=True)
        x += residue

        residue = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens):
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output