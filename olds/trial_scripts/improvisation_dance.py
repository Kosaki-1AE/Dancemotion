import torch
import torch.nn as nn
import torch.nn.functional as F


class StillnessAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class ResponsibilityScorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        # 観測スコア（責任）を計算
        return torch.sigmoid(self.fc(x))  # 値域：0～1

class FeedForwardResponse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class StillnessAI(nn.Module):
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.attention = StillnessAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.responsibility = ResponsibilityScorer(dim)
        self.ff = FeedForwardResponse(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        attn = self.attention(x)
        x = self.norm1(x + attn)

        R = self.responsibility(x)  # 責任スコア
        ff = self.ff(x)
        output = self.norm2(x + ff)

        return output, R
