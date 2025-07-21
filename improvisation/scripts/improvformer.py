import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# 🔄 Attention可視化用の保存リスト
attention_maps = []

# ✅ Attention Hook 登録関数
def get_attention_hook(name):
    def hook(module, input, output):
        attn_weights = output[1]  # output: (attn_output, attn_weights)
        attention_maps.append((name, attn_weights.detach().cpu()))
    return hook

def _register_attention_hooks(self):
    self.attention_maps = []
    def hook_fn(module, input, output):
        # output: (attn_output, attn_weights) → PyTorch2.0以降対応
        if isinstance(output, tuple) and len(output) == 2:
            self.attention_maps.append(output[1].detach().cpu())
    for layer in self.layers:
        layer.self_attn.register_forward_hook(hook_fn)
        
# ======================
# モデル定義（ImprovFormer）
# ======================
class StillnessPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(StillnessPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
        self._register_attention_hooks()

    def forward(self, x, stillness=None):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        # Stillnessスコアのブーストが来た場合（未使用ならNoneで無視）
        if stillness is not None:
            x = x + stillness  # shape: [B, T, d_model]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, use_stillness=False):
        super().__init__()
        self.use_stillness = use_stillness
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, stillness=None):
        x = x + self.pe[:, :x.size(1), :]
        if self.use_stillness and stillness is not None:
            x = x + stillness
        return x

class SelfAttentionWithHook(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_weights = None

    def forward(self, x):
        out, weights = self.attn(x, x, x, need_weights=True)
        self.attn_weights = weights.detach()
        return out

class ImprovFormer(nn.Module):
    def __init__(self, input_dim, output_dim, num_meaning_classes=4, num_heads=2, num_layers=2, dim_feedforward=256, use_stillness=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward, use_stillness=use_stillness)
        self.pos_encoding = PositionalEncoding(output_dim)

        self.attn1 = SelfAttentionWithHook(output_dim, num_heads)
        self.attn2 = SelfAttentionWithHook(output_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim_feedforward, nhead=num_heads, dim_feedforward=dim_feedforward,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(dim_feedforward, output_dim)
        self.meaning_classifier = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_meaning_classes)
        )
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        self.attention_maps = []
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                self.attention_maps.append(output[1].detach().cpu())
        for layer in self.layers:
            layer.self_attn.register_forward_hook(hook_fn)

    def forward(self, x, stillness=None, return_attention=False):
        self.attention_maps.clear()
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.ffn(x)
        x = self.pos_encoder(x, stillness)
        for layer in self.layers:
            x = layer(x)
        out = self.output_proj(x)
        cls_token = out[:, -1, :]
        meaning_logits = self.meaning_classifier(cls_token)
        if return_attention:
            return out, meaning_logits, self.attention_maps
        else:
            return out, meaning_logits

class MeaningfulImprovFormer(ImprovFormer):
    def __init__(self, input_dim, output_dim, num_meaning_classes, num_heads=2, num_layers=2, dim_feedforward=256, max_len=5000):
        super().__init__(input_dim, output_dim, num_meaning_classes, num_heads, num_layers, dim_feedforward)
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        self.pos_encoder = StillnessPositionalEncoding(dim_feedforward, max_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim_feedforward, nhead=num_heads, dim_feedforward=dim_feedforward,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(dim_feedforward, output_dim)
        
        self.meaning_classifier = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_meaning_classes)
        )

    def forward(self, x, stillness=None, return_attention=False):
        x = self.input_proj(x)
        x = self.pos_encoder(x, stillness)

        self.attentions = []

        for layer in self.layers:
            # ここでhookを入れて attention weight を取るようにしてもOK
            x = layer(x)

        out = self.output_proj(x)               # [B, T, output_dim]
        cls_token = out[:, -1, :]               # 最終時刻の出力
        meaning_logits = self.meaning_classifier(cls_token)

        if return_attention:
            return out, meaning_logits, self.attentions  # ここで attention を返す想定（hook利用時）
        return out, meaning_logits

class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_weights = None

    def forward(self, x):
        # Self-attention層を明示的に分けて処理
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=True)
        self.attn_weights = attn_weights  # [B, num_heads, T, T] or [B, T, T] depending on pytorch version
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x
    
# ======================
# 推論関数
# ======================
def predict(model, input_seq):
    """
    推論関数
    Args:
        model (nn.Module): 学習済みImprovFormerモデル
        input_seq (ndarray or tensor): [seq_len, input_dim]
    Returns:
        prediction (ndarray): [seq_len, output_dim]
    """
    model.eval()
    with torch.no_grad():
        if isinstance(input_seq, np.ndarray):
            input_seq = torch.tensor(input_seq, dtype=torch.float32)
        input_seq = input_seq.unsqueeze(0)  # [1, seq_len, input_dim]
        output = model(input_seq)
        return output.squeeze(0).cpu().numpy()

def visualize_attention(attention, layer_idx=0, head_idx=0, title="Self-Attention"):
    """
    attention: shape [B, num_heads, T, T]
    """
    attn = attention[layer_idx].detach().cpu().numpy()  # [B, H, T, T]
    attn_head = attn[0, head_idx]  # 先頭バッチ・指定ヘッド

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_head, cmap='viridis')
    plt.colorbar()
    plt.title(f"{title} - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.show()

# ======================
# テスト実行例（省略可）
# ======================
if __name__ == '__main__':
    input_dim = 51
    output_dim = 51

    model = ImprovFormer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_meaning_classes=4,
        num_heads=2,
        num_layers=2,
        dim_feedforward=256
    )
    model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))

    dummy_input = np.random(30, input_dim)
    result = predict(model, dummy_input)
    print("Predicted shape:", result.shape)
    __all__ = ['ImprovFormer', 'MeaningfulImprovFormer', 'predict', 'visualize_attention']


