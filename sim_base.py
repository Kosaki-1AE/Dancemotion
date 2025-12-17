import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Positional Encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ----------------------------
# Masks
# ----------------------------
def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    """
    Causal mask for decoder self-attn.
    Returns (sz, sz) where upper triangle is -inf.
    """
    device = device or torch.device("cpu")
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

def make_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    tokens: (batch, seq_len)
    returns: (batch, seq_len) True where PAD
    """
    return tokens.eq(pad_id)


# ----------------------------
# Full Transformer (Embedding -> Encoder -> Decoder -> Linear)
# ----------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        if share_embeddings and (src_vocab_size == tgt_vocab_size):
            self.tgt_tok_emb = self.src_tok_emb
        else:
            self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Core Transformer
        # batch_first=True にして (batch, seq, d_model) で統一
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        # Output head
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Optional: tie output with target embedding (よくやる)
        # 条件: 同一語彙サイズ & embedding共有したい場合
        self.tie_output = False
        if share_embeddings and (src_vocab_size == tgt_vocab_size):
            self.tie_output = True
            self.generator.weight = self.tgt_tok_emb.weight

    def encode(self, src_tokens: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        # (batch, src_len) -> (batch, src_len, d_model)
        src = self.src_tok_emb(src_tokens) * math.sqrt(self.d_model)
        src = self.pos_enc(src)
        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )
        return memory

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ):
        tgt = self.tgt_tok_emb(tgt_tokens) * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)
        out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor):
        """
        src_tokens: (batch, src_len)
        tgt_tokens: (batch, tgt_len)  ※ teacher forcing用(入力側)
        returns logits: (batch, tgt_len, tgt_vocab_size)
        """
        device = src_tokens.device

        src_pad = make_padding_mask(src_tokens, self.pad_id)  # (batch, src_len)
        tgt_pad = make_padding_mask(tgt_tokens, self.pad_id)  # (batch, tgt_len)

        tgt_len = tgt_tokens.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_len, device=device)  # (tgt_len, tgt_len)

        memory = self.encode(src_tokens, src_key_padding_mask=src_pad)
        dec_out = self.decode(
            tgt_tokens,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )

        logits = self.generator(dec_out)  # (batch, tgt_len, vocab)
        return logits


# ----------------------------
# Quick sanity check
# ----------------------------
if __name__ == "__main__":
    B, src_len, tgt_len = 2, 7, 6
    src_vocab, tgt_vocab = 1000, 1200
    pad_id = 0

    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=pad_id,
        share_embeddings=False,
    )

    src = torch.randint(1, src_vocab, (B, src_len))
    tgt_in = torch.randint(1, tgt_vocab, (B, tgt_len))

    # 例: 末尾だけPADにしてみる
    src[:, -1] = pad_id
    tgt_in[:, -1] = pad_id

    logits = model(src, tgt_in) 
    print("logits:", logits.shape)  # (B, tgt_len, tgt_vocab)　　　