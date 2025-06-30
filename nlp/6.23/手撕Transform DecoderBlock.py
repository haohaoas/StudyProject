import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn=nn.MultiheadAttention(embedding_dim,num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embedding_dim)
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, src_key_padding_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention
        self_attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # 2. Cross-Attention
        cross_attn_output, _ = self.cross_attn(x, encoder_outputs, encoder_outputs,
                                               key_padding_mask=src_key_padding_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        # 3. FeedForward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim, num_heads, ffn_hidden_dim, dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_outputs, src_key_padding_mask=None, tgt_mask=None):
        for block in self.blocks:
            x = block(x, encoder_outputs, src_key_padding_mask, tgt_mask)
        return x
