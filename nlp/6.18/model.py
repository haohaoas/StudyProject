import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_dim, batch_first=True)

    def forward(self, src_idx_tensor):
        vectors = self.embedding(src_idx_tensor)
        outputs, hidden = self.rnn(vectors)      # outputs: [B, T, H], hidden: [1, B, H]
        return outputs, hidden                   # 返回所有步的 hidden 以供注意力

class QKVCrossAttention(nn.Module):
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or dec_hidden_dim
        self.W_q = nn.Linear(dec_hidden_dim, attn_dim)
        self.W_k = nn.Linear(enc_hidden_dim, attn_dim)
        self.W_v = nn.Linear(enc_hidden_dim, attn_dim)
        self.scale = math.sqrt(attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [B, 1, H_dec]   encoder_outputs: [B, T, H_enc]
        Q = self.W_q(decoder_hidden)         # [B, 1, D]
        K = self.W_k(encoder_outputs)        # [B, T, D]
        V = self.W_v(encoder_outputs)        # [B, T, D]
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale   # [B, 1, T]
        attn_weights = torch.softmax(scores, dim=-1)               # [B, 1, T]
        context = torch.matmul(attn_weights, V)                    # [B, 1, D]
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_size, hidden_dim, enc_hidden_dim=None, attn_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(target_vocab_size, embedding_size, padding_idx=0, max_norm=1)
        self.attn = QKVCrossAttention(hidden_dim, enc_hidden_dim or hidden_dim, attn_dim)
        self.rnn = nn.GRU(embedding_size + (attn_dim or hidden_dim), hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, encoder_outputs, encoder_hidden, target_seq=None, teacher_forcing_ratio=0.5):
        # encoder_outputs: [B, T, H_enc]
        # encoder_hidden:  [1, B, H_enc]
        batch_size = encoder_outputs.size(0)
        max_len = target_seq.size(1)
        inputs = target_seq[:, 0].unsqueeze(1)
        hidden = encoder_hidden          # 让 decoder 初始化为 encoder 最后 hidden
        outputs = []
        for t in range(max_len):
            embedded = self.embedding(inputs)          # [B, 1, E]
            # Q = decoder当前hidden; K,V=encoder所有hidden
            context, attn_weights = self.attn(hidden.transpose(0,1), encoder_outputs)   # context: [B, 1, D]
            rnn_input = torch.cat([embedded, context], dim=-1)#embedded:当前decoder的输入token context 当前encoder输出中经attention得到的上下文向量
            output, hidden = self.rnn(rnn_input, hidden)
            logit = self.fc(output)                                    # [B, 1, vocab]
            outputs.append(logit)
            top1 = logit.argmax(-1)                                    # [B, 1]
            if t + 1 < max_len and target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                inputs = target_seq[:, t + 1].unsqueeze(1)
            else:
                inputs = top1
        return torch.cat(outputs, dim=1)   # [B, max_len, vocab]

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_idx_tensor, target_seq=None, teacher_forcing_ratio=0.5):
        encoder_outputs, encoder_hidden = self.encoder(src_idx_tensor)
        decoder_output = self.decoder(
            encoder_outputs, encoder_hidden, target_seq, teacher_forcing_ratio
        )
        return decoder_output