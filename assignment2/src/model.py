import torch
import torch.nn as nn
import math


class FNN_LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len, window_size=5):
        super().__init__()
        self.seq_len = seq_len
        self.window_size = window_size # 相当于 N-gram 的 N
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 动态计算 Padding
        # 为了实现 Causal Conv (只看过去)，我们需要在左侧补 (window_size - 1) 个 0
        # PyTorch 的 padding 参数是两边都补，所以我们稍后需要手动裁剪右边
        self.pad_size = window_size - 1
        
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=hidden_dim, 
            kernel_size=window_size,
            padding=self.pad_size
        )
        
        self.relu = nn.ReLU()
        
        # 1x1 卷积相当于对每个位置独立做全连接，但共享权重
        self.fc_out = nn.Conv1d(hidden_dim, vocab_size, kernel_size=1)

    def forward(self, x):     
        # 1. Embedding
        # [batch, seq_len, embed] -> [batch, embed, seq_len] (Conv1d 需要通道在中间)
        embeds = self.embedding(x).transpose(1, 2)
        
        # 2. Convolution
        h = self.conv1(embeds)
        h = self.relu(h)
        
        # 3. Causal Padding 处理
        if self.pad_size > 0:
            h = h[:, :, :-self.pad_size] # 截掉多余的 padding
        logits = self.fc_out(h)

        return logits.transpose(1, 2)
    
class RNN_LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        output, _ = self.lstm(embeds)
        logits = self.fc(output)
        return logits

class Transformer_LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, layers=2, dropout=0.01):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        seq_len = x.size(1)
        embeds = self.embedding(x) * math.sqrt(self.embed_dim)
        embeds = embeds + self.pos_encoder[:,:seq_len,:]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        output = self.transformer_encoder(embeds, mask=mask, is_causal=True)
        logits = self.fc(output)
        return logits