import torch


# 通用训练配置 (所有模型共享)
TRAIN_CONFIG = {
    'batch_size': 1024,      # 批次大小
    'epochs': 5,             # 训练轮数
    'lr': 0.0005,            # 初始学习率
    'betas': (0.9, 0.999),   # AdamW 参数
    'seq_len': 64,           # 序列长度 (数据加载和模型都需要)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 模型特定架构配置
MODEL_ARCH_CONFIGS = {
    'FNN': {
        'embed_dim': 256,
        'hidden_dim': 1024,
        # seq_len 从 TRAIN_CONFIG 继承
    },
    'RNN': {
        'embed_dim': 256,
        'hidden_dim': 1024,
        # RNN 结构上不需要 seq_len，但 Dataset 需要
    },
    'Transformer': {
        'embed_dim': 256,
        'hidden_dim': 1024,
        'num_heads': 8,
        'layers': 4,
        'dropout': 0.1
    }
}