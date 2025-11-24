import os
import json
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from path import *


def merge_config(common_config, arch_config):
    """
    合并通用配置和架构配置，返回一个新的字典
    """
    config = common_config.copy()
    config.update(arch_config)
    return config

def split_dataset(source_file, train_file, valid_file, valid_lines=1000):
    print(f"Reading source file: {source_file} ...")
    
    if not os.path.exists(source_file):
        print(f"Error: can't find source file {source_file}")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines}")
    
    if total_lines <= valid_lines:
        raise ValueError("The amount of data is too small to create a validation set.")

    # 切分数据
    train_data = lines[:-valid_lines]
    valid_data = lines[-valid_lines:]
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    print(f"Save train data to: {train_file} (Number of lines: {len(train_data)})")
    
    # 写入验证集
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_data)
    print(f"Save valid data to: {valid_file} (Number of lines: {len(valid_data)})")

class RealTextDataset(Dataset):
    def __init__(self, file_path, vocab=None, seq_len=20, is_train=True):
        self.seq_len = seq_len
        
        # 1. 读取并清洗文本
        with open(file_path, 'r', encoding='utf-8') as f:
            # 简单清洗：去除换行符，将所有行拼接成一个长字符串
            # 对于语言模型，通常我们将整个语料视为一个长流
            text = f.read().replace('\n', '')
        
        self.data_chars = list(text) # 将字符串转为字符列表 ['今', '天', ...]
        total_chars = len(self.data_chars)
        print(f"Loaded {file_path}: {total_chars} characters.")

        # 2. 构建词表 (如果是训练集)
        if vocab is not None:
            print("Using provided vocabulary.")
            self.token2idx = vocab
        elif is_train:
            print("Building vocabulary from scratch...")
            # 统计词频，构建词表
            vocab_counter = Counter(self.data_chars)
            vocab_list = sorted(vocab_counter, key=vocab_counter.get, reverse=True)
            self.token2idx = {char: idx+1 for idx, char in enumerate(vocab_list)}
            self.token2idx['<unk>'] = 0
        else:
            raise ValueError("Validation set must use training vocab (vocab cannot be None)!")
        
        self.idx2token = {idx: char for char, idx in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
        
        # 3. 将文本转换为整数索引
        self.data_ids = [self.token2idx.get(char, self.token2idx['<unk>']) for char in self.data_chars]
        self.data_ids = torch.tensor(self.data_ids, dtype=torch.long)

    def __len__(self):
        # 数据量 = 总长度 - 序列长度
        return len(self.data_ids) - self.seq_len

    def __getitem__(self, idx):
        # 输入: text[i : i+seq_len]
        # 目标: text[i+1 : i+seq_len+1]
        src = self.data_ids[idx : idx + self.seq_len]
        trg = self.data_ids[idx + 1 : idx + self.seq_len + 1]
        return src, trg

def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"💾 Vocabulary saved to {path}")

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"📖 Vocabulary loaded from {path}")
    return vocab

def get_dataloaders(config):
    vocab = None
    if os.path.exists(VOCAB_FILE):
        print(f"Found saved vocabulary at {VOCAB_FILE}")
        vocab = load_vocab(VOCAB_FILE)

    print("Processing Training Data...")
    train_ds = RealTextDataset(TRAIN_FILE, vocab=vocab, seq_len=config['seq_len'], is_train=True)
    
    if vocab is None:
        save_vocab(train_ds.token2idx, VOCAB_FILE)

    # 更新 config 中的 vocab_size，因为是根据数据动态生成的
    config['vocab_size'] = train_ds.vocab_size
    print(f"Vocab Size: {config['vocab_size']}")
    
    print("Processing Validation Data...")
    # 注意：验证集传入 train_ds.token2idx
    val_ds = RealTextDataset(VALID_FILE, vocab=train_ds.token2idx, seq_len=config['seq_len'], is_train=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def save_checkpoint(model, vocab, config, metrics, filename):
    """
    保存模型权重、词表配置和训练指标
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,          # 保存词表(char->idx映射)
        'config': config,        # 保存超参数
        'metrics': metrics       # 保存训练曲线数据
    }
    path = os.path.join(CHECKPOINT_PATH, filename)
    torch.save(checkpoint, path)
    print(f"✅ 模型与词表已保存至: {path}")

def load_checkpoint(filename, model_class, device='cpu'):
    """
    加载模型和词表(由于有 wandb，其实也不太需要了)
    """
    path = os.path.join(CHECKPOINT_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型文件: {path}")
        
    print(f"🔄 正在加载模型: {path} ...")
    checkpoint = torch.load(path, map_location=device)
    
    vocab = checkpoint['vocab']
    config = checkpoint['config']
    
    # 根据配置重新初始化模型结构
    if 'RNN' in filename:
        model = model_class(len(vocab), config['embed_dim'], config['hidden_dim'])
    elif 'FNN' in filename:
        model = model_class(len(vocab), config['embed_dim'], config['hidden_dim'], config['seq_len'])
    elif 'Transformer' in filename:
        model = model_class(
            len(vocab),
            config['embed_dim'], 
            config['hidden_dim'], 
            config['num_heads'],
            config['layers'],
            config['dropout']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, vocab, checkpoint['metrics']

if __name__ == "__main__":
    split_dataset(SOURCE_FILE, TRAIN_FILE, VALID_FILE)