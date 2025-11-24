import torch
from model import *
from config import *
from utils import *
from path import *


def generate_text(model, start_text, vocab, idx2token, config, max_new_tokens=50, temperature=1.0, top_k=10):
    """
    自回归文本生成 (Autoregressive Generation)
    兼容 Transformer, RNN 和 FNN
    """
    model.eval()
    device = config['device']
    
    # 1. 预处理输入
    input_ids = [vocab.get(c, vocab['<unk>']) for c in start_text]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device) # [1, len]
    
    generated_text = start_text
    
    print(f"📖 生成中 [{type(model).__name__}]: {start_text}", end="", flush=True)
    
    # 获取 Padding 的索引 (通常 <unk> 是 0)
    pad_idx = vocab.get('<unk>', 0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # A. 截断逻辑 (Transformer/RNN/FNN 都需要处理过长序列)
            if input_tensor.size(1) > config['seq_len']:
                cond = input_tensor[:, -config['seq_len']:]
            else:
                cond = input_tensor
                
            # B. 填充逻辑 (专为 FNN 设计)
            # FNN 强制要求输入长度等于 seq_len，否则 fc层 维度对不上
            if type(model).__name__ == 'FNN_LM' and cond.size(1) < config['seq_len']:
                pad_len = config['seq_len'] - cond.size(1)
                pad_tensor = torch.full((1, pad_len), pad_idx, dtype=torch.long, device=device)
                cond = torch.cat((pad_tensor, cond), dim=1) # [pad, context]

            # C. 前向传播
            logits = model(cond)
            
            # 取最后一个时间步
            logits = logits[:, -1, :] / temperature
            
            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 计算概率并采样
            probs = F.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            input_tensor = torch.cat((input_tensor, next_token_idx), dim=1)
            
            # 解码
            char = idx2token.get(next_token_idx.item(), '<unk>')
            generated_text += char
            print(char, end="", flush=True)
            
    print("\n")
    return generated_text

def test_model(model_name, start_text="今天天气"):
    print(f"\n>>> Testing {model_name}...")
    
    # 1. 准备配置
    if model_name not in MODEL_ARCH_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    # 加载词表
    vocab = load_vocab(VOCAB_FILE)
    idx2token = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    # 合并配置
    config = merge_config(TRAIN_CONFIG, MODEL_ARCH_CONFIGS[model_name])
    config['vocab_size'] = vocab_size
    
    # 2. 初始化模型结构
    device = config['device']
    if model_name == 'FNN':
        model = FNN_LM(vocab_size, config['embed_dim'], config['hidden_dim'], config['seq_len'])
    elif model_name == 'RNN':
        model = RNN_LM(vocab_size, config['embed_dim'], config['hidden_dim'])
    elif model_name == 'Transformer':
        model = Transformer_LM(vocab_size, config['embed_dim'], config['hidden_dim'], 
                               config['num_heads'], config['layers'], config['dropout'])
    
    # 3. 加载权重
    # 注意文件名要和你 train_and_eval.py 里保存的一致，通常是 "FNN.pth" 等
    ckpt_path = os.path.join(CHECKPOINT_PATH, f"{model_name}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 处理 DataParallel 保存时带有的 'module.' 前缀
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 4. 生成文本
    generate_text(model, start_text, vocab, idx2token, config, max_new_tokens=20)

if __name__ == "__main__":
    test_model("FNN", "也就是")
    test_model("RNN", "也就是")
    test_model("Transformer", "也就是")