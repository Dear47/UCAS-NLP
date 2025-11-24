import torch
import torch.nn as nn
import torch.optim as optim

import os
import math
import time
import wandb
from tqdm import tqdm

from utils import *
from model import *
from config import *


WANDB_PROJECT = "NLP-Assignment2-LM"
WANDB_MODE = "offline"

def calculate_perplexity(loss):
    """
    计算困惑度，数学上可以证明困惑度就是交叉熵的指数形式：ppl=exp(CE)
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def run_experiment(model, train_loader, val_loader, config, model_name="Model", vocab=None):
    wandb.init(
        project=WANDB_PROJECT, 
        name=f"{model_name}",
        config=config,
        mode=WANDB_MODE,
        reinit=True
    )

    device = torch.device(config['device'])

    # 🚀 显式检查：打印当前模型到底在什么设备上
    if device.type == 'cuda':
        print(f"✅ [Check] GPU Available: {torch.cuda.get_device_name(0)}")
        # 🚀 启用 CuDNN Benchmark (针对输入尺寸固定的网络加速)
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️ [Check] Using CPU! Training will be slow.")

    model = model.to(config['device'])

    # 🚀 [加速 1] PyTorch 2.0 编译
    # ⚠️ 与下面的 DataParallel 不兼容，会导致 AttributeError，只能二选一
    # if hasattr(torch, 'compile'):
    #     print("Compiling model with torch.compile...")
    #     model = torch.compile(model)

    # 🚀 [加速 2] DataParallel
    use_multi_gpu = False
    if torch.cuda.device_count() > 1:
        print(f"\n🚀 [System] Detect {torch.cuda.device_count()} GPU, use DataParallel!")
        model = nn.DataParallel(model)
        use_multi_gpu = True
    else:
        print(f"\n💻 [System] Use single GPU/CPU: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], betas=config['betas'])

    # 🚀 学习率调度器
    # 自动进行 Warmup (热身)，防止训练初期梯度爆炸
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'], 
        steps_per_epoch=len(train_loader), 
        epochs=config['epochs']
    )

    # 🚀 [加速 3] 混合精度训练 (AMP)
    scaler = torch.amp.GradScaler()
    print(f"\n>>> Training {model_name} (Batch Size: {config['batch_size']})...")

    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    best_ppl = float('inf')

    # 确定 autocast 的设备类型
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    # 确定混合精度类型：CUDA用float16，CPU通常用bfloat16
    amp_dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16

    for epoch in range(config['epochs']):
        start_time = time.time()
        model.train()
        total_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", unit="batch")
        for step, (x, y) in enumerate(train_pbar):
            x, y = x.to(config['device'], non_blocking=True), y.to(config['device'], non_blocking=True)  # non_blocking加速传输
            optimizer.zero_grad()
            
            # 🚀 混合精度上下文
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                output = model(x)
                loss = criterion(output.view(-1, config['vocab_size']), y.view(-1))
            
            # 🚀 Scaler 反向传播
            scaler.scale(loss).backward()

            # 🚀 [关键] 梯度裁剪 (Gradient Clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_loss = loss.item()
            total_loss += current_loss
            lr_current = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({'loss': f"{current_loss:.4f}", 'lr': f"{lr_current:.6f}"})

            # 📝 WandB Log (Step Level)
            wandb.log({
                "train_loss": current_loss, 
                "learning_rate": lr_current,
                "epoch": epoch
            })

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Valid]", unit="batch")
        with torch.no_grad():
            for x, y in val_pbar:
                x, y = x.to(config['device']), y.to(config['device'])
                output = model(x)
                loss = criterion(output.view(-1, config['vocab_size']), y.view(-1))
                val_loss += loss.item()
                val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        ppl = calculate_perplexity(avg_val_loss)

        epoch_time = time.time() - start_time
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_ppl'].append(ppl)

        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | Val PPL: {ppl:.2f}")

        # 📝 WandB Log (Epoch Level)
        wandb.log({
            "val_loss": avg_val_loss,
            "val_ppl": ppl
        })

        if ppl < best_ppl:
            best_ppl = ppl

    # 保存 (处理 DataParallel 的 module 前缀)
    if use_multi_gpu:
        model_to_save = model.module
    else:
        model_to_save = model

    save_filename = f"{model_name.replace(' ', '_')}.pth"
    save_checkpoint(model_to_save, vocab, config, history, save_filename)

    wandb.finish()

    return ppl, history

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"

    base_config = TRAIN_CONFIG.copy()
    # # 1. 加载数据
    train_loader, val_loader = get_dataloaders(base_config)

    # # 2. 加载词表
    vocab = train_loader.dataset.token2idx

    all_results = {}

    # 3. 训练
    # 3.1 FNN
    fnn_config = merge_config(base_config, MODEL_ARCH_CONFIGS['FNN'])
    fnn_model = FNN_LM(
        fnn_config['vocab_size'], 
        fnn_config['embed_dim'], 
        fnn_config['hidden_dim'], 
        fnn_config['seq_len']
    )
    fnn_ppl, fnn_hist = run_experiment(fnn_model, train_loader, val_loader, fnn_config, "FNN", vocab)
    all_results['FNN'] = fnn_hist

    # 3.2 RNN (LSTM)
    rnn_config = merge_config(base_config, MODEL_ARCH_CONFIGS['RNN'])
    rnn_model = RNN_LM(
        rnn_config['vocab_size'], 
        rnn_config['embed_dim'], 
        rnn_config['hidden_dim']
    )
    rnn_ppl, rnn_hist = run_experiment(rnn_model, train_loader, val_loader, rnn_config, "RNN", vocab)
    all_results['RNN'] = rnn_hist

    # 3.3 Self-Attention (Transformer)
    att_config = merge_config(base_config, MODEL_ARCH_CONFIGS['Transformer'])
    att_model = Transformer_LM(
        att_config['vocab_size'], 
        att_config['embed_dim'], 
        att_config['hidden_dim'], 
        att_config['num_heads'], 
        att_config['layers'], 
        att_config['dropout']
    )
    att_ppl, att_hist = run_experiment(att_model, train_loader, val_loader, att_config, "Transformer", vocab)
    all_results['Transformer'] = att_hist

    # 4. 展示最终结果
    print("\n" + "="*30)
    print("Final Results (Lower PPL is better):")
    print(f"FNN PPL:         {fnn_ppl:.2f}")
    print(f"RNN PPL:         {rnn_ppl:.2f}")
    print(f"Transformer PPL: {att_ppl:.2f}")
    print("="*30)