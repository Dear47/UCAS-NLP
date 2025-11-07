import re
import math
import collections
import matplotlib.pyplot as plt
from pathlib import Path
from pylab import mpl
import glob
import os

# 设置中文显示字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams['axes.unicode_minus'] = False

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def entropy(freq_dict):
    """计算香农熵，输入为频率字典（值为概率）"""
    H = 0.0
    for p in freq_dict.values():
        if p > 0:
            H -= p * math.log2(p)
    return H

def get_top_items(freq_counter, top_n=50):
    """返回前 top_n 项的 (rank, freq) 列表"""
    sorted_items = freq_counter.most_common(top_n)
    ranks = list(range(1, len(sorted_items) + 1))
    freqs = [freq for _, freq in sorted_items]
    return ranks, freqs

def get_file_size_mb(filepath):
    """获取文件大小（MB）"""
    return os.path.getsize(filepath) / (1024 * 1024)

def analyse_corpus_en(filepath:str, ifplot:bool=True)->tuple[float, float]:
    """
    分析指定的.txt文件, 并选择是否绘图。

    :param filepath: (str) 指定.txt文件的绝对路径
    :param ifplot: (bool) 选择是否绘图
    """
    print(f"处理英文语料{filepath}...")

    en_text = read_file(filepath).lower()
    file_size = get_file_size_mb(filepath)

    # 提取字母（a-z）
    en_letters = re.findall(r'[a-z]', en_text)
    total_letters = len(en_letters)
    letter_counter = collections.Counter(en_letters)
    letter_probs = {char: count / total_letters for char, count in letter_counter.items()}

    # 提取单词（仅字母组成的词）
    en_words = re.findall(r'\b[a-z]+\b', en_text)
    total_words = len(en_words)
    word_counter = collections.Counter(en_words)
    word_probs = {word: count / total_words for word, count in word_counter.items()}

    # 计算熵
    letter_entropy = entropy(letter_probs)
    word_entropy = entropy(word_probs)

    print(f"英文字符熵: {letter_entropy:.4f} bits")
    print(f"英文单词熵: {word_entropy:.4f} bits")

    if ifplot:
        plt.figure(figsize=(15, 10))

        # 英文字符 Zipf
        plt.subplot(2, 2, 1)
        ranks, freqs = get_top_items(letter_counter, top_n=26)
        plt.loglog(ranks, freqs, 'o-', label='English Letters')
        plt.title('Zipf Law: English Letters')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.grid(True, which="both", ls="--")

        # 英文单词 Zipf(取前100)
        plt.subplot(2, 2, 2)
        ranks, freqs = get_top_items(word_counter, top_n=100)
        plt.loglog(ranks, freqs, 'o-', markersize=2, label='English Words')
        plt.title('Zipf Law: English Words (Top 100)')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.grid(True, which="both", ls="--")

        # 概率分布直方图（英文字符）
        plt.subplot(2, 2, 3)
        letters_sorted = sorted(letter_probs.items(), key=lambda x: x[1], reverse=True)
        chars, probs = zip(*letters_sorted)
        plt.bar(range(len(chars)), probs)
        plt.title('English Letter Probability Distribution')
        plt.xlabel('Letter')
        plt.ylabel('Probability')
        plt.xticks(range(len(chars)), chars)

        # 空白或补充信息
        plt.subplot(2, 2, 4)
        plt.axis('off')
        info = (
            f"Entropy Summary:\n"
            f"English Letters: {letter_entropy:.4f} bits\n"
            f"English Words: {word_entropy:.4f} bits\n"
            f"Total English letters: {total_letters}\n"
            f"Total English words: {total_words}\n"
        )
        plt.text(0.1, 0.5, info, fontsize=12, verticalalignment='center', family='monospace')

        plt.savefig(f"zipf_law_english_{file_size:.1f}.png",dpi=300)
        plt.close()

    print("分析完成!")

    return letter_entropy, word_entropy

def analyse_corpus_ch(filepath:str, ifplot:bool=True)->float:
    """
    分析指定的.txt文件, 并选择是否绘图。
    
    :param filepath: (str) 指定.txt文件的绝对路径
    :param ifplot: (bool) 选择是否绘图
    """
    print(f"处理中文语料{filepath}...")

    ch_text = read_file(filepath)
    file_size = get_file_size_mb(filepath)

    # 提取汉字（Unicode 范围：\u4e00-\u9fff）
    ch_chars = re.findall(r'[\u4e00-\u9fff]', ch_text)
    total_chars = len(ch_chars)
    char_counter = collections.Counter(ch_chars)
    char_probs = {char: count / total_chars for char, count in char_counter.items()}

    # 计算熵
    char_entropy = entropy(char_probs)
    print(f"中文字符熵: {char_entropy:.4f} bits")

    if ifplot:

        plt.figure(figsize=(30, 15))
        # 中文字符 Zipf(取前100)
        plt.subplot(3, 1, 1)
        ranks, freqs = get_top_items(char_counter, top_n=100)
        plt.loglog(ranks, freqs, 'o-', markersize=2, color='red', label='Chinese Characters')
        plt.title('Zipf Law: Chinese Characters (Top 100)')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.grid(True, which="both", ls="--")

        # 概率分布（中文前50字）
        plt.subplot(3, 1, 2)
        top_ch = char_counter.most_common(50)
        chars_ch, counts_ch = zip(*top_ch)
        probs_ch = [c / total_chars for c in counts_ch]
        plt.bar(range(len(chars_ch)), probs_ch, color='orange')
        plt.title('Top 50 Chinese Characters Probability')
        plt.xlabel('Character')
        plt.ylabel('Probability')
        plt.xticks(range(len(chars_ch)), chars_ch, rotation=90, fontsize=8)

        # 空白或补充信息
        plt.subplot(3, 1, 3)
        plt.axis('off')
        info = (
            f"Entropy Summary:\n"
            f"Chinese Chars: {char_entropy:.4f} bits\n"
            f"Total Chinese chars: {total_chars}"
        )
        plt.text(0.1, 0.5, info, fontsize=24, verticalalignment='center', family='monospace')
        plt.savefig(f"zipf_law_chinese_{file_size:.1f}.png",dpi=300)
        plt.close()

    print("分析完成!")
    
    return char_entropy

def plot_entropy_vs_corpus_size(ch_sizes, ch_entropies, en_letter_sizes, en_letter_entropies, en_word_sizes, en_word_entropies):
    """
    绘制熵值随语料规模变化的曲线
    """
    
    # 中文熵值图
    plt.figure(figsize=(15, 5))
    plt.plot(ch_sizes, ch_entropies, 'ro-', linewidth=2, markersize=6, label='实际熵值')
    # 添加标准熵值参考线（中文标准熵值约为9-10 bits）
    std_ch_entropy = 9.5  # 中文字符的标准熵值参考
    plt.axhline(y=std_ch_entropy, color='red', linestyle='--', alpha=0.7, label=f'标准熵值 ({std_ch_entropy} bits)')
    plt.xlabel('语料规模 (MB)')
    plt.ylabel('熵值 (bits)')
    plt.title('中文字符熵值 vs 语料规模')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("chinese_entropy_for_corpus_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 英文字母熵值图
    plt.figure(figsize=(15, 5))
    plt.plot(en_letter_sizes, en_letter_entropies, 'bo-', linewidth=2, markersize=6, label='实际熵值')
    # 添加标准熵值参考线（英文字母标准熵值约为4.0-4.2 bits）
    std_en_letter_entropy = 4.1
    plt.axhline(y=std_en_letter_entropy, color='blue', linestyle='--', alpha=0.7, label=f'标准熵值 ({std_en_letter_entropy} bits)')
    plt.xlabel('语料规模 (MB)')
    plt.ylabel('熵值 (bits)')
    plt.title('英文字母熵值 vs 语料规模')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("letter_entropy_for_corpus_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 英文单词熵值图
    plt.figure(figsize=(15, 5))
    plt.plot(en_word_sizes, en_word_entropies, 'go-', linewidth=2, markersize=6, label='实际熵值')
    # 添加标准熵值参考线（英文单词标准熵值约为10-12 bits）
    std_en_word_entropy = 11.0
    plt.axhline(y=std_en_word_entropy, color='green', linestyle='--', alpha=0.7, label=f'标准熵值 ({std_en_word_entropy} bits)')
    plt.xlabel('语料规模 (MB)')
    plt.ylabel('熵值 (bits)')
    plt.title('英文单词熵值 vs 语料规模')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("word_entropy_for_corpus_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    char_entropies = []
    letter_entropies = []
    word_entropies = []
    ch_sizes = []
    en_letter_sizes = []
    en_word_sizes = []
    
    base_dir = Path(__file__).parent.parent

    # 分析中文语料
    corpus_ch_dir = base_dir / "merged_corpus_ch"
    if not os.path.isdir(corpus_ch_dir):
        print(f"[!] 错误: 目录不存在: {corpus_ch_dir}")
    ch_txt_files = glob.glob(os.path.join(corpus_ch_dir, "corpus_ch_*.txt"))
    if not ch_txt_files:
        print(f"[!] 警告: 在 {corpus_ch_dir} 中未找到中文语料文件。")
    else:
        print(f"[*] 找到 {len(ch_txt_files)} 个中文语料文件。")
        # 按文件大小排序，从小到大
        ch_txt_files.sort(key=lambda x: get_file_size_mb(x))
        
        for filepath in ch_txt_files:
            file_size = get_file_size_mb(filepath)
            char_entropy = analyse_corpus_ch(filepath=filepath, ifplot=True)
            char_entropies.append(char_entropy)
            ch_sizes.append(file_size)
            print(f"中文文件: {os.path.basename(filepath)}, 大小: {file_size:.2f} MB, 熵值: {char_entropy:.4f} bits")

    # 分析英文语料
    corpus_en_dir = base_dir / "merged_corpus_en"
    if not os.path.isdir(corpus_en_dir):
        print(f"[!] 错误: 目录不存在: {corpus_en_dir}")
    en_txt_files = glob.glob(os.path.join(corpus_en_dir, "corpus_en_*.txt"))
    if not en_txt_files:
        print(f"[!] 警告: 在 {corpus_en_dir} 中未找到英文语料文件。")
    else:
        print(f"[*] 找到 {len(en_txt_files)} 个英文语料文件。")
        # 按文件大小排序，从小到大
        en_txt_files.sort(key=lambda x: get_file_size_mb(x))
        
        for filepath in en_txt_files:
            file_size = get_file_size_mb(filepath)
            letter_entropy, word_entropy = analyse_corpus_en(filepath=filepath, ifplot=True)
            letter_entropies.append(letter_entropy)
            word_entropies.append(word_entropy)
            en_letter_sizes.append(file_size)
            en_word_sizes.append(file_size)
            print(f"英文文件: {os.path.basename(filepath)}, 大小: {file_size:.2f} MB, 字母熵: {letter_entropy:.4f} bits, 单词熵: {word_entropy:.4f} bits")

    # 绘制熵值随语料规模变化的曲线
    if ch_sizes and en_letter_sizes and en_word_sizes:
        plot_entropy_vs_corpus_size(ch_sizes, char_entropies, en_letter_sizes, letter_entropies, en_word_sizes, word_entropies)
        
        # 输出统计信息
        print("\n" + "="*50)
        print("熵值统计分析结果:")
        print("="*50)
        print(f"中文语料:")
        print(f"  样本数量: {len(ch_sizes)}")
        print(f"  熵值范围: {min(char_entropies):.4f} - {max(char_entropies):.4f} bits")
        print(f"  平均熵值: {sum(char_entropies)/len(char_entropies):.4f} bits")
        print(f"英文语料:")
        print(f"  字母熵值范围: {min(letter_entropies):.4f} - {max(letter_entropies):.4f} bits")
        print(f"  单词熵值范围: {min(word_entropies):.4f} - {max(word_entropies):.4f} bits")
        print(f"  平均字母熵值: {sum(letter_entropies)/len(letter_entropies):.4f} bits")
        print(f"  平均单词熵值: {sum(word_entropies)/len(word_entropies):.4f} bits")
    else:
        print("[!] 警告: 没有足够的数据进行熵值变化分析")

    

    