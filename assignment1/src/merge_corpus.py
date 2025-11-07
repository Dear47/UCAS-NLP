import os
import glob
import math
import pathlib

def merge_and_rename(input_dir, output_dir, output_prefix, target_size_mb=None):
    """
    合并指定目录下的 .txt 文件，并按大小重命名。

    :param input_dir: (str) 包含源 txt 文件的目录 (例如 "corpus_chinese")
    :param output_dir: (str) 合并后文件的存放目录
    :param output_prefix: (str) 合并后文件的前缀 (例如 "corpus_ch_")
    :param target_size_mb: (float, optional) 期望的合并大小 (MB)。
                           如果设置此参数, 函数将按顺序合并文件, 
                           直到合并后的文件大小 >= target_size_mb。
                           如果为 None (默认), 则合并所有文件。
    """
    
    print(f"\n[*] 正在处理目录: {input_dir}")
    
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"[!] 错误: 目录不存在: {input_dir}")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] 输出目录: {output_dir}")

    # 查找所有 .txt 文件
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print(f"[!] 警告: 在 {input_dir} 中未找到 .txt 文件。")
        return

    print(f"[*] 找到 {len(txt_files)} 个 .txt 文件。")

    target_size_bytes = None
    if target_size_mb is not None:
        target_size_bytes = target_size_mb * 1024 * 1024
        print(f"[*] 目标大小: {target_size_mb} MB (约 {target_size_bytes} 字节)")
    else:
        print("[*] 目标大小: 无限制 (将合并所有文件)")

    # 定义临时和最终输出文件的完整路径
    temp_output_filename = os.path.join(output_dir, f"{output_prefix}temp_merge.txt")

    files_merged_count = 0
    
    try:
        # 合并文件
        with open(temp_output_filename, 'w', encoding='utf-8') as outfile:
            for filepath in txt_files:
                
                # 检查 *当前* 输出文件大小
                current_size_bytes = outfile.tell()
                
                # 如果设置了目标，并且已至少合并了一个文件，并且当前大小已达标
                if target_size_bytes is not None and files_merged_count > 0 and current_size_bytes >= target_size_bytes:
                    print(f"[*] 已达到目标大小 (当前 {current_size_bytes} 字节)。停止合并。")
                    break

                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        # 读取内容并写入，在每个文件内容后加一个换行符分隔
                        content = infile.read()
                        outfile.write(content)
                        outfile.write("\n") # 用一个换行符分隔不同页面的语料
                        
                        files_merged_count += 1 # 成功合并一个文件
                        
                except Exception as e:
                    print(f"[!] 读取文件时出错 {filepath}: {e}")

        
        if files_merged_count == 0:
            print("[!] 警告: 未合并任何文件。")
            if os.path.exists(temp_output_filename):
                os.remove(temp_output_filename)
            return
            
        print(f"[*] 总共合并了 {files_merged_count} / {len(txt_files)} 个文件。")

        # 获取合并后文件的确切大小
        total_size_bytes = os.path.getsize(temp_output_filename)
        
        # 计算 MB, 1 MB = 1024 * 1024 bytes
        if total_size_bytes == 0:
            print("[!] 警告: 合并后的文件大小为 0 字节。")
            os.remove(temp_output_filename) # 删除空文件
            return

        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # 格式化大小
        size_str = f"{math.ceil(total_size_mb * 100) / 100}MB"
        
        # 生成最终文件名
        final_output_filename = os.path.join(output_dir, f"{output_prefix}{size_str}.txt")
        
        # 移除已存在的目标文件（如果有）
        if os.path.exists(final_output_filename):
            print(f"[!] 警告: 目标文件 {final_output_filename} 已存在，将被覆盖。")
            os.remove(final_output_filename)

        # 重命名临时文件
        os.rename(temp_output_filename, final_output_filename)
        
        print(f"[+] 合并完成!")
        print(f"    总大小: {total_size_mb:.6f} MB ({total_size_bytes} 字节)")
        print(f"    最终文件: {final_output_filename}")

    except Exception as e:
        print(f"[!] 合并过程中发生严重错误: {e}")
        # 如果出错，尝试删除临时文件
        if os.path.exists(temp_output_filename):
            os.remove(temp_output_filename)

if __name__ == "__main__":
    # 获取脚本所在的 (scripts/ 目录) 的 (父目录)
    base_dir = pathlib.Path(__file__).parent.parent

    print("--- 合并中文语料 ---")
    merge_and_rename(
        input_dir=base_dir / "raw_corpus_ch",
        output_dir=base_dir / "merged_corpus_ch",
        output_prefix="corpus_ch_", # 修改前缀
        target_size_mb=7 
    )
    
    print("\n--- 合并英文语料 ---")
    merge_and_rename(
        input_dir=base_dir / "raw_corpus_en",
        output_dir=base_dir / "merged_corpus_en",
        output_prefix="corpus_en_", # 修改前缀
        target_size_mb=14 # 传递 None 或不传此参数
    )