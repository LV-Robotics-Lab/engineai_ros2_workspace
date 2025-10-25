#!/usr/bin/env python3
"""
合并多个contact CSV文件的脚本
"""

import os
import sys
import pandas as pd
import glob
from datetime import datetime
import time

def merge_contact_csv_files(log_dir, output_file=None):
    """
    合并指定目录下的所有contact_data_*.csv文件
    
    Args:
        log_dir: 包含CSV文件的目录
        output_file: 输出文件名，如果为None则自动生成
    """
    print(f"正在合并目录: {log_dir}")
    
    # 查找所有contact_data_*.csv文件
    pattern = os.path.join(log_dir, "contact_data_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"在目录 {log_dir} 中没有找到contact_data_*.csv文件")
        return None
    
    # 按文件名排序
    csv_files.sort()
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # 读取并合并所有CSV文件
    all_dataframes = []
    total_rows = 0
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"开始处理 {len(csv_files)} 个CSV文件")
    print(f"{'='*60}")
    
    for i, csv_file in enumerate(csv_files):
        file_start_time = time.time()
        print(f"\n📁 处理文件 {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        try:
            # 显示进度条
            progress = (i / len(csv_files)) * 100
            bar_length = 30
            filled_length = int(bar_length * i // len(csv_files))
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"  进度: [{bar}] {progress:.1f}% ({i+1}/{len(csv_files)})")
            
            # 读取CSV文件
            print(f"  📖 正在读取文件...")
            df = pd.read_csv(csv_file)
            read_time = time.time() - file_start_time
            print(f"  ✅ 读取完成: {len(df)} 行数据 (耗时: {read_time:.2f}秒)")
            
            # 添加源文件信息 - 使用文件名（不含扩展名）作为标识
            source_name = os.path.splitext(os.path.basename(csv_file))[0]  # 去掉.csv扩展名
            df['source_file'] = source_name
            
            all_dataframes.append(df)
            total_rows += len(df)
            
            # 显示累计统计
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / (i + 1)
            remaining_files = len(csv_files) - (i + 1)
            estimated_remaining_time = remaining_files * avg_time_per_file
            
            print(f"  📊 累计: {total_rows} 行数据")
            print(f"  ⏱️  预计剩余时间: {estimated_remaining_time:.1f}秒")
            
        except Exception as e:
            print(f"  ❌ 错误: 无法读取文件 {csv_file}: {e}")
            continue
    
    if not all_dataframes:
        print("没有成功读取任何CSV文件")
        return None
    
    # 合并所有数据框
    print(f"\n{'='*60}")
    print(f"🔄 正在合并 {len(all_dataframes)} 个数据框...")
    print(f"{'='*60}")
    
    merge_start_time = time.time()
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merge_time = time.time() - merge_start_time
    
    print(f"✅ 合并完成! (耗时: {merge_time:.2f}秒)")
    print(f"📊 总行数: {len(merged_df)}")
    print(f"📊 总列数: {len(merged_df.columns)}")
    
    total_processing_time = time.time() - start_time
    print(f"⏱️  总处理时间: {total_processing_time:.2f}秒")
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取原文件夹名称
        folder_name = os.path.basename(os.path.abspath(log_dir))
        output_file = os.path.join(log_dir, f"merged_contact_data_{folder_name}_{timestamp}.csv")
    elif not os.path.isabs(output_file):
        # 如果输出文件不是绝对路径，则保存到log_dir目录中
        output_file = os.path.join(log_dir, output_file)
    
    # 保存合并后的文件
    print(f"\n{'='*60}")
    print(f"💾 正在保存到: {output_file}")
    print(f"{'='*60}")
    
    save_start_time = time.time()
    
    # 分块保存以显示进度
    total_rows = len(merged_df)
    chunk_size = max(1000, total_rows // 20)  # 至少分20个块，每块至少1000行
    
    print(f"📊 总行数: {total_rows:,}")
    print(f"📦 分块大小: {chunk_size:,} 行/块")
    print(f"🔄 预计分块数: {(total_rows + chunk_size - 1) // chunk_size}")
    
    # 创建进度显示
    def save_with_progress():
        import io
        output_buffer = io.StringIO()
        
        # 写入CSV头部
        merged_df.head(0).to_csv(output_buffer, index=False)
        header_size = len(output_buffer.getvalue())
        
        # 分块写入数据
        chunks_written = 0
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = merged_df.iloc[start_idx:end_idx]
            
            # 写入数据块
            chunk_buffer = io.StringIO()
            chunk_df.to_csv(chunk_buffer, index=False, header=False)
            chunk_data = chunk_buffer.getvalue()
            
            # 显示进度
            chunks_written += 1
            progress = (chunks_written / total_chunks) * 100
            bar_length = 30
            filled_length = int(bar_length * chunks_written // total_chunks)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            elapsed = time.time() - save_start_time
            if chunks_written > 1:
                avg_time_per_chunk = elapsed / chunks_written
                remaining_chunks = total_chunks - chunks_written
                estimated_remaining = remaining_chunks * avg_time_per_chunk
                print(f"\r  💾 保存进度: [{bar}] {progress:.1f}% ({chunks_written}/{total_chunks}) "
                      f"⏱️ 已用: {elapsed:.1f}s 预计剩余: {estimated_remaining:.1f}s", end='', flush=True)
            else:
                print(f"\r  💾 保存进度: [{bar}] {progress:.1f}% ({chunks_written}/{total_chunks}) "
                      f"⏱️ 已用: {elapsed:.1f}s", end='', flush=True)
            
            # 将数据写入文件
            with open(output_file, 'a' if chunks_written > 1 else 'w') as f:
                if chunks_written == 1:
                    # 第一次写入，包含头部
                    f.write(merged_df.head(0).to_csv(index=False))
                f.write(chunk_data)
        
        print()  # 换行
    
    # 执行分块保存
    save_with_progress()
    save_time = time.time() - save_start_time
    
    file_size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"✅ 保存完成! (耗时: {save_time:.2f}秒)")
    print(f"📁 输出文件: {output_file}")
    print(f"📏 文件大小: {file_size_mb:.1f} MB")
    print(f"⚡ 写入速度: {file_size_mb/save_time:.1f} MB/s")
    
    # 显示数据统计
    print(f"\n{'='*60}")
    print(f"📊 合并后的数据统计")
    print(f"{'='*60}")
    print(f"📈 总行数: {len(merged_df)}")
    print(f"⏰ 时间范围: {merged_df['timestamp'].min():.3f} - {merged_df['timestamp'].max():.3f} 秒")
    print(f"💪 力范围: {merged_df['force_magnitude'].min():.3f} - {merged_df['force_magnitude'].max():.3f} N")
    print(f"📊 平均力: {merged_df['force_magnitude'].mean():.3f} N")
    print(f"📊 中位数力: {merged_df['force_magnitude'].median():.3f} N")
    
    # 按源文件统计
    print(f"\n{'='*60}")
    print(f"📁 按源文件统计")
    print(f"{'='*60}")
    source_stats = merged_df.groupby('source_file').size().sort_values(ascending=False)
    for source, count in source_stats.items():
        percentage = (count / len(merged_df)) * 100
        print(f"  📄 {source}: {count} 行 ({percentage:.1f}%)")
    
    # 显示源文件列的信息
    print(f"\n{'='*60}")
    print(f"ℹ️  源文件列信息")
    print(f"{'='*60}")
    print(f"✅ 源文件列已添加到数据中，列名: 'source_file'")
    print(f"📊 包含 {len(source_stats)} 个不同的源文件")
    print(f"🔍 源文件列示例值: {list(source_stats.index[:3])}")  # 显示前3个源文件名
    
    return output_file

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 merge_contact_data.py <log_directory> [output_file]")
        print("示例: python3 merge_contact_data.py logs/forward-200.0N-20251005_162754")
        print("示例: python3 merge_contact_data.py logs/forward-200.0N-20251005_162754 merged_contact.csv")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(log_dir):
        print(f"错误: 目录 {log_dir} 不存在")
        sys.exit(1)
    
    # 合并CSV文件
    merged_file = merge_contact_csv_files(log_dir, output_file)
    
    if merged_file:
        print(f"\n✅ 合并成功! 输出文件: {merged_file}")
        print(f"\n现在可以使用以下命令来可视化合并后的数据:")
        print(f"python3 scripts/mujoco_xml_contact_display.py {merged_file} src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1500 true 1,2")
    else:
        print("❌ 合并失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
