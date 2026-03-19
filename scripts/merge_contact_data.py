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
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from mujoco_data_io import load_contact_file, write_contact_data_bin

_DIRECTION_ORDER = [
    # longer first to avoid forward_left being parsed as forward
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
    "forward",
    "backward",
    "left",
    "right",
]


def extract_direction_from_text(text: str):
    if not text:
        return None
    t = str(text).lower()
    for d in _DIRECTION_ORDER:
        if d in t:
            return d
    return None


def extract_direction_from_path(path: str):
    """从路径各级目录名中提取方向（支持 8 方向）。"""
    try:
        parts = os.path.abspath(path).split(os.sep)
    except Exception:
        parts = [str(path)]
    for part in parts:
        d = extract_direction_from_text(part)
        if d:
            return d
    return None


def parse_fall_type_info(csv_file_path, log_dir):
    """
    从文件路径和目录名中解析摔倒类型、方向和csv时间编号
    
    Args:
        csv_file_path: CSV文件的完整路径
        log_dir: 日志目录路径
        
    Returns:
        格式化的字符串: "摔倒类型-方向-csv时间编号"，如果无法解析则返回None
    """
    try:
        # 获取文件名（不含扩展名）
        file_basename = os.path.basename(csv_file_path)
        file_name = os.path.splitext(file_basename)[0]  # 去掉.csv扩展名
        
        # 从文件名中提取时间戳（格式：YYYYMMDD_HHMMSS）
        # 对于CSV时间编号，通常取第一个时间戳（例如：contact_data_backward-00.0N-0.5s-20251021_213710_20251025_235705.csv）
        # 第一个时间戳是CSV时间编号，第二个可能是合并时间戳
        abs_csv_path = os.path.abspath(csv_file_path)
        abs_log_dir = os.path.abspath(log_dir)

        time_pattern = r'(\d{8}_\d{6})'
        time_matches = re.findall(time_pattern, file_name)
        csv_time = time_matches[0] if time_matches else None
        # contact_data.csv 无时间戳时，用父目录名（如 20260317_123521）
        if not csv_time:
            parent = os.path.basename(os.path.dirname(abs_csv_path))
            if re.match(r"^\d{8}_\d{6}$", parent):
                csv_time = parent
        if not csv_time:
            return None
        
        # 从文件路径中向上查找父目录，找到包含test_前缀的目录（如test_poweroff_100）
        # 这样可以正确处理子目录中的文件
        path_parts = abs_csv_path.split(os.sep)
        folder_name = None
        for i in range(len(path_parts) - 1, -1, -1):
            part = path_parts[i]
            if part.startswith('test_') or 'poweroff' in part.lower() or 'slip' in part.lower() or 'stumble' in part.lower() or 'push' in part.lower():
                folder_name = part
                break
        
        # 如果没找到，使用log_dir的basename作为备选
        if not folder_name:
            folder_name = os.path.basename(abs_log_dir)
        
        # 从目录名中提取摔倒类型
        # 例如: test_poweroff_100 -> poweroff
        # 例如: test_slip_100 -> slip
        # 例如: test_stumble_100 -> stumble
        fall_type = None
        if 'poweroff' in folder_name.lower():
            fall_type = 'poweroff'
        elif 'slip' in folder_name.lower():
            fall_type = 'slip'
        elif 'stumble' in folder_name.lower():
            fall_type = 'stumble'
        elif 'passive' in folder_name.lower() and 'push' in folder_name.lower():
            fall_type = 'passive_push'
        elif 'push' in folder_name.lower():
            fall_type = 'push'
        else:
            # 尝试从目录名中提取（去除test_和_100等后缀）
            fall_type_match = re.search(r'(?:test_)?([a-z]+)(?:_\d+)?', folder_name.lower())
            if fall_type_match:
                fall_type = fall_type_match.group(1)
        
        # 从路径/文件名中提取方向（支持 8 方向）
        direction = extract_direction_from_path(abs_csv_path)
        if not direction:
            direction = extract_direction_from_text(file_name)
        if not direction:
            direction_match = re.search(
                r"(forward_left|forward_right|backward_left|backward_right|backward|forward|left|right)",
                file_name.lower(),
            )
            direction = direction_match.group(1) if direction_match else "unknown"
        
        if fall_type and direction and csv_time:
            return f"{fall_type}-{direction}-{csv_time}"
        else:
            return None
            
    except Exception as e:
        print(f"  警告: 解析摔倒类型信息时出错: {e}")
        return None


def _glob_contact_logs(log_dir: str, *, recursive: bool) -> list:
    """
    仿真可能写出 contact_data_YYYYMMDD_HHMMSS.csv，也可能只有 contact_data.csv / .bin（无时间戳后缀）。
    默认合并时需两者都收进来。
    """
    patterns = (
        "contact_data_*.csv",
        "contact_data_*.bin",
        "contact_data.csv",
        "contact_data.bin",
    )
    found: list = []
    if recursive:
        for p in patterns:
            found.extend(glob.glob(os.path.join(log_dir, "**", p), recursive=True))
    else:
        for p in patterns:
            found.extend(glob.glob(os.path.join(log_dir, p)))
    # 去重、排序
    seen = set()
    out = []
    for f in found:
        a = os.path.abspath(f)
        if a not in seen:
            seen.add(a)
            out.append(a)
    return sorted(out)


def _filter_rows_min_force(df, min_force_n, basename_for_log=""):
    """
    丢弃「力」小于 min_force_n (N) 的行。
    优先用 force_normal（取绝对值），否则 force_magnitude。
    """
    if min_force_n is None or min_force_n <= 0:
        return df
    if "force_normal" in df.columns:
        col = "force_normal"
        series = pd.to_numeric(df[col], errors="coerce").abs()
    elif "force_magnitude" in df.columns:
        col = "force_magnitude"
        series = pd.to_numeric(df[col], errors="coerce").abs()
    else:
        print(f"  ⚠️  {basename_for_log}: 无 force_normal/force_magnitude，跳过最小力过滤")
        return df
    before = len(df)
    mask = series >= float(min_force_n)
    dropped = int((~mask).sum())
    out = df.loc[mask].copy()
    if dropped:
        print(
            f"  ✂️  最小力过滤 (≥{min_force_n:g} N，列 {col}): "
            f"删除 {dropped:,} 行，保留 {len(out):,} 行"
        )
    return out


def merge_contact_csv_files(
    log_dir,
    output_file=None,
    add_fall_type_column=False,
    file_pattern="contact_data_*.csv",
    write_bin=False,
    group_by_direction=False,
    fast_append=False,
    min_force_n=None,
):
    """
    合并指定目录下的所有CSV文件
    
    Args:
        log_dir: 包含CSV文件的目录
        output_file: 输出文件名，如果为None则自动生成
        add_fall_type_column: 是否添加"摔倒类型-方向-csv时间编号"列
        file_pattern: 要合并的文件模式，默认为 contact_data 日志（含无后缀的 contact_data.csv）
        min_force_n: 若设置，每个文件读入后丢弃力小于该值（N）的行；优先 |force_normal|
    """
    print(f"正在合并目录: {log_dir}")
    print(f"文件模式: {file_pattern}")
    
    if file_pattern == "contact_data_*.csv":
        csv_files = _glob_contact_logs(log_dir, recursive=False)
    else:
        pattern = os.path.join(log_dir, file_pattern)
        csv_files = glob.glob(pattern)
    is_recursive = False
    
    # 如果当前目录没有找到，递归查找子目录
    if not csv_files:
        if file_pattern == "contact_data_*.csv":
            csv_files = _glob_contact_logs(log_dir, recursive=True)
        else:
            pattern_recursive = os.path.join(log_dir, "**", file_pattern)
            csv_files = glob.glob(pattern_recursive, recursive=True)
        if csv_files:
            print(f"在当前目录未找到匹配的文件，递归查找子目录...")
            is_recursive = True
    
    if not csv_files:
        print(f"在目录 {log_dir} 及其子目录中没有找到匹配 {file_pattern} 的文件")
        return None
    
    # 如果是递归查找，且使用的是默认模式（contact_data_*.csv），支持按方向或按子目录分组
    # 如果使用自定义模式（如 merged_contact_data_*.csv），直接合并所有文件，不分组
    if is_recursive and file_pattern == "contact_data_*.csv":
        from collections import defaultdict
        abs_log_dir = os.path.abspath(log_dir)

        if group_by_direction:
            csv_by_group = defaultdict(list)  # direction -> files
            for csv_file in csv_files:
                d = extract_direction_from_path(csv_file) or "unknown"
                csv_by_group[d].append(os.path.abspath(csv_file))
            for k in csv_by_group:
                csv_by_group[k].sort()
            print(f"找到 {len(csv_files)} 个文件，按方向分为 {len(csv_by_group)} 组:")
            for k, files in sorted(csv_by_group.items()):
                print(f"  🧭 {k}: {len(files)} 个文件")

            output_files = []
            for direction, sub_csv_files in sorted(csv_by_group.items()):
                print(f"\n{'='*60}")
                print(f"🔄 处理方向: {direction}")
                print(f"{'='*60}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ext_out = ".bin" if write_bin else ".csv"
                if output_file is None:
                    sub_output_file = os.path.abspath(
                        os.path.join(log_dir, f"merged_contact_data_{direction}_{timestamp}{ext_out}")
                    )
                else:
                    base_name = os.path.splitext(os.path.basename(output_file))[0]
                    ext = ".bin" if write_bin else (os.path.splitext(output_file)[1] or ".csv")
                    sub_output_file = os.path.abspath(os.path.join(log_dir, f"{base_name}_{direction}{ext}"))

                merged_file = _merge_csv_files_from_list(
                    sub_csv_files,
                    log_dir,
                    sub_output_file,
                    add_fall_type_column,
                    write_bin=write_bin,
                    min_force_n=min_force_n,
                )
                if merged_file:
                    output_files.append(merged_file)

            if output_files:
                print(f"\n{'='*60}")
                print(f"✅ 按方向合并完成! 共生成 {len(output_files)} 个文件:")
                for f in output_files:
                    print(f"  📄 {f}")
                return output_files[0] if len(output_files) == 1 else output_files
            return None

        # 默认：按子目录分组（保持原行为）
        csv_by_dir = defaultdict(list)
        for csv_file in csv_files:
            abs_csv_file = os.path.abspath(csv_file)
            rel_path = os.path.relpath(abs_csv_file, abs_log_dir)
            sub_dir = os.path.dirname(rel_path) or "."
            csv_by_dir[sub_dir].append(abs_csv_file)

        for sub_dir in csv_by_dir:
            csv_by_dir[sub_dir].sort()

        print(f"找到 {len(csv_files)} 个CSV文件，分布在 {len(csv_by_dir)} 个子目录中:")
        for sub_dir, files in sorted(csv_by_dir.items()):
            print(f"  📁 {sub_dir}: {len(files)} 个文件")

        output_files = []
        for sub_dir, sub_csv_files in sorted(csv_by_dir.items()):
            print(f"\n{'='*60}")
            print(f"🔄 处理子目录: {sub_dir}")
            print(f"{'='*60}")

            if sub_dir == ".":
                sub_log_dir = log_dir
            else:
                sub_log_dir = os.path.join(log_dir, sub_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = os.path.basename(os.path.abspath(log_dir))
            sub_dir_name = os.path.basename(sub_dir) if sub_dir != "." else folder_name

            ext_out = ".bin" if write_bin else ".csv"
            if output_file is None:
                sub_output_file = os.path.abspath(
                    os.path.join(log_dir, f"merged_contact_data_{sub_dir_name}_{timestamp}{ext_out}")
                )
            else:
                base_name = os.path.splitext(os.path.basename(output_file))[0]
                ext = ".bin" if write_bin else (os.path.splitext(output_file)[1] or ".csv")
                sub_output_file = os.path.abspath(os.path.join(log_dir, f"{base_name}_{sub_dir_name}{ext}"))
            merged_file = _merge_csv_files_from_list(
                sub_csv_files,
                sub_log_dir,
                sub_output_file,
                add_fall_type_column,
                write_bin=write_bin,
                min_force_n=min_force_n,
            )
            if merged_file:
                output_files.append(merged_file)

        if output_files:
            print(f"\n{'='*60}")
            print(f"✅ 所有子目录合并完成! 共生成 {len(output_files)} 个文件:")
            for f in output_files:
                print(f"  📄 {f}")
            return output_files[0] if len(output_files) == 1 else output_files
        return None
    
    # 如果不是递归查找，按原来的方式处理（当前目录下的所有文件合并成一个）
    csv_files.sort()
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(file)}")

    if fast_append and not write_bin and file_pattern != "contact_data_*.csv":
        # fast append: 只做表头 + 按行追加（跳过 pandas concat 与统计），用于二次合并 merged_contact_data_*.csv
        abs_log_dir = os.path.abspath(log_dir)
        out_path = output_file
        if out_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = os.path.basename(os.path.abspath(log_dir))
            out_path = os.path.join(log_dir, f"merged_contact_data_{folder_name}_{timestamp}.csv")
        elif not os.path.isabs(str(out_path)):
            abs_out = os.path.abspath(str(out_path))
            if not abs_out.startswith(abs_log_dir):
                out_path = os.path.join(log_dir, str(out_path))
            else:
                out_path = abs_out
        out_path = os.path.abspath(str(out_path))

        if not csv_files:
            print("未找到可追加的 CSV 文件。")
            return None

        print(f"\n⚡ fast-append 模式：将 {len(csv_files)} 个 CSV 按行追加到：{out_path}")

        header = None
        with open(out_path, "w", encoding="utf-8", newline="") as fout:
            for idx, fpath in enumerate(csv_files):
                with open(fpath, "r", encoding="utf-8", errors="ignore", newline="") as fin:
                    h = fin.readline()
                    if not h:
                        continue
                    if idx == 0:
                        header = h
                        fout.write(header)
                    else:
                        if header is not None and h != header:
                            print(f"警告: 表头不一致，文件={fpath} 将仍继续追加（可能影响下游解析）")
                        # header mismatch 时仍跳过当前文件表头，直接追加内容
                    for line in fin:
                        fout.write(line)

        print("✅ fast-append 完成。")
        return out_path
    
    # 调用合并函数处理文件列表
    return _merge_csv_files_from_list(
        csv_files,
        log_dir,
        output_file,
        add_fall_type_column,
        write_bin=write_bin,
        min_force_n=min_force_n,
    )

def _merge_csv_files_from_list(
    csv_files,
    log_dir,
    output_file=None,
    add_fall_type_column=False,
    write_bin=False,
    min_force_n=None,
):
    """
    从文件列表合并CSV文件的内部函数
    
    Args:
        csv_files: CSV文件路径列表
        log_dir: 日志目录路径（用于解析摔倒类型信息）
        output_file: 输出文件名，如果为None则自动生成
        add_fall_type_column: 是否添加"摔倒类型-方向-csv时间编号"列
        
    Returns:
        输出文件路径
    """
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
            df = load_contact_file(csv_file)
            read_time = time.time() - file_start_time
            print(f"  ✅ 读取完成: {len(df)} 行数据 (耗时: {read_time:.2f}秒)")

            df = _filter_rows_min_force(
                df, min_force_n, basename_for_log=os.path.basename(csv_file)
            )
            if len(df) == 0:
                print(f"  ⚠️  最小力过滤后无数据，跳过该文件")
                continue
            
            # 添加源文件信息 - 使用文件名（不含扩展名）作为标识
            source_name = os.path.splitext(os.path.basename(csv_file))[0]  # 去掉.csv扩展名
            df['source_file'] = source_name
            
            # 如果需要，添加摔倒类型列
            if add_fall_type_column:
                fall_type_info = parse_fall_type_info(csv_file, log_dir)
                if fall_type_info:
                    df['fall_type_info'] = fall_type_info
                else:
                    # 如果无法解析，使用默认值
                    df['fall_type_info'] = 'unknown-unknown-unknown'
                    print(f"  ⚠️  警告: 无法解析文件 {os.path.basename(csv_file)} 的摔倒类型信息，使用默认值")
            
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
        folder_name = os.path.basename(os.path.abspath(log_dir))
        ext_out = ".bin" if write_bin else ".csv"
        output_file = os.path.join(log_dir, f"merged_contact_data_{folder_name}_{timestamp}{ext_out}")
    elif not os.path.isabs(output_file):
        # 如果输出文件不是绝对路径，检查是否已经是完整路径（包含log_dir）
        # 如果output_file已经包含log_dir的路径，直接使用；否则拼接
        abs_log_dir = os.path.abspath(log_dir)
        abs_output_file = os.path.abspath(output_file)
        if not abs_output_file.startswith(abs_log_dir):
            # 如果输出文件路径不在log_dir下，则保存到log_dir目录中
            output_file = os.path.join(log_dir, output_file)
        else:
            # 如果已经在log_dir下，直接使用（可能是从子目录合并时传入的完整路径）
            output_file = abs_output_file
    
    print(f"\n{'='*60}")
    print(f"💾 正在保存到: {output_file} ({'binary' if write_bin else 'CSV'})")
    print(f"{'='*60}")
    save_start_time = time.time()
    total_rows = len(merged_df)

    if write_bin:
        cols_drop = [c for c in ("source_file", "fall_type_info") if c in merged_df.columns]
        df_bin = merged_df.drop(columns=cols_drop, errors="ignore")
        print(f"📊 总行数: {total_rows:,} → 写入 .bin（与仿真 contact_data.bin 同格式，fig4_violin 可读）")
        write_contact_data_bin(df_bin, output_file, num_joints=0)
        save_time = time.time() - save_start_time
    else:
        chunk_size = max(1000, total_rows // 20)
        print(f"📊 总行数: {total_rows:,}")
        print(f"📦 分块大小: {chunk_size:,} 行/块")
        print(f"🔄 预计分块数: {(total_rows + chunk_size - 1) // chunk_size}")

        def save_with_progress():
            import io

            chunks_written = 0
            total_chunks = (total_rows + chunk_size - 1) // chunk_size
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk_df = merged_df.iloc[start_idx:end_idx]
                chunk_buffer = io.StringIO()
                chunk_df.to_csv(chunk_buffer, index=False, header=False)
                chunk_data = chunk_buffer.getvalue()
                chunks_written += 1
                progress = (chunks_written / total_chunks) * 100
                bar_length = 30
                filled_length = int(bar_length * chunks_written // total_chunks)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                elapsed = time.time() - save_start_time
                if chunks_written > 1:
                    avg_time_per_chunk = elapsed / chunks_written
                    remaining_chunks = total_chunks - chunks_written
                    estimated_remaining = remaining_chunks * avg_time_per_chunk
                    print(
                        f"\r  💾 保存进度: [{bar}] {progress:.1f}% ({chunks_written}/{total_chunks}) "
                        f"⏱️ 已用: {elapsed:.1f}s 预计剩余: {estimated_remaining:.1f}s",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  💾 保存进度: [{bar}] {progress:.1f}% ({chunks_written}/{total_chunks}) "
                        f"⏱️ 已用: {elapsed:.1f}s",
                        end="",
                        flush=True,
                    )
                with open(output_file, "a" if chunks_written > 1 else "w") as f:
                    if chunks_written == 1:
                        f.write(merged_df.head(0).to_csv(index=False))
                    f.write(chunk_data)
            print()

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
    
    # 如果添加了摔倒类型列，显示统计信息
    if add_fall_type_column and 'fall_type_info' in merged_df.columns:
        print(f"\n{'='*60}")
        print(f"🏷️  摔倒类型列信息")
        print(f"{'='*60}")
        fall_type_stats = merged_df.groupby('fall_type_info').size().sort_values(ascending=False)
        print(f"✅ 摔倒类型列已添加到数据中，列名: 'fall_type_info'")
        print(f"📊 包含 {len(fall_type_stats)} 个不同的摔倒类型组合")
        for fall_type, count in fall_type_stats.items():
            percentage = (count / len(merged_df)) * 100
            print(f"  🏷️  {fall_type}: {count} 行 ({percentage:.1f}%)")
    
    return output_file

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='合并多个contact CSV文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 合并指定目录下的CSV文件（如方向子目录）
  python3 merge_contact_data.py logs/test_poweroff_100/backward-00.0N-0.5s-20251021_213710
  
  # 合并父目录下的所有CSV文件（会递归查找子目录）
  python3 merge_contact_data.py logs/test_poweroff_100
  
  # 添加摔倒类型列
  python3 merge_contact_data.py logs/test_poweroff_100/backward-00.0N-0.5s-20251021_213710 --add-fall-type
  
  # 指定输出文件名
  python3 merge_contact_data.py logs/test_poweroff_100 merged_output.csv --add-fall-type
  python3 merge_contact_data.py logs/test_poweroff_100 --bin
  python3 merge_contact_data.py logs/test_poweroff_100 merged_merged.bin
  
  # 合并已经合并过的CSV文件（二次合并）
  python3 merge_contact_data.py logs/test_poweroff_100 --pattern "merged_contact_data_*.csv"

  # 丢弃 |force_normal| < 400 N 的行再合并（减小大表体积）
  python3 merge_contact_data.py logs/8dir_run --group-by-direction --min-force-n 400
        '''
    )
    parser.add_argument('log_directory', help='包含CSV文件的目录（可以是方向子目录或父目录）')
    parser.add_argument('output_file', nargs='?', default=None, help='输出文件名（可选）')
    parser.add_argument('--add-fall-type', action='store_true', 
                       help='添加"摔倒类型-方向-csv时间编号"列（格式：poweroff-backward-20251021_213713）')
    parser.add_argument('--pattern', default='contact_data_*.csv',
                       help='默认 contact_data_*.csv 会同时匹配 contact_data.csv/.bin（无时间戳）。'
                            '也可用 merged_contact_data_*.csv 等自定义模式')
    parser.add_argument(
        "--bin",
        action="store_true",
        help="输出 merged_contact_data*.bin（与仿真二进制同格式，fig4_violin / mujoco_data_io 可读）；默认 CSV",
    )
    parser.add_argument(
        "--group-by-direction",
        action="store_true",
        help="当递归合并 contact_data_*.{csv,bin} 时，按 8 方向（forward/forward_left/...）聚合输出；"
             "适用于 8dir-... 这类批量采集根目录（否则会按子目录分组输出大量文件）",
    )
    parser.add_argument(
        "--fast-append",
        action="store_true",
        help="二次合并专用：当你把已经按方向合并好的 merged_contact_data_*.csv 再合成一个大文件时，"
             "启用此模式会跳过 pandas concat 与统计，使用按行追加（只支持 CSV）。",
    )
    parser.add_argument(
        "--min-force-n",
        type=float,
        default=None,
        metavar="N",
        help="合并前丢弃力过小的行（单位 N）。优先按 force_normal 的绝对值比较，无该列则用 force_magnitude。"
             "例如 400 表示只保留 |force_normal|≥400 的行。与 --fast-append 不兼容（fast-append 不按行解析）。",
    )
    args = parser.parse_args()
    log_dir = args.log_directory
    output_file = args.output_file
    write_bin = args.bin or (output_file is not None and str(output_file).lower().endswith(".bin"))
    if args.bin and output_file and not str(output_file).lower().endswith(".bin"):
        output_file = os.path.splitext(output_file)[0] + ".bin"
    add_fall_type_column = args.add_fall_type
    file_pattern = args.pattern
    
    if not os.path.exists(log_dir):
        print(f"错误: 目录 {log_dir} 不存在")
        sys.exit(1)
    
    if add_fall_type_column:
        print("✅ 已启用: 将添加'摔倒类型-方向-csv时间编号'列")
    if args.min_force_n is not None:
        if args.fast_append:
            print("错误: --min-force-n 与 --fast-append 不能同时使用（fast-append 不按行解析 CSV）")
            sys.exit(1)
        print(f"✅ 已启用: 最小力过滤 ≥ {args.min_force_n:g} N（force_normal 优先，取绝对值）")
    
    # 合并CSV文件
    merged_file = merge_contact_csv_files(
        log_dir,
        output_file,
        add_fall_type_column,
        file_pattern,
        write_bin=write_bin,
        group_by_direction=args.group_by_direction,
        fast_append=args.fast_append,
        min_force_n=args.min_force_n,
    )
    
    if merged_file:
        print(f"\n✅ 合并成功! 输出文件: {merged_file}")
        print(f"\n现在可以使用以下命令来可视化合并后的数据:")
        print(
            f"python3 scripts/mujoco_xml_contact_display.py {merged_file} "
            f"src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1500 \"\" true true"
        )
        print("\n现在可以使用以下命令来生成护具厚度图与 YZ 护具 TSV（默认写出，文件名带 CSV 父目录时间后缀，如 yz_map_front_20260317_122531.tsv）:")
        print(
            f"python3 scripts/plot_contact_grid.py {merged_file} --target-force 1.0 "
            f"--no-hardcode --no-force-filter  # target-force 单位 kN"
        )
        print("\n现在可以使用以下命令来绘制各部位 force_normal 小提琴图:")
        print(f"python3 scripts/fig4_violin.py {merged_file}")
    else:
        print("❌ 合并失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
