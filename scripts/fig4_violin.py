import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import re
import os

# 检查并设置可用字体
def get_available_font(font_names):
    """检查字体是否可用，返回第一个可用的字体名称"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in font_names:
        if font_name in available_fonts:
            return font_name
    # 如果都不可用，返回默认字体
    return 'DejaVu Sans'

# 设置字体
TIMES_FONT = get_available_font(['Times New Roman', 'TimesNewRoman', 'Nimbus Roman', 'DejaVu Serif'])
MYRIAD_FONT = get_available_font(['Myriad Pro', 'MyriadPro', 'DejaVu Sans'])

print(f"字体设置: Times字体={TIMES_FONT}, Myriad字体={MYRIAD_FONT}")

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def plot_violin_with_median(csv_path,
                            x_col='class',
                            y_col='speed',
                            figsize=(15, 10),
                            sep=','):
    """
    从 csv 读入数据，按 x_col 画 y_col 的小提琴图，
    并在每个小提琴上叠加该类的中位数点。
    """
    # 读数据
    data = pd.read_csv(csv_path, sep=sep)

    # 计算每一类的中位数
    medians = (
        data
        .groupby(x_col)[y_col]
        .median()
        .reindex(data[x_col].unique())   # 保持类别顺序和图一致
        .values
    )

    # 画图
    sns.set_theme(style="darkgrid", font=TIMES_FONT, font_scale=3)

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        x=x_col,
        y=y_col,
        data=data,
        linewidth=0.2,
        palette="pastel",
        ax=ax
    )

    # 叠加中位数点
    inds = np.arange(len(medians))
    ax.scatter(inds, medians, marker='o', color='white', s=60, zorder=3)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    plt.tight_layout()
    plt.show()


def get_group_name(row):
    """根据规则将 link 分组，不区分左右"""
    link_name = str(row["body2_name"]).lower()
    robot_z = row.get("robot_frame_z", 0) if pd.notna(row.get("robot_frame_z")) else 0
    
    # 1. Shoulder: shoulder_pitch, shoulder_roll
    if "shoulder_pitch" in link_name or "shoulder_roll" in link_name:
        return "Shoulder"
    
    # 2. Elbow: shoulder_yaw, elbow_yaw, elbow_pitch
    if "shoulder_yaw" in link_name or "elbow_yaw" in link_name or "elbow_pitch" in link_name:
        return "Elbow"
    
    # 3. Torso: torso, 不分左右
    if "torso" in link_name:
        return "Torso"
    
    # 4. Hip: base, hip_pitch, hip_roll, hip_yaw (robot_frame_z > 0.55m)
    if "base" in link_name:
        return "Hip"
    if "hip_pitch" in link_name or "hip_roll" in link_name:
        return "Hip"
    if "hip_yaw" in link_name and robot_z > 0.55:
        return "Hip"
    
    # 5. Knee: hip_yaw (robot_frame_z < 0.55m), knee_pitch
    if "hip_yaw" in link_name and robot_z < 0.55:
        return "Knee"
    if "knee_pitch" in link_name:
        return "Knee"
    
    # Default: return original link name
    return row['body2_name']


def read_contact_data(csv_path):
    """
    读取接触数据 CSV 文件，参考 violin_link_force.py 的读取方式
    返回包含 body2_name, force_normal, group_name 的 DataFrame
    """
    print(f"[步骤 1/6] 检查文件是否存在...")
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    print(f"  ✓ 文件存在: {csv_path}")
    
    print(f"[步骤 2/6] 读取 CSV 文件...")
    file_size = os.path.getsize(csv_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.2f} MB")
    
    # First, check what columns are available in the CSV
    print(f"  检查 CSV 文件中的列...")
    try:
        sample_df = pd.read_csv(csv_path, nrows=1, engine='python', encoding='utf-8')
    except:
        try:
            sample_df = pd.read_csv(csv_path, nrows=1, engine='python', encoding='latin-1')
        except:
            sample_df = pd.DataFrame()
    
    all_csv_columns = list(sample_df.columns) if not sample_df.empty else []
    print(f"  CSV 文件中的所有列: {all_csv_columns}")
    
    # Determine which columns to read
    columns_to_read = set()
    if "body1_name" in all_csv_columns:
        columns_to_read.add("body1_name")
    columns_to_read.add("body2_name")
    if "force_normal" in all_csv_columns:
        columns_to_read.add("force_normal")
    if "normal_force" in all_csv_columns:
        columns_to_read.add("normal_force")
    if "robot_frame_x" in all_csv_columns:
        columns_to_read.add("robot_frame_x")
    if "robot_frame_y" in all_csv_columns:
        columns_to_read.add("robot_frame_y")
    if "robot_frame_z" in all_csv_columns:
        columns_to_read.add("robot_frame_z")
    
    print(f"  将读取以下列: {sorted(columns_to_read)}")
    use = lambda c: c in columns_to_read
    
    # Read with progress bar if tqdm is available
    if HAS_TQDM:
        chunk_size = 50000
        chunks = []
        
        print(f"  正在计算文件总行数...")
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f) - 1
        except:
            with open(csv_path, 'r', encoding='latin-1', errors='ignore') as f:
                total_lines = sum(1 for _ in f) - 1
        
        print(f"  总行数: {total_lines:,}, 开始读取...")
        
        try:
            reader = pd.read_csv(csv_path, usecols=use, engine='python', encoding='utf-8', chunksize=chunk_size)
            total_chunks = (total_lines // chunk_size) + 1
            for chunk in tqdm(reader, total=total_chunks, desc="  读取进度", unit="块", ncols=80):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
            print(f"  ✓ 使用 UTF-8 编码成功读取")
        except (UnicodeDecodeError, UnicodeError):
            reader = pd.read_csv(csv_path, usecols=use, engine='python', encoding='latin-1', chunksize=chunk_size)
            chunks = []
            total_chunks = (total_lines // chunk_size) + 1
            for chunk in tqdm(reader, total=total_chunks, desc="  读取进度", unit="块", ncols=80):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
            print(f"  ✓ 使用 latin-1 编码成功读取")
    else:
        print(f"  正在读取文件（这可能需要一些时间，建议安装 tqdm: pip install tqdm）...")
        try:
            data = pd.read_csv(csv_path, usecols=use, engine='python', encoding='utf-8')
            print(f"  ✓ 使用 UTF-8 编码成功读取")
        except (UnicodeDecodeError, UnicodeError):
            data = pd.read_csv(csv_path, usecols=use, engine='python', encoding='latin-1')
            print(f"  ✓ 使用 latin-1 编码成功读取")
    
    print(f"  原始数据行数: {len(data):,}")
    print(f"  实际读取的列: {list(data.columns)}")
    
    # Unify force column name
    if "force_normal" not in data.columns and "normal_force" in data.columns:
        data = data.rename(columns={"normal_force": "force_normal"})
        print(f"  ✓ 统一列名: normal_force -> force_normal")
    
    # Clean data
    print(f"[步骤 3/6] 清理数据...")
    cols_to_keep = ["body2_name", "force_normal"]
    if "body1_name" in data.columns:
        cols_to_keep.append("body1_name")
    if "robot_frame_x" in data.columns:
        cols_to_keep.append("robot_frame_x")
    if "robot_frame_y" in data.columns:
        cols_to_keep.append("robot_frame_y")
    if "robot_frame_z" in data.columns:
        cols_to_keep.append("robot_frame_z")
    
    df = data[cols_to_keep].copy()
    df["force_normal"] = pd.to_numeric(df["force_normal"], errors="coerce")
    df["body2_name"] = df["body2_name"].astype(str)
    if "body1_name" in df.columns:
        df["body1_name"] = df["body1_name"].astype(str)
    if "robot_frame_x" in df.columns:
        df["robot_frame_x"] = pd.to_numeric(df["robot_frame_x"], errors="coerce")
    if "robot_frame_y" in df.columns:
        df["robot_frame_y"] = pd.to_numeric(df["robot_frame_y"], errors="coerce")
    if "robot_frame_z" in df.columns:
        df["robot_frame_z"] = pd.to_numeric(df["robot_frame_z"], errors="coerce")
    
    df_before = len(df)
    df = df.dropna(subset=["body2_name", "force_normal"])
    print(f"  清理前: {df_before} 行, 清理后: {len(df)} 行")
    
    # Filter out ankle and head
    print(f"[步骤 4/6] 过滤 ankle 和 head...")
    df_before_filter = len(df)
    mask_keep = ~df["body2_name"].str.contains(r"ankle|head", flags=re.IGNORECASE, na=False)
    df = df.loc[mask_keep].copy()
    print(f"  过滤前: {df_before_filter} 行, 过滤后: {len(df)} 行")
    
    # Group links
    print(f"[步骤 5/6] 根据规则对 link 进行分组...")
    df["group_name"] = df.apply(get_group_name, axis=1)
    print(f"  分组完成，共 {df['group_name'].nunique()} 个不同的组")
    
    # Show group statistics
    group_counts = df["group_name"].value_counts().sort_index()
    print(f"  分组统计:")
    for group, count in group_counts.items():
        print(f"    {group}: {count} 条数据")
    
    return df


def plot_force_normal_violin(csv_path, figsize=(11.5, 5), dpi=300, output_path=None, force_max=None, force_min_kN=0.0, use_quantile=True):
    """
    从接触数据 CSV 文件读入数据（body2_name 和 force_normal），
    对 body2_name 进行分类后绘制小提琴图。
    每个部位只绘制一个小提琴图（没有 w/wo 分组）。
    
    参数:
        csv_path: CSV 文件路径
        figsize: 图片尺寸（厘米），默认 (11.5, 5) cm
        dpi: 图片分辨率，默认 300
        output_path: 输出 PNG 文件路径，如果为 None 则自动生成
        force_max: force_normal 的最大值，超过此值的数据将被过滤（如果为 None 则不过滤，单位：N）
        force_min_kN: force_kN 的最小值，小于此值的数据将被过滤（默认 0.0，单位：kN）
        use_quantile: 如果为 True，使用99分位数设置y轴范围（更好地显示中位数附近的数据）；
                      如果为 False，使用实际最大值（显示所有数据，但可能有极值压缩）
    """
    # 将厘米转换为英寸（matplotlib 使用英寸）
    figsize_inches = (figsize[0] / 2.54, figsize[1] / 2.54)
    
    # 读取数据
    print(f"[步骤 1/7] 读取接触数据...")
    data = read_contact_data(csv_path)
    
    # 将 force_normal 从 N 转换为 kN（除以1000）
    print(f"[步骤 2/7] 准备绘图数据...")
    data['force_kN'] = data['force_normal'] / 1000.0  # 将 N 转换为 kN
    print("  已将力从N转换为kN（除以1000）")
    
    # 过滤小于 force_min_kN 的数据
    if force_min_kN > 0:
        before_filter_min = len(data)
        data = data[data['force_kN'] >= force_min_kN].copy()
        print(f"  过滤 force_kN < {force_min_kN}kN: {before_filter_min - len(data)} 行")
        print(f"  剩余有效数据: {len(data)} 行")
    
    # 如果 force_max 指定，过滤数据（force_max单位是N，需要转换为kN进行比较）
    if force_max is not None:
        before_filter = len(data)
        force_max_kN = force_max / 1000.0  # 将force_max从N转换为kN
        data = data[data['force_kN'] <= force_max_kN].copy()
        print(f"  过滤 force_kN > {force_max_kN}kN (即 force_normal > {force_max}N): {before_filter - len(data)} 行")
        print(f"  剩余有效数据: {len(data)} 行")
    
    # 定义分组排序顺序（不区分左右）
    def group_sort_key(group_name):
        order = {
            "Shoulder": 1,
            "Elbow": 2,
            "Torso": 3,
            "Hip": 4,
            "Knee": 5,
        }
        return order.get(group_name, 99)
    
    # 获取所有分组并排序
    all_groups = sorted(data['group_name'].unique(), key=group_sort_key)
    print(f"  找到 {len(all_groups)} 个分组: {all_groups}")
    
    # 画图
    print(f"[步骤 3/7] 正在绘制小提琴图...")
    print(f"  图片尺寸: {figsize[0]:.2f}cm x {figsize[1]:.2f}cm ({figsize_inches[0]:.2f}in x {figsize_inches[1]:.2f}in)")
    sns.set_theme(style="white", font=MYRIAD_FONT, font_scale=1)  # 使用white主题，无网格背景
    
    fig = plt.figure(figsize=figsize_inches)
    # 设置背景透明
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    
    # 设置页边距和坐标轴位置
    ax = fig.add_axes([0.12, 0.15, 0.87, 0.75])
    # 设置 axes 背景透明
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0)
    ax.set_axisbelow(False)  # 确保网格线在图形下方
    
    # 设置字体大小
    label_fontsize = 12  # 12pt
    tick_fontsize = 10   # 10pt
    
    # 准备数据：为每个分组准备数据列表
    datasets = []
    group_positions = []
    group_labels = []  # 保存对应的分组名称
    for i, group in enumerate(all_groups, 1):
        group_data = data[data['group_name'] == group]['force_kN'].values
        if len(group_data) > 0:
            datasets.append(group_data)
            group_positions.append(i)
            group_labels.append(group)
    
    # 输出各小提琴图的值范围统计和计算宽度
    print(f"[步骤 3.5/7] 各小提琴图数据范围统计（单位：kN）...")
    all_min = []
    all_max = []
    group_counts = {}  # 存储每个分组的数据量
    for i, (pos, group) in enumerate(zip(group_positions, group_labels)):
        group_data = data[data['group_name'] == group]['force_kN'].values
        if len(group_data) > 0:
            min_val = group_data.min()
            max_val = group_data.max()
            median_val = np.median(group_data)
            q25 = np.percentile(group_data, 25)
            q75 = np.percentile(group_data, 75)
            all_min.append(min_val)
            all_max.append(max_val)
            group_counts[group] = len(group_data)
            print(f"  {group:12s}: 最小值={min_val:8.3f} kN, 最大值={max_val:8.3f} kN, "
                  f"中位数={median_val:8.3f} kN, Q25={q25:8.3f} kN, Q75={q75:8.3f} kN, "
                  f"数据量={len(group_data):,}")
    if len(all_min) > 0:
        global_min = min(all_min)
        global_max = max(all_max)
        print(f"  全局范围: 最小值={global_min:.3f} kN, 最大值={global_max:.3f} kN")
    
    # 输出每个部位前10个最大力的详细信息
    print(f"[步骤 3.6/7] 各部位前10个最大力详细信息...")
    for group in group_labels:
        group_df = data[data['group_name'] == group].copy()
        if len(group_df) > 0:
            # 按force_kN降序排序，取前10个
            top10 = group_df.nlargest(10, 'force_kN')
            print(f"\n  {group} 部位前10个最大力:")
            print(f"    {'排名':<4} {'body1_name':<30} {'body2_name':<30} {'力值(kN)':<12} {'robot_frame_x':<15} {'robot_frame_y':<15} {'robot_frame_z':<15}")
            print(f"    {'-'*4} {'-'*30} {'-'*30} {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
            for idx, (_, row) in enumerate(top10.iterrows(), 1):
                body1 = row.get('body1_name', 'N/A') if 'body1_name' in row else 'N/A'
                body2 = row.get('body2_name', 'N/A')
                force = row.get('force_kN', 0.0)
                x = row.get('robot_frame_x', 'N/A') if 'robot_frame_x' in row else 'N/A'
                y = row.get('robot_frame_y', 'N/A') if 'robot_frame_y' in row else 'N/A'
                z = row.get('robot_frame_z', 'N/A') if 'robot_frame_z' in row else 'N/A'
                
                # 格式化数值
                force_str = f"{force:.3f}" if isinstance(force, (int, float)) else str(force)
                x_str = f"{x:.3f}" if isinstance(x, (int, float)) and pd.notna(x) else str(x)
                y_str = f"{y:.3f}" if isinstance(y, (int, float)) and pd.notna(y) else str(y)
                z_str = f"{z:.3f}" if isinstance(z, (int, float)) and pd.notna(z) else str(z)
                
                print(f"    {idx:<4} {str(body1):<30} {str(body2):<30} {force_str:<12} {x_str:<15} {y_str:<15} {z_str:<15}")
    
    # 计算每个小提琴图的宽度（根据数据量归一化）
    print(f"[步骤 3.7/7] 计算小提琴图宽度（根据数据量）...")
    if len(group_counts) > 0:
        max_count = max(group_counts.values())
        base_width = 0.6  # 基础宽度
        violin_widths = []
        for group in group_labels:
            count = group_counts.get(group, 0)
            if max_count > 0:
                width_ratio = count / max_count  # 相对于最大数据量的比例
                width = base_width * width_ratio
            else:
                width = base_width
            violin_widths.append(width)
            print(f"  {group:12s}: 数据量={count:8,}, 宽度比例={width_ratio:.4f}, 宽度={width:.4f}")
        print(f"  最大数据量: {max_count:,}, 基础宽度: {base_width:.4f}")
    else:
        violin_widths = [0.6] * len(group_positions)
    
    # 使用 matplotlib 的 violinplot 绘制（不使用 hue，每个部位一个小提琴）
    print(f"[步骤 4/7] 绘制小提琴图...")
    parts = ax.violinplot(
        dataset=datasets,
        positions=group_positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=violin_widths  # 使用根据数据量计算的宽度
    )
    
    # 设置小提琴图样式（使用单一颜色，参考 fig5_violin.py 的 w 颜色）
    color = '#F1B584'  # w 的颜色
    edge_color = '#E46C0A'  # w 的轮廓颜色
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor(edge_color)
        pc.set_linewidth(1)  # 轮廓线宽度1pt
    
    # 绘制四分位数线
    print(f"[步骤 5/7] 绘制四分位数线和中位数点...")
    for i, (pos, group) in enumerate(zip(group_positions, group_labels)):
        group_data = data[data['group_name'] == group]['force_kN'].values
        if len(group_data) > 0:
            quartiles = np.percentile(group_data, [25, 50, 75])
            # 绘制四分位数线
            ax.plot([pos, pos], [quartiles[0], quartiles[2]], 
                   color=edge_color, linewidth=0.5, zorder=2)
            ax.plot([pos, pos], [quartiles[1], quartiles[1]], 
                   color=edge_color, linewidth=0.5, zorder=2)
            # 绘制中位数点（白圈）
            median_val = quartiles[1]
            ax.scatter(pos, median_val, marker='o', color='white', 
                      s=np.pi * 5, zorder=3, edgecolors='black', linewidths=0.5)
    
    # 设置横纵坐标标签
    ax.set_xlabel('')  # 不显示 x 轴标签
    ax.set_ylabel('Force (kN)', fontsize=label_fontsize, fontfamily=MYRIAD_FONT)
    # ax.set_title('Force Distribution by Body Part', fontsize=label_fontsize, fontfamily=MYRIAD_FONT)
    
    # 设置刻度标签
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
    # 确保 y 轴刻度标签使用 Times 字体
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
    
    # 设置横坐标标签（使用 group_labels，确保与 group_positions 对应）
    ax.set_xticks(group_positions)
    # 将每个标签的首字母大写
    capitalized_labels = [label.capitalize() for label in group_labels]
    ax.set_xticklabels(capitalized_labels, rotation=0, ha='center', fontfamily=MYRIAD_FONT)
    ax.tick_params(axis='x', pad=0)
    
    # 设置坐标轴线的宽度为1pt，颜色为黑色
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    # 隐藏上边和右边的轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置 y 轴范围和刻度（单位：kN）
    # y轴最小值从过滤值开始
    y_min = force_min_kN
    y_min_data = data['force_kN'].min() if len(data) > 0 else force_min_kN
    y_max_data = data['force_kN'].max() if len(data) > 0 else 10.0
    
    if use_quantile:
        # 使用99分位数来设置y轴范围，这样可以更好地显示中位数附近的数据，
        # 让小提琴图的形状和分布细节更清晰可见
        y_max_99 = data['force_kN'].quantile(0.99) if len(data) > 0 else 10.0
        y_max = max(y_max_99 * 1.1, y_min + 5.0)
        method_str = "99分位数"
    else:
        # 使用实际的最大值，确保显示所有数据
        y_max = max(y_max_data * 1.1, y_min + 5.0)
        method_str = "实际最大值"
        y_max_99 = None
    
    print(f"[步骤 6/7] 设置y轴范围...")
    print(f"  数据范围: [{y_min_data:.3f}, {y_max_data:.3f}] kN")
    if use_quantile and y_max_99 is not None:
        print(f"  99分位数: {y_max_99:.3f} kN")
    print(f"  y轴范围: [{y_min:.3f}, {y_max:.3f}] kN (基于{method_str})")
    if y_max_data > y_max and len(data) > 0:
        num_outliers = len(data[data['force_kN'] > y_max])
        pct_outliers = num_outliers / len(data) * 100
        print(f"  警告: 有 {num_outliers} 个数据点 ({pct_outliers:.2f}%) 超出y轴范围")
    
    # 根据数据范围自动设置y轴刻度（从y_min开始）
    y_range = y_max - y_min
    if y_range <= 5:
        y_ticks = np.linspace(y_min, y_max, 5)
    elif y_range <= 10:
        # 从y_min开始，步长为2
        step = 2.0
        y_ticks = np.arange(y_min, y_max + step, step)
    elif y_range <= 20:
        # 从y_min开始，步长为5
        step = 5.0
        y_ticks = np.arange(y_min, y_max + step, step)
    elif y_range <= 50:
        # 从y_min开始，步长为10
        step = 10.0
        y_ticks = np.arange(y_min, y_max + step, step)
    else:
        # 对于更大的范围，使用更灵活的刻度
        y_ticks = np.linspace(y_min, y_max, 5)
    y_ticks = [int(t) if t == int(t) else round(t, 1) for t in y_ticks]
    print(f"  y轴刻度: {y_ticks}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='y', which='major', length=2, width=1, 
                   color='black', direction='out', left=True, labelleft=True)
    
    # 自动生成输出文件名（如果未指定）
    print(f"[步骤 7/7] 正在保存图片...")
    import os
    if output_path is None:
        # 获取 CSV 文件所在目录
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        # 获取 CSV 文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        # 生成完整路径，保存到 CSV 文件所在目录
        output_path = os.path.join(csv_dir, f'{base_name}_force_normal_violin.png')
    
    # 生成 SVG 文件名（与 PNG 在同一目录）
    svg_output_path = os.path.splitext(output_path)[0] + '.svg'
    
    # 保存 PNG 图片
    plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0, transparent=True)
    print(f"  PNG 图片已保存至: {output_path}")
    
    # 保存 SVG 图片
    plt.savefig(svg_output_path, format='svg', bbox_inches=None, pad_inches=0, transparent=True)
    print(f"  SVG 图片已保存至: {svg_output_path}")
    
    print(f"  图片尺寸: {figsize[0]:.2f}cm x {figsize[1]:.2f}cm, DPI: {dpi}")
    plt.close()
    print("✓ 所有步骤完成！")


def plot_pixel_pressures_violin(csv_path, figsize=(11.5, 5), dpi=300, group_by='body_part', hue='status', output_path=None):
    """
    从像素力 CSV 文件读入数据，绘制分组小提琴图。
    假设 CSV 文件包含 body_part, status, force_normal 或 force 列（单位N）。
    按 body_part 分组，每个部位显示两个小提琴图（有护具/无护具）。
    
    参数:
        csv_path: CSV 文件路径
        figsize: 图片尺寸（厘米），默认 (18.54, 4) cm
        dpi: 图片分辨率，默认 300
        group_by: 分组列名，默认 'body_part'
        hue: 颜色分组列名，默认 'status'
        output_path: 输出 PNG 文件路径，如果为 None 则自动生成
    """
    # 将厘米转换为英寸（matplotlib 使用英寸）
    # 1 英寸 = 2.54 厘米
    figsize_inches = (figsize[0] / 2.54, figsize[1] / 2.54)
    # 读数据（使用逗号分隔，有表头）
    print(f"[步骤 1/6] 正在读取数据文件: {csv_path}")
    data = pd.read_csv(csv_path, sep=',')
    print(f"  读取完成，共 {len(data)} 行数据")
    
    # 确定力列名（优先使用 force_normal，然后是 force，最后是 pressure_MPa 作为兼容）
    force_col = None
    if 'force_normal' in data.columns:
        force_col = 'force_normal'
    elif 'force' in data.columns:
        force_col = 'force'
    elif 'normal_force' in data.columns:
        force_col = 'normal_force'
    elif 'pressure_MPa' in data.columns:
        # 兼容旧代码：如果有 pressure_MPa，假设需要转换（但通常不应该这样）
        force_col = 'pressure_MPa'
        print("  警告: 使用 pressure_MPa 列，假设需要转换为力")
    else:
        raise ValueError("CSV文件中未找到力相关的列（force_normal, force, normal_force）")
    
    print(f"  使用列: {force_col}")
    
    # 确保力列是数值类型
    print("[步骤 2/6] 正在清理数据...")
    data['force_kN'] = pd.to_numeric(data[force_col], errors='coerce')
    
    # 如果原始数据单位是N，转换为kN（除以1000）
    # 如果原始数据已经是kN或MPa，可能需要调整
    # 假设 force_normal/force 单位是N，需要转换为kN
    if force_col in ['force_normal', 'force', 'normal_force']:
        data['force_kN'] = data['force_kN'] / 1000.0  # N 转换为 kN
        print("  已将力从N转换为kN（除以1000）")
    elif force_col == 'pressure_MPa':
        # 如果使用pressure_MPa，假设已经是合适的单位（但这不是力的单位）
        print("  警告: 使用pressure_MPa作为力，单位可能不正确")
    
    # 删除包含 NaN 的行（如果有转换失败的数据）
    before_drop = len(data)
    data = data.dropna(subset=['force_kN'])
    print(f"  删除 NaN 值: {before_drop - len(data)} 行")
    
    # 过滤掉小于 2kN 的数据
    before_filter = len(data)
    data = data[data['force_kN'] >= 2.0]
    print(f"  过滤小于 2kN 的数据: {before_filter - len(data)} 行")
    print(f"  剩余有效数据: {len(data)} 行")
    
    # 画图
    print("[步骤 3/6] 正在绘制小提琴图...")
    print(f"  图片尺寸: {figsize[0]:.2f}cm x {figsize[1]:.2f}cm ({figsize_inches[0]:.2f}in x {figsize_inches[1]:.2f}in)")
    sns.set_theme(style="white", font=MYRIAD_FONT, font_scale=1)  # 使用white主题，无网格背景
    
    fig = plt.figure(figsize=figsize_inches)
    # 设置背景透明
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    
    # 设置页边距和坐标轴位置（单位：相对于figure的百分比）
    # 四个页边距：
    # - 左边距：12% (left = 0.12)
    # - 右边距：1% (right = 1 - left - width = 1 - 0.12 - 0.87 = 0.01)
    # - 下边距：15% (bottom = 0.15)
    # - 上边距：10% (top = 1 - bottom - height = 1 - 0.15 - 0.75 = 0.10)
    # left=0.12, bottom=0.15, width=0.87, height=0.75
    ax = fig.add_axes([0.30, 0.15, 0.70, 0.75])
    # 设置 axes 背景透明
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0)
    ax.set_axisbelow(False)  # 确保网格线在图形下方
    
    # 设置字体大小（单位：pt）
    # 横纵坐标标签和图题：12pt
    # 数字（刻度标签）：10pt
    label_fontsize = 12  # 12pt
    tick_fontsize = 10   # 10pt
    
    # 使用 hue 参数按 status 分组，每个部位显示两个小提琴图
    # 设置小提琴图填充颜色：w 为 #F1B584，wo 为 #D1D3D4（与 fig3_violin.py 一致）
    color_dict = {
        'w': '#F1B584',
        'w/o': '#D1D3D4',
        'wo': '#D1D3D4'  # 同时支持 'w/o' 和 'wo' 两种写法
    }
    # 设置轮廓线颜色：w 为橙色 #E46C0A，wo 为深灰 #666666（与 fig3_violin.py 一致）
    edge_color_dict = {
        'w': '#E46C0A',  # 橙色
        'w/o': '#666666',  # 深灰
        'wo': '#666666'   # 深灰
    }
    # 获取所有唯一的status值，并按照字典顺序创建颜色列表
    statuses_temp = sorted(data[hue].unique())
    color_palette = [color_dict.get(status, '#D1D3D4') for status in statuses_temp]
    
    print(f"  Status顺序: {statuses_temp}")
    print(f"  颜色列表: {color_palette}")
    
    # 计算每个 body_part 和 status 组合的数据量，用于按比例调整小提琴图宽度
    print("[步骤 3.1/6] 正在计算各组合的数据量...")
    count_df = data.groupby([group_by, hue]).size().reset_index(name='count')
    print("  各组合数据量统计:")
    for _, row in count_df.iterrows():
        print(f"    {row[group_by]} - {row[hue]}: {row['count']} 条数据")
    
    # 对于每个 body_part，计算 w 和 wo 的相对宽度比例
    body_parts = sorted(data[group_by].unique())
    width_ratios = {}
    for body_part in body_parts:
        part_counts = count_df[count_df[group_by] == body_part]
        if len(part_counts) > 1:
            # 计算该部位内各状态的数据量比例
            max_count = part_counts['count'].max()
            ratios = {}
            for _, row in part_counts.iterrows():
                ratios[row[hue]] = row['count'] / max_count if max_count > 0 else 1.0
            width_ratios[body_part] = ratios
            print(f"    {body_part} 宽度比例: {ratios}")
    
    violin_plot = sns.violinplot(
        x=group_by,
        y='force_kN',
        hue=hue,
        data=data,
        linewidth=0.2,
        palette=color_palette,
        hue_order=statuses_temp,  # 明确指定hue的顺序
        inner='quartiles',  # 显示四分位数线
        density_norm='count',  # 按数据量决定小提琴图的宽度（全局归一化）
        width=0.8,  # 设置整体宽度
        ax=ax
    )
    print("  小提琴图绘制完成")
    
    # 手动调整每个小提琴图的宽度，使其按照每个 body_part 内的数据量比例
    print("[步骤 3.2/6] 正在按数据量比例调整小提琴图宽度...")
    from matplotlib.patches import PathPatch
    
    # 获取所有小提琴图的路径
    violin_collections = []
    for collection in ax.collections:
        if hasattr(collection, 'get_paths') and len(collection.get_paths()) > 0:
            violin_collections.append(collection)
    
    # 为每个 body_part 和 status 组合重新绘制小提琴图，使用正确的宽度比例
    if len(violin_collections) > 0 and len(width_ratios) > 0:
        # 清除现有的小提琴图
        for collection in violin_collections:
            collection.remove()
        
        # 重新绘制，使用 matplotlib 的 violinplot 以便精确控制宽度
        x_positions = ax.get_xticks()
        violin_idx = 0
        
        for i, body_part in enumerate(body_parts):
            if i >= len(x_positions):
                continue
            x_base = x_positions[i]
            
            # 计算该部位内各状态的位置偏移
            n_statuses = len(statuses_temp)
            if n_statuses == 2:
                offsets = [-0.2, 0.2]  # 两个状态时，左右各偏移0.2
            else:
                # 多个状态时，均匀分布
                offsets = [(j - (n_statuses - 1) / 2) * 0.4 / n_statuses for j in range(n_statuses)]
            
            for j, status in enumerate(statuses_temp):
                if j >= len(offsets):
                    continue
                
                # 获取该组合的数据
                subset_data = data[(data[group_by] == body_part) & (data[hue] == status)]['force_kN'].values
                
                if len(subset_data) > 0:
                    # 计算宽度比例
                    if body_part in width_ratios and status in width_ratios[body_part]:
                        width_ratio = width_ratios[body_part][status]
                    else:
                        width_ratio = 1.0
                    
                    # 基础宽度（相对于 x 轴单位）
                    base_width = 0.3 * width_ratio
                    
                    # 绘制小提琴图
                    x_pos = x_base + offsets[j]
                    parts = ax.violinplot(
                        [subset_data],
                        positions=[x_pos],
                        widths=[base_width],
                        showmeans=False,
                        showmedians=False,
                        showextrema=False
                    )
                    
                    # 设置颜色和样式
                    for pc in parts['bodies']:
                        pc.set_facecolor(color_dict.get(status, '#D1D3D4'))
                        pc.set_alpha(0.7)
                        pc.set_edgecolor(edge_color_dict.get(status, '#666666'))
                        pc.set_linewidth(1)  # 轮廓线宽度1pt
                    
                    # 绘制四分位数线（使用与轮廓线相同的颜色）
                    quartiles = np.percentile(subset_data, [25, 50, 75])
                    edge_color = edge_color_dict.get(status, '#666666')
                    ax.plot([x_pos, x_pos], [quartiles[0], quartiles[2]], 
                           color=edge_color, linewidth=0.5, zorder=2)
                    ax.plot([x_pos, x_pos], [quartiles[1], quartiles[1]], 
                           color=edge_color, linewidth=0.5, zorder=2)
                    
                    violin_idx += 1
        
        print(f"  已重新绘制 {violin_idx} 个小提琴图，宽度按数据量比例调整")
        
        # 手动创建图例（因为清除了 seaborn 绘制的小提琴图，图例也丢失了）
        from matplotlib.patches import Patch
        legend_elements = []
        label_mapping = {'w': 'w/', 'wo': 'w/o', 'w/o': 'w/o'}
        for status in statuses_temp:
            label = label_mapping.get(status, status)
            legend_elements.append(
                Patch(facecolor=color_dict.get(status, '#D1D3D4'), 
                      edgecolor=edge_color_dict.get(status, '#666666'), 
                      linewidth=1, label=label, alpha=0.7)  # 图例轮廓线宽度1pt
            )
        legend = ax.legend(handles=legend_elements, loc='upper left', framealpha=0.5, title='')
        legend.get_frame().set_alpha(0.5)
        # 设置图例文字字体
        for text in legend.get_texts():
            text.set_fontfamily(MYRIAD_FONT)
    
    # 统一设置所有小提琴图轮廓线的宽度，确保只有一种线宽
    # matplotlib的violinplot创建的路径集合
    for collection in ax.collections:
        if hasattr(collection, 'get_paths') and len(collection.get_paths()) > 0:
            # 这是小提琴图的路径集合
            collection.set_linewidth(1)  # 统一设置为1pt
    # 也检查patches（可能有些版本使用patches）
    for patch in ax.patches:
        if isinstance(patch, PathPatch):
            patch.set_linewidth(1)  # 统一设置为1pt
    
    # 如果图例还没有创建（使用 seaborn 绘制的情况），设置图例位置和透明度
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(True)
        legend.get_frame().set_alpha(0.5)  # 设置透明度，0.0为完全透明，1.0为完全不透明
        legend.set_loc('upper left')  # 设置图例位置为右上角
        legend.set_title('')  # 去掉图例标题（去掉"status"字样）
        # 修改图例标签：w -> w/, wo -> w/o，并设置字体
        label_mapping = {'w': 'w/', 'wo': 'w/o', 'w/o': 'w/o'}
        for text in legend.get_texts():
            original_text = text.get_text()
            new_text = label_mapping.get(original_text, original_text)
            text.set_text(new_text)
            text.set_fontfamily(MYRIAD_FONT)
    
    # 计算每个 body_part 和 status 组合的中位数
    print("[步骤 4/6] 正在计算中位数...")
    medians_df = (
        data
        .groupby([group_by, hue])['force_kN']
        .median()
        .reset_index()
    )
    
    # 获取所有唯一的 body_part 和 status，并保持顺序（与重新绘制时一致）
    body_parts = sorted(data[group_by].unique())  # 使用排序以保持顺序一致
    statuses = sorted(data[hue].unique())  # 排序以确保顺序一致
    print(f"  找到 {len(body_parts)} 个部位，{len(statuses)} 种状态")
    print(f"  共计算 {len(medians_df)} 个中位数")
    
    # 通过访问 violinplot 创建的图形元素来获取每个小提琴图的准确中心位置
    # 由于我们重新绘制了小提琴图，现在使用 matplotlib 的 violinplot
    # 小提琴图存储在 ax.collections 中
    print("[步骤 5/6] 正在标记中位数点...")
    
    # 获取 x 轴刻度位置（这些是每个 body_part 的中心位置）
    x_tick_positions = ax.get_xticks()
    
    # 尝试从 collections 中获取小提琴图
    violin_collections = []
    for collection in ax.collections:
        # 检查是否是 violin plot 的 collection（通常有路径数据）
        if hasattr(collection, 'get_paths') and len(collection.get_paths()) > 0:
            violin_collections.append(collection)
    
    # 如果 collections 中没有找到，尝试从 patches 中获取
    violin_patches = [p for p in ax.patches if isinstance(p, PathPatch)]
    
    print(f"  找到 {len(violin_collections)} 个小提琴图集合，{len(violin_patches)} 个 patches")
    print(f"  x 轴刻度位置: {x_tick_positions}")
    
    # 为每个组合计算中位数并标记在图上
    # 使用重新绘制时设置的 x 位置来精确定位
    marked_count = 0
    for i, body_part in enumerate(body_parts):
        for j, status in enumerate(statuses):
            median_val = medians_df[
                (medians_df[group_by] == body_part) & 
                (medians_df[hue] == status)
            ]['force_kN'].values
            
            if len(median_val) > 0:
                # 计算该组合的y轴数据范围（kN）
                subset_data = data[(data[group_by] == body_part) & (data[hue] == status)]['force_kN']
                y_min = subset_data.min()
                y_max = subset_data.max()
                
                # 计算对应的 violin 索引（按照重新绘制时的顺序）
                # 每个 x 位置有 len(statuses) 个小提琴图
                violin_idx = i * len(statuses) + j
                
                # 优先从 violin plot 的路径中计算中心位置
                x_center = None
                if violin_idx < len(violin_collections):
                    collection = violin_collections[violin_idx]
                    paths = collection.get_paths()
                    if len(paths) > 0:
                        # 获取所有路径的顶点并计算中心
                        all_x = []
                        for path in paths:
                            vertices = path.vertices
                            # 获取 collection 的 transform
                            transform = collection.get_transform()
                            # 将顶点从路径坐标转换到数据坐标
                            if transform is not None:
                                # 先转换到显示坐标，再转换到数据坐标
                                vertices_display = transform.transform(vertices)
                                vertices_data = ax.transData.inverted().transform(vertices_display)
                            else:
                                # 如果 transform 是 None，直接转换
                                vertices_data = ax.transData.inverted().transform(vertices)
                            all_x.extend(vertices_data[:, 0])
                        
                        if len(all_x) > 0:
                            # 小提琴图是对称的，中轴线的 x 坐标应该是所有 x 坐标的中位数
                            x_center = np.median(all_x)
                
                # 如果从路径中无法获取，使用 x 轴刻度位置和 hue 偏移（与重新绘制时使用的偏移一致）
                if x_center is None:
                    if i < len(x_tick_positions):
                        x_base = x_tick_positions[i]  # 基础 x 位置（body_part 的中心）
                        # 计算 hue 的偏移（与重新绘制时使用的偏移一致）
                        n_statuses = len(statuses)
                        if n_statuses == 2:
                            offsets = [-0.2, 0.2]  # 两个状态时，左右各偏移0.2
                        else:
                            offsets = [(k - (n_statuses - 1) / 2) * 0.4 / n_statuses for k in range(n_statuses)]
                        if j < len(offsets):
                            x_center = x_base + offsets[j]
                        else:
                            hue_offset = (j - (len(statuses) - 1) / 2) * 0.2
                            x_center = x_base + hue_offset
                    else:
                        # 如果无法获取刻度位置，使用估算方法
                        x_center = i + (j - (len(statuses) - 1) / 2) * 0.2
                
                if x_center is not None:
                    # 将数据坐标转换为figure坐标（单位cm）
                    # 1. 数据坐标 -> axes坐标（0-1，相对于axes）
                    data_point = np.array([[x_center, median_val[0]]])
                    axes_point = ax.transAxes.inverted().transform(ax.transData.transform(data_point))
                    # 2. axes坐标 -> figure坐标（0-1范围）
                    axes_bbox = ax.get_position()  # axes在figure中的位置（0-1范围）
                    figure_x = axes_bbox.x0 + axes_point[0, 0] * axes_bbox.width
                    figure_y = axes_bbox.y0 + axes_point[0, 1] * axes_bbox.height
                    # 3. figure坐标（0-1）-> cm
                    violin_x_cm = figure_x * figsize[0]  # x位置（cm）
                    violin_y_cm = figure_y * figsize[1]  # y位置（cm）
                    
                    # 计算 x 坐标的范围（用于显示，使用与中心位置相同的转换方法）
                    if violin_idx < len(violin_collections):
                        collection = violin_collections[violin_idx]
                        paths = collection.get_paths()
                        if len(paths) > 0:
                            all_x_coords = []
                            for path in paths:
                                vertices = path.vertices
                                transform = collection.get_transform()
                                if transform is not None:
                                    vertices_display = transform.transform(vertices)
                                    vertices_data = ax.transData.inverted().transform(vertices_display)
                                else:
                                    vertices_data = ax.transData.inverted().transform(vertices)
                                all_x_coords.extend(vertices_data[:, 0])
                            if len(all_x_coords) > 0:
                                x_min = np.min(all_x_coords)
                                x_max = np.max(all_x_coords)
                                x_range = x_max - x_min
                            else:
                                x_min = x_max = x_range = 0
                        else:
                            x_min = x_max = x_range = 0
                    else:
                        x_min = x_max = x_range = 0
                    
                    print(f"    {body_part} - {status}:")
                    print(f"      小提琴图中轴线位置 (x, cm): {violin_x_cm:.4f} cm")
                    print(f"      白圈（中位数点）位置 (x, cm): {violin_x_cm:.4f} cm, (y, cm): {violin_y_cm:.4f} cm")
                    print(f"      中位数值 (y, kN): {median_val[0]:.4f} kN")
                    print(f"      y轴数据范围: [{y_min:.4f}, {y_max:.4f}] kN")
                    if x_range > 0:
                        print(f"      小提琴图 x 范围 (数据坐标): [{x_min:.4f}, {x_max:.4f}], 宽度: {x_range:.4f}")
                    
                    # 白圈直径12pt，半径6pt，面积 = π * r^2 = π * 36 ≈ 113.1 points^2
                    ax.scatter(x_center, median_val[0], marker='o', color='white', 
                              s=np.pi * 5, zorder=3, edgecolors='black', linewidths=0.5)
                    marked_count += 1
                else:
                    # 如果无法获取准确位置，使用估算方法（备用方案）
                    x_offset = (j - (len(statuses) - 1) / 2) * 0.2
                    x_pos = i + x_offset
                    
                    # 将数据坐标转换为figure坐标（单位cm）
                    data_point = np.array([[x_pos, median_val[0]]])
                    # 1. 数据坐标 -> axes坐标（0-1，相对于axes）
                    axes_point = ax.transAxes.inverted().transform(ax.transData.transform(data_point))
                    # 2. axes坐标 -> figure坐标（0-1范围）
                    axes_bbox = ax.get_position()  # axes在figure中的位置（0-1范围）
                    figure_x = axes_bbox.x0 + axes_point[0, 0] * axes_bbox.width
                    figure_y = axes_bbox.y0 + axes_point[0, 1] * axes_bbox.height
                    # 3. figure坐标（0-1）-> cm
                    violin_x_cm = figure_x * figsize[0]  # x位置（cm）
                    violin_y_cm = figure_y * figsize[1]  # y位置（cm）
                    
                    print(f"    {body_part} - {status}:")
                    print(f"      警告: 无法获取小提琴图位置，使用估算位置")
                    print(f"      估算中轴线位置 (x, cm): {violin_x_cm:.4f} cm")
                    print(f"      白圈位置 (x, cm): {violin_x_cm:.4f} cm, (y, cm): {violin_y_cm:.4f} cm")
                    print(f"      中位数值 (y, kN): {median_val[0]:.4f} kN")
                    print(f"      y轴数据范围: [{y_min:.4f}, {y_max:.4f}] kN")
                    # 白圈直径12pt，半径6pt，面积 = π * r^2 = π * 36 ≈ 113.1 points^2
                    ax.scatter(x_pos, median_val[0], marker='o', color='white', 
                              s=np.pi * 9, zorder=3, edgecolors='black', linewidths=0.5)
                    marked_count += 1
    print(f"  已标记 {marked_count} 个中位数点")
    
    # 设置横纵坐标标签和图题字体大小为 12pt
    ax.set_xlabel('')  # 不显示 x 轴标签
    ax.set_ylabel('Force (kN)', fontsize=label_fontsize, fontfamily=MYRIAD_FONT)
    ax.set_title('Force Distribution of Whole Body Sampling', fontsize=label_fontsize, fontfamily=MYRIAD_FONT)
    
    # 设置刻度标签（数字）字体大小为 10pt
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
    # 确保 y 轴刻度标签（数字）使用 Times 字体
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
    
    # 横坐标part名称水平不用倾斜，并将首字母大写
    # 先获取当前的刻度位置
    x_ticks = ax.get_xticks()
    # 获取当前的横轴标签
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    # 将每个标签的首字母大写
    capitalized_labels = [label.capitalize() for label in current_labels]
    # 先设置刻度位置，再设置标签
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(capitalized_labels, rotation=0, ha='center', fontfamily=MYRIAD_FONT)
    # 设置横轴和横坐标标签之间的间距为2pt
    ax.tick_params(axis='x', pad=0)
    
    # 设置坐标轴线的宽度为1pt，颜色为黑色
    # 左轴线1pt，下轴线1pt
    ax.spines['left'].set_linewidth(1)   # 1pt
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)  # 1pt
    ax.spines['bottom'].set_color('black')
    # 隐藏上边和右边的轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置左轴（y轴）的主刻度线和刻度标
    # 根据数据范围自动设置y轴刻度（从2kN开始，因为过滤了小于2kN的数据）
    y_min_data = data['force_kN'].min() if len(data) > 0 else 2.0
    y_max_data = data['force_kN'].max() if len(data) > 0 else 20.0
    # 设置合理的y轴刻度，从2开始，步长根据数据范围调整
    if y_max_data <= 10:
        y_ticks = [2, 4, 6, 8, 10]
    elif y_max_data <= 20:
        y_ticks = [2, 5, 10, 15, 20]
    elif y_max_data <= 50:
        y_ticks = [2, 10, 20, 30, 40, 50]
    else:
        # 对于更大的范围，使用更灵活的刻度
        y_ticks = np.linspace(2, y_max_data, 5)
        y_ticks = [int(t) if t == int(t) else round(t, 1) for t in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    # 设置Y轴范围，从2kN开始（因为过滤了小于2kN的数据）
    ax.set_ylim(2, max(y_max_data * 1.1, 10))
    # 确保主刻度线显示为短横线，长度10pt
    # matplotlib 的 length 参数单位是 points
    ax.tick_params(axis='y', which='major', length=2, width=1, 
                   color='black', direction='out', left=True, labelleft=True)
    
    # 自动生成输出文件名（如果未指定）
    print("[步骤 6/6] 正在保存图片...")
    import os
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f'{base_name}_violin_plot.png'
    
    # 生成 SVG 文件名
    svg_output_path = os.path.splitext(output_path)[0] + '.svg'
    
    # 保存 PNG 图片（不使用 bbox_inches='tight'，因为我们已经手动设置了页边距，背景透明）
    plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0, transparent=True)
    print(f"  PNG 图片已保存至: {output_path}")
    
    # 保存 SVG 图片（不使用 bbox_inches='tight'，因为我们已经手动设置了页边距，背景透明）
    plt.savefig(svg_output_path, format='svg', bbox_inches=None, pad_inches=0, transparent=True)
    print(f"  SVG 图片已保存至: {svg_output_path}")
    
    print(f"  图片尺寸: {figsize[0]:.2f}cm x {figsize[1]:.2f}cm, DPI: {dpi}")
    plt.close()
    print("✓ 所有步骤完成！")


# 使用示例
if __name__ == "__main__":
    # 示例1: 绘制像素压力小提琴图（原有功能）
    # plot_pixel_pressures_violin('results/fig5_all_pixel_pressures.csv')
    
    # 示例2: 绘制 force_normal 小提琴图（新功能）
    # 从 body2_name 和 force_normal 读取数据，对 body2_name 进行分类后绘制
    
    # ===== 配置参数 =====
    # 过滤小于此值的力数据（单位：kN），y轴最小值将从该值开始
    FORCE_MIN_KN = 5.0  # 过滤小于2kN的数据
    
    # y轴范围设置方式：
    # - True: 使用99分位数（更好地显示中位数附近的数据，小提琴图细节更清晰）
    # - False: 使用实际最大值（显示所有数据，但可能有极值压缩）
    USE_QUANTILE = False  # 改为 False 使用实际最大值
    
    csv_path = "/home/wang22/engineai/engineai_ros2_workspace/logs/4in1/merged_contact_data_4in1_20251026_194302.csv"
    plot_force_normal_violin(
        csv_path=csv_path,
        figsize=(13.5, 3.85),
        dpi=300,
        output_path=None,  # 自动生成文件名：{csv文件名}_force_normal_violin.png
        force_max=None,  # 可选：过滤超过此值的 force_normal，例如 force_max=10000（单位：N）
        force_min_kN=FORCE_MIN_KN,  # 过滤小于此值的力数据（单位：kN）
        use_quantile=USE_QUANTILE  # y轴范围设置方式
    )
