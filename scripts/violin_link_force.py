import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("      提示: 安装 tqdm 可以显示更好的进度条 (pip install tqdm)")

# -------------------------------
# Config
# -------------------------------
file_path = "/home/wang22/engineai/engineai_ros2_workspace/logs/4in1/merged_contact_data_4in1_20251026_194302.csv"
# Save image to the same directory as the CSV file
file_dir = Path(file_path).parent
out_path = file_dir / "force_normal_violin_single.png"

# -------------------------------
# Read ONLY the two needed columns
# -------------------------------
print(f"[1/9] 检查文件是否存在...")
if not Path(file_path).exists():
    raise FileNotFoundError(f"File not found: {file_path}")
print(f"      ✓ 文件存在: {file_path}")

print(f"[2/9] 读取 CSV 文件...")
# Get file size for progress indication
file_size = os.path.getsize(file_path)
file_size_mb = file_size / (1024 * 1024)
print(f"      文件大小: {file_size_mb:.2f} MB")

# First, check what columns are available in the CSV
print(f"      检查 CSV 文件中的列...")
try:
    sample_df = pd.read_csv(file_path, nrows=1, engine='python', encoding='utf-8')
except:
    try:
        sample_df = pd.read_csv(file_path, nrows=1, engine='python', encoding='latin-1')
    except:
        sample_df = pd.DataFrame()

all_csv_columns = list(sample_df.columns) if not sample_df.empty else []
print(f"      CSV 文件中的所有列: {all_csv_columns}")

# Determine which columns to read
columns_to_read = set()
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

print(f"      将读取以下列: {sorted(columns_to_read)}")
use = lambda c: c in columns_to_read

# Read with progress bar if tqdm is available
if HAS_TQDM:
    # Use chunksize to read in chunks and show progress
    chunk_size = 50000  # Read 50k rows at a time for better performance
    chunks = []
    
    # First, count total rows for progress bar
    print(f"      正在计算文件总行数...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
    except:
        # If UTF-8 fails, try latin-1
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
    
    print(f"      总行数: {total_lines:,}, 开始读取...")
    
    try:
        reader = pd.read_csv(file_path, usecols=use, engine='python', encoding='utf-8', chunksize=chunk_size)
        total_chunks = (total_lines // chunk_size) + 1
        for chunk in tqdm(reader, total=total_chunks, desc="      读取进度", unit="块", ncols=80):
            chunks.append(chunk)
        data = pd.concat(chunks, ignore_index=True)
        print(f"      ✓ 使用 UTF-8 编码成功读取")
    except (UnicodeDecodeError, UnicodeError):
        # Try with different encoding if UTF-8 fails
        reader = pd.read_csv(file_path, usecols=use, engine='python', encoding='latin-1', chunksize=chunk_size)
        chunks = []
        total_chunks = (total_lines // chunk_size) + 1
        for chunk in tqdm(reader, total=total_chunks, desc="      读取进度", unit="块", ncols=80):
            chunks.append(chunk)
        data = pd.concat(chunks, ignore_index=True)
        print(f"      ✓ 使用 latin-1 编码成功读取")
else:
    # Fallback: read normally but show simple progress messages
    print(f"      正在读取文件（这可能需要一些时间，建议安装 tqdm: pip install tqdm）...")
    try:
        data = pd.read_csv(file_path, usecols=use, engine='python', encoding='utf-8')
        print(f"      ✓ 使用 UTF-8 编码成功读取")
    except (UnicodeDecodeError, UnicodeError):
        # Try with different encoding if UTF-8 fails
        data = pd.read_csv(file_path, usecols=use, engine='python', encoding='latin-1')
        print(f"      ✓ 使用 latin-1 编码成功读取")

print(f"      原始数据行数: {len(data):,}")
print(f"      实际读取的列: {list(data.columns)}")

# Unify force column name
if "force_normal" not in data.columns and "normal_force" in data.columns:
    data = data.rename(columns={"normal_force": "force_normal"})
    print(f"      ✓ 统一列名: normal_force -> force_normal")

# Check if robot_frame columns exist
robot_frame_cols = [col for col in data.columns if "robot_frame" in col.lower()]
if robot_frame_cols:
    print(f"      找到 robot_frame 相关列: {robot_frame_cols}")
else:
    print(f"      警告: 未找到 robot_frame_x/y/z 列")
    # Try to find similar column names
    all_cols_lower = [col.lower() for col in data.columns]
    possible_cols = [col for col in data.columns if any(x in col.lower() for x in ["x", "y", "z", "frame", "position"])]
    if possible_cols:
        print(f"      可能相关的列: {possible_cols}")

print(f"[3/7] 清理数据...")
# Keep needed columns; clean
cols_to_keep = ["body2_name", "force_normal"]
if "robot_frame_x" in data.columns:
    cols_to_keep.append("robot_frame_x")
if "robot_frame_y" in data.columns:
    cols_to_keep.append("robot_frame_y")
if "robot_frame_z" in data.columns:
    cols_to_keep.append("robot_frame_z")
    
df = data[cols_to_keep].copy()
df["force_normal"] = pd.to_numeric(df["force_normal"], errors="coerce")
df["body2_name"]   = df["body2_name"].astype(str)
if "robot_frame_x" in df.columns:
    df["robot_frame_x"] = pd.to_numeric(df["robot_frame_x"], errors="coerce")
if "robot_frame_y" in df.columns:
    df["robot_frame_y"] = pd.to_numeric(df["robot_frame_y"], errors="coerce")
if "robot_frame_z" in df.columns:
    df["robot_frame_z"] = pd.to_numeric(df["robot_frame_z"], errors="coerce")
df_before = len(df)
df = df.dropna(subset=["body2_name", "force_normal"])
print(f"      清理前: {df_before} 行, 清理后: {len(df)} 行")

# -------------------------------
# Filter force_normal >= 2000
# -------------------------------
print(f"[4/7] 过滤 force_normal < 2000...")
df_before_force_filter = len(df)
df = df[df["force_normal"] >= 2000].copy()
print(f"      过滤前: {df_before_force_filter} 行, 过滤后: {len(df)} 行")
if df.empty:
    raise ValueError("No rows remain after filtering force_normal >= 2000.")

# -------------------------------
# Filter out ankle + head
# -------------------------------
print(f"[5/7] 过滤 ankle 和 head...")
df_before_filter = len(df)
mask_keep = ~df["body2_name"].str.contains(r"ankle|head", flags=re.IGNORECASE, na=False)
df = df.loc[mask_keep].copy()
print(f"      过滤前: {df_before_filter} 行, 过滤后: {len(df)} 行")
if df.empty:
    raise ValueError("No rows remain after filtering out ankle/shoulder and cleaning.")

# -------------------------------
# Group links according to rules
# -------------------------------
print(f"[6/7] 根据规则对 link 进行分组...")

def get_group_name(row):
    """根据规则将 link 分组，并区分左右"""
    link_name = str(row["body2_name"]).lower()
    robot_y = row.get("robot_frame_y", 0) if pd.notna(row.get("robot_frame_y")) else 0
    robot_z = row.get("robot_frame_z", 0) if pd.notna(row.get("robot_frame_z")) else 0
    
    # Determine left/right side
    # For base: y+ is left, y- is right
    # For others: check link name for left/right indicators
    is_left = False
    
    # Check link name for explicit left/right indicators
    if any(x in link_name for x in ["left", "l_", "_l", "l_shoulder", "l_elbow", "l_hip", "l_knee"]):
        is_left = True
    elif any(x in link_name for x in ["right", "r_", "_r", "r_shoulder", "r_elbow", "r_hip", "r_knee"]):
        is_left = False
    elif "base" in link_name:
        # For base: y+ is left, y- is right
        is_left = robot_y > 0 if pd.notna(robot_y) else False
    else:
        # Default: try to infer from y coordinate
        # y+ is left, y- is right
        if pd.notna(robot_y):
            is_left = robot_y > 0
        else:
            # If no y coordinate, try to infer from link name pattern
            # Common patterns: left_*, left*_*, *_left_*, etc.
            is_left = "left" in link_name or link_name.startswith("l_")
    
    side = "Left" if is_left else "Right"
    
    # 1. Shoulder: shoulder_pitch, shoulder_roll
    if "shoulder_pitch" in link_name or "shoulder_roll" in link_name:
        return f"{side}_Shoulder"
    
    # 2. Elbow: shoulder_yaw, elbow_yaw, elbow_pitch
    if "shoulder_yaw" in link_name or "elbow_yaw" in link_name or "elbow_pitch" in link_name:
        return f"{side}_Elbow"
    
    # 3. Torso: torso (no left/right)
    if "torso" in link_name:
        return "Torso"
    
    # 4. Hip: base (根据robot_frame_y区分), hip_pitch, hip_roll, hip_yaw (robot_frame_z > -0.25m)
    if "base" in link_name:
        return f"{side}_Hip"
    if "hip_pitch" in link_name or "hip_roll" in link_name:
        return f"{side}_Hip"
    if "hip_yaw" in link_name and robot_z > 0.55:
        return f"{side}_Hip"
    
    # 5. Knee: hip_yaw (robot_frame_z < -0.25m), knee_pitch (robot_frame_z >= -0.5m)
    if "hip_yaw" in link_name and robot_z < 0.55:
        return f"{side}_Knee"
    if "knee_pitch" in link_name and robot_z >= 0.23:
        return f"{side}_Knee"
    
    # 6. Crus: knee_pitch (robot_frame_z < -0.5m)
    if "knee_pitch" in link_name and robot_z < 0.23:
        return f"{side}_Crus"
    
    # Default: return original link name with side
    return f"{side}_{row['body2_name']}"

# Apply grouping
df["group_name"] = df.apply(get_group_name, axis=1)
print(f"      分组完成，共 {df['group_name'].nunique()} 个不同的组")

# Debug: Show robot_frame_z distribution for knee_pitch links
if "robot_frame_z" in df.columns:
    knee_pitch_data = df[df["body2_name"].str.contains("knee_pitch", case=False, na=False)]
    if len(knee_pitch_data) > 0:
        print(f"      调试: knee_pitch 数据统计:")
        print(f"        总数: {len(knee_pitch_data)}")
        if pd.notna(knee_pitch_data["robot_frame_z"]).any():
            print(f"        robot_frame_z 范围: [{knee_pitch_data['robot_frame_z'].min():.3f}, {knee_pitch_data['robot_frame_z'].max():.3f}]")
            print(f"        robot_frame_z < -0.5 的数量: {len(knee_pitch_data[knee_pitch_data['robot_frame_z'] < -0.5])}")
            print(f"        robot_frame_z >= -0.5 的数量: {len(knee_pitch_data[knee_pitch_data['robot_frame_z'] >= -0.5])}")

# Show detailed mapping: original link -> group
print(f"      原始 link 到分组的映射:")
link_to_group = df.groupby("body2_name")["group_name"].first().sort_index()
for link, group in link_to_group.items():
    count = len(df[df["body2_name"] == link])
    print(f"        {link} -> {group} ({count} 条数据)")

# Show group statistics
group_counts = df["group_name"].value_counts().sort_index()
print(f"      分组统计:")
for group, count in group_counts.items():
    print(f"        {group}: {count} 条数据")

# -------------------------------
# Output robot_frame x, y, z distribution for each link
# -------------------------------
print(f"\n[调试] 每个 link 的 robot_frame_x, y, z 分布:")
print(f"      df 中的列: {list(df.columns)}")

# Check which columns exist
has_x = "robot_frame_x" in df.columns
has_y = "robot_frame_y" in df.columns
has_z = "robot_frame_z" in df.columns

if has_x and has_y and has_z:
    print(f"      ✓ 找到所有 robot_frame_x/y/z 列")
    for link in sorted(df["body2_name"].unique()):
        link_data = df[df["body2_name"] == link]
        if len(link_data) > 0:
            print(f"\n  {link}:")
            print(f"    数据条数: {len(link_data)}")
            
            # robot_frame_x
            x_data = link_data["robot_frame_x"].dropna()
            if len(x_data) > 0:
                print(f"    robot_frame_x: min={x_data.min():.3f}, max={x_data.max():.3f}, mean={x_data.mean():.3f}, std={x_data.std():.3f}")
            else:
                print(f"    robot_frame_x: 无数据")
            
            # robot_frame_y
            y_data = link_data["robot_frame_y"].dropna()
            if len(y_data) > 0:
                print(f"    robot_frame_y: min={y_data.min():.3f}, max={y_data.max():.3f}, mean={y_data.mean():.3f}, std={y_data.std():.3f}")
            else:
                print(f"    robot_frame_y: 无数据")
            
            # robot_frame_z
            z_data = link_data["robot_frame_z"].dropna()
            if len(z_data) > 0:
                print(f"    robot_frame_z: min={z_data.min():.3f}, max={z_data.max():.3f}, mean={z_data.mean():.3f}, std={z_data.std():.3f}")
            else:
                print(f"    robot_frame_z: 无数据")
else:
    missing_cols = []
    if not has_x:
        missing_cols.append("robot_frame_x")
    if not has_y:
        missing_cols.append("robot_frame_y")
    if not has_z:
        missing_cols.append("robot_frame_z")
    print(f"      ✗ 缺少以下列: {', '.join(missing_cols)}")
    print(f"      提示: 这些列可能不在 CSV 文件中，或者列名不同")

# -------------------------------
# Robust y-limits across ALL data
# -------------------------------
print(f"[7/7] 计算 y 轴范围...")
y_lo, y_hi = df["force_normal"].quantile([0.01, 0.99]).tolist()
if y_lo == y_hi:  # degenerate case
    y_lo -= 1.0
    y_hi += 1.0
print(f"      y 轴范围: [{y_lo:.2f}, {y_hi:.2f}]")
print(f"      force_normal 统计: min={df['force_normal'].min():.2f}, max={df['force_normal'].max():.2f}, mean={df['force_normal'].mean():.2f}")

# -------------------------------
# Prepare data for all groups in one plot
# -------------------------------
print(f"[8/8] 准备绘图数据...")

# Define all possible groups in order
def group_sort_key(group_name):
    """定义分组排序顺序"""
    order = {
        "Left_Shoulder": 1,
        "Right_Shoulder": 2,
        "Left_Elbow": 3,
        "Right_Elbow": 4,
        "Torso": 5,
        "Left_Hip": 6,
        "Right_Hip": 7,
        "Left_Knee": 8,
        "Right_Knee": 9,
        "Left_Crus": 10,
        "Right_Crus": 11,
    }
    return order.get(group_name, 99)

# All possible groups (always show all, even if empty)
all_possible_groups = [
    "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow",
    "Torso",
    "Left_Hip", "Right_Hip",
    "Left_Knee", "Right_Knee",
    "Left_Crus", "Right_Crus"
]

# Show statistics for all groups
print(f"      所有可能的分组 (共 {len(all_possible_groups)} 个):")
for i, group in enumerate(all_possible_groups, 1):
    count = len(df[df["group_name"] == group]) if group in df["group_name"].values else 0
    status = "✓" if count > 0 else "✗ (无数据)"
    print(f"        {i}. {group}: {count} 条数据 {status}")

# Prepare datasets for each group (only include groups with data)
datasets = []
group_positions = []
group_labels = []

for i, group in enumerate(all_possible_groups, 1):
    group_data = df[df["group_name"] == group]
    if len(group_data) > 0:
        datasets.append(group_data["force_normal"].to_numpy())
        group_positions.append(i)
        group_labels.append(group)
    else:
        # For empty groups, add a placeholder label but don't draw
        group_labels.append(group)

pos = np.array(group_positions)
print(f"      准备绘制 {len(datasets)} 个小提琴图 (共 {len(all_possible_groups)} 个分组，其中 {len(all_possible_groups) - len(datasets)} 个为空)")

# -------------------------------
# Create single figure with all links
# -------------------------------
print(f"      创建图形...")
# Adjust figure size based on number of groups
n_groups = len(all_possible_groups)
fig_width = max(12, n_groups * 0.8)  # At least 12, or 0.8 per group
print(f"      图形尺寸: {fig_width:.1f} x 6")
fig, ax = plt.subplots(figsize=(fig_width, 6))

# Draw violin plot
parts = ax.violinplot(
    dataset=datasets,
    positions=pos,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.8
)

# Style the violins
for pc in parts["bodies"]:
    pc.set_alpha(0.6)
    pc.set_linewidth(0.8)
if "cmedians" in parts:
    parts["cmedians"].set_linewidth(1.2)

# Set y-limits
# ax.set_ylim(y_lo, y_hi)

# Set x-axis labels - show all groups, even empty ones
ax.set_xticks(np.arange(1, len(all_possible_groups) + 1))
ax.set_xticklabels(all_possible_groups, rotation=45, ha="right", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3, axis="y")
ax.set_ylabel("force_normal", fontsize=11)
ax.set_xlabel("Link Group", fontsize=11)
ax.set_title("force_normal Distribution by Link Group (filtered: force >= 2000, no ankle/head)", fontsize=13, pad=20)

print(f"      绘制小提琴图...")
fig.tight_layout()
print(f"      保存图片...")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# -------------------------------
# Save filtered CSV to original file directory
# -------------------------------
print(f"[9/9] 保存过滤后的 CSV 文件到原文件所在目录...")
file_stem = Path(file_path).stem
file_suffix = Path(file_path).suffix
output_csv_path = file_dir / f"{file_stem}_filtered_ge2000{file_suffix}"
print(f"      原文件目录: {file_dir}")
print(f"      保存文件名: {output_csv_path.name}")
df.to_csv(output_csv_path, index=False)
print(f"      ✓ CSV 已保存到: {output_csv_path}")
print(f"      保存的数据行数: {len(df):,}")

print(f"\n✓ 完成! 小提琴图已保存到: {out_path.resolve()}")
print(f"✓ 过滤后的 CSV 已保存到: {output_csv_path}")
