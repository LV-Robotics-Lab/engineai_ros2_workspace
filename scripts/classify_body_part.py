#!/usr/bin/env python3
"""
身体部分分类模块
根据body1和body2名称，使用CSV表格规则进行分类
"""

import pandas as pd
from pathlib import Path

# 获取脚本目录
script_dir = Path(__file__).parent

# 加载分类规则CSV
CLASSIFY_RULES_CSV = script_dir / 'ClassifyRules.csv'
classify_rules_df = None


def load_classify_rules():
    """加载分类规则CSV文件"""
    global classify_rules_df
    if classify_rules_df is None:
        if CLASSIFY_RULES_CSV.exists():
            classify_rules_df = pd.read_csv(CLASSIFY_RULES_CSV, index_col=0)
            print(f"已加载分类规则: {CLASSIFY_RULES_CSV}")
        else:
            print(f"警告: 分类规则文件不存在: {CLASSIFY_RULES_CSV}")
            classify_rules_df = pd.DataFrame()
    return classify_rules_df


def extract_link_type(link_name):
    """
    从LINK_XXX_L格式的链接名称中提取类型
    
    参数:
        link_name: 链接名称，如 "LINK_SHOULDER_PITCH_L" 或 "LINK_TORSO"
    
    返回:
        link_type: 提取的类型，如 "shoulder", "torso", "elbow" 等
    """
    if pd.isna(link_name) or link_name == '':
        return 'world'
    
    link_name = str(link_name).upper()
    
    # 移除LINK_前缀
    if link_name.startswith('LINK_'):
        link_name = link_name[5:]
    
    # 移除后缀 _L 或 _R
    if link_name.endswith('_L') or link_name.endswith('_R'):
        link_name = link_name[:-2]
    
    # 转换为小写
    link_name = link_name.lower()
    
    # 匹配类型
    if 'torso' in link_name:
        return 'torso'
    elif 'shoulder' in link_name:
        return 'shoulder'
    elif 'elbow' in link_name:
        return 'elbow'
    elif 'hip' in link_name:
        return 'hip'
    elif 'knee' in link_name:
        return 'knee'
    elif 'base' in link_name:
        return 'hip'  # base通常归类为hip
    elif 'ankle' in link_name or 'foot' in link_name:
        return 'knee'  # 踝关节和脚通常归类为knee
    elif 'head' in link_name:
        return 'torso'  # 头部通常归类为torso
    else:
        return 'world'  # 默认返回world


def classify_body_part_by_rules(body1_name, body2_name):
    """
    根据body1和body2名称，查询CSV表格，返回分类名称
    
    参数:
        body1_name: body1的链接名称，如 "LINK_SHOULDER_PITCH_L"
        body2_name: body2的链接名称，如 "LINK_ELBOW_YAW_R"
    
    返回:
        classification: 分类名称，如 "shoulder", "elbow", "torso" 等
    """
    rules_df = load_classify_rules()
    
    if rules_df.empty:
        # 如果规则表为空，返回默认值
        return 'Unknown'
    
    # 提取body1和body2的类型
    body1_type = extract_link_type(body1_name)
    body2_type = extract_link_type(body2_name)
    
    # 查询表格
    # body2_type是行索引，body1_type是列名
    try:
        if body2_type in rules_df.index and body1_type in rules_df.columns:
            classification = rules_df.loc[body2_type, body1_type]
            # 如果值是'/'，表示无碰撞或无效
            if pd.isna(classification) or str(classification).strip() == '/':
                return 'Unknown'
            return str(classification).strip().lower()
        else:
            # 如果找不到匹配，返回默认值
            return 'Unknown'
    except Exception as e:
        print(f"查询分类规则时出错: {e}, body1_type={body1_type}, body2_type={body2_type}")
        return 'Unknown'


def get_body_part(row, use_coordinates=True):
    """
    根据body1_name和body2_name，使用CSV表格分类规则识别身体部分
    
    参数:
        row: DataFrame的一行，包含body1_name, body2_name, robot_frame_y, robot_frame_z
        use_coordinates: 是否使用坐标信息判断左右侧（默认True）
    
    返回:
        body_part: 身体部分名称，如 "Left_Shoulder", "Right_Elbow", "Torso" 等
    """
    body1_name = row.get('body1_name', '')
    body2_name = row.get('body2_name', '')
    robot_y = row.get('robot_frame_y', 0) if pd.notna(row.get('robot_frame_y')) else 0
    robot_z = row.get('robot_frame_z', 0) if pd.notna(row.get('robot_frame_z')) else 0
    
    # 使用CSV表格分类规则
    classification = classify_body_part_by_rules(body1_name, body2_name)
    
    if classification == 'Unknown':
        return "Unknown"
    
    # 确定左右侧（基于body2_name）
    link_name = str(body2_name).lower()
    is_left = False
    if any(x in link_name for x in ["left", "l_", "_l", "l_shoulder", "l_elbow", "l_hip", "l_knee"]):
        is_left = True
    elif any(x in link_name for x in ["right", "r_", "_r", "r_shoulder", "r_elbow", "r_hip", "r_knee"]):
        is_left = False
    elif "base" in link_name and use_coordinates:
        # For base: y+ is left, y- is right
        is_left = robot_y > 0 if pd.notna(robot_y) else False
    elif use_coordinates:
        # Default: y+ is left, y- is right
        if pd.notna(robot_y):
            is_left = robot_y > 0
    
    side = "Left" if is_left else "Right"
    
    # 根据分类结果返回带左右侧的名称（torso不分左右）
    if classification == 'torso':
        return "Torso"
    else:
        # 首字母大写
        classification_capitalized = classification.capitalize()
        return f"{side}_{classification_capitalized}"

