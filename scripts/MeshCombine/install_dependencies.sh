#!/bin/bash

echo "正在使用conda安装URDF mesh合并工具所需的依赖..."

# 检查yml文件是否存在
if [ ! -f "scripts/MeshCombine/urdf_environment.yml" ]; then
    echo "错误: scripts/MeshCombine/urdf_environment.yml 文件不存在！"
    echo "请确保 scripts/MeshCombine/urdf_environment.yml 文件在当前目录中。"
    exit 1
fi

# 使用conda environment.yml创建环境
echo "使用conda environment.yml创建环境..."
conda env create -f scripts/MeshCombine/urdf_environment.yml

# 激活环境
echo "激活环境..."
conda activate urdf_mesh_tools

echo "依赖安装完成！"
echo "现在可以运行: python urdf_mesh_combiner.py"
