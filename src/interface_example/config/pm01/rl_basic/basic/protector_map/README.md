# 护具 map（Protector Map）

本目录存放**默认站立系**下的护具/防护查表数据，供 tracking reward 中力衰减使用。

- **坐标系**：与 `contact_pos_in_default_standing_frame()` 输出一致（仿真**世界系**，机器人处于 default qpos 时的世界系）。
- **用途**：用接触点默认系坐标 `pos_default [B, N, 3]` 查表得到厚度/衰减，再乘到接触力上。

### 原点 (0, 0, 0)

- **(0, 0, 0)** = **世界原点**（仿真世界系原点）。平地场景下 **z = 0 即地面高度**。
- 因此 map 的 **z ∈ [0, 1.4]** 表示**离地高度** 0～1.4 m；**y ∈ [-0.4, 0.4]** 为左右；**x** 正为前、负为后（与 default 站立时机器人前方一致）。
- PM 默认站立时 root 约在 (0, 0, 0.76)，即脚踝离地约 0.76 m；map 覆盖从地面到约 1.4 m 高。

## YZ 平面 TSV 网格

| 文件 | 适用区域 | 说明 |
|------|----------|------|
| `yz_map_front.tsv` | x ≥ 0（前侧） | YZ 平面网格 |
| `yz_map_back.tsv` | x < 0（后侧） | YZ 平面网格 |

- **Y**：-0.4 m ~ +0.4 m，步长 0.05 m（17 列）
- **Z**：0 m ~ 1.4 m，步长 0.05 m（29 行）
- **像素**：0.05 m × 0.05 m 方格
- **像素值**：直接为护具厚度（mm），如 0、6、12、18、24

**查表逻辑**：表头/行首的 Y、Z 是**格心**坐标；代码用**最近邻**查表。半格距 = 方格间距 / 2，由相邻格心坐标差**计算得出**（非写死）。每个格子覆盖以该格心为中心的 (间距×间距) 区域。  
例如步长 0.05 m 时，**Y=0 列、Z=0.1 行** 的值为 6 → 表示 **-0.025 ≤ y < 0.025、0.075 ≤ z < 0.125**（单位 m）范围内的厚度均为 6 mm。

TSV 格式：注释行（以 `#` 开头），可含 `# density 0.4` 指定材料密度（use_protector_map=true 时 CHR 公式使用）；表头行 `z\y` + Y 坐标；以下每行为 Z 值 + 该行各格像素值（制表符分隔）。

## 力衰减参数

`fitted_parameters.json`：与 `scripts/ThicknessCalculate/force_calculator.py` 同公式的拟合参数（C, alpha, beta, gamma）。  
reward 中按 `F_after = C * (t_mm^alpha) * (p^beta) * (F_before^gamma)` 计算衰减后力（t 为查表厚度 mm，F 单位 kN），再对衰减后力做加权惩罚得到 reduce_contact_force reward。

## 路径引用

```python
from mjlab.tasks.tracking.config.pm1.env_cfgs import PROTECTOR_MAP_DIR

path_front = PROTECTOR_MAP_DIR / "yz_map_front.tsv"
path_back = PROTECTOR_MAP_DIR / "yz_map_back.tsv"
```
