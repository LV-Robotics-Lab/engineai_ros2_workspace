import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 数据列：t(m), p(密度), m(kg), h(m), v(m/s), E0(J), F1-F3(kN), Fpk(kN), FA(kN)
df = pd.read_csv("data1.csv")

# 计算缓冲前冲击力F (kN) - 使用动量公式 F = m*v/Δt
# 平均冲击时间 Δt = 0.00026554s
dt = 0.00026554  # s
F_before = (df['m'].values * df['v'].values) / dt / 1000  # 转换为kN

# 准备拟合数据
lnFpk = np.log(df['Fpk'].values)  # 缓冲后冲击力Fpk的对数
lnt = np.log(df['t'].values)      # 厚度t的对数
lnp = np.log(df['p'].values)      # 密度p的对数
lnF = np.log(F_before)            # 缓冲前冲击力F的对数

X = np.column_stack([lnt, lnp, lnF])  # 输入：厚度、密度、缓冲前冲击力
y = lnFpk                              # 输出：缓冲后冲击力

# 线性回归（对数域）
reg = LinearRegression().fit(X, y)
b0 = reg.intercept_
b1, b2, b3 = reg.coef_

C = np.exp(b0)
alpha = b1  # 厚度t的指数
beta = b2   # 密度p的指数  
gamma = b3  # 缓冲前冲击力F的指数

print(f"拟合模型: Fpk = C * t^{alpha:.3f} * p^{beta:.3f} * F^{gamma:.3f}")
print(f"参数: C={C:.3e}, alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")
print(f"其中: t为厚度(m), p为密度, F为缓冲前冲击力(kN), Fpk为缓冲后冲击力(kN)")

# 计算拟合效果
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print(f"\n=== 拟合效果评估 ===")
print(f"R² (决定系数): {r2:.4f}")
print(f"RMSE (均方根误差): {rmse:.4f}")
print(f"MAE (平均绝对误差): {mae:.4f}")

# 计算原始尺度下的拟合效果
Fpk_pred = np.exp(y_pred)
Fpk_actual = df['Fpk'].values
r2_original = r2_score(Fpk_actual, Fpk_pred)
mse_original = mean_squared_error(Fpk_actual, Fpk_pred)
rmse_original = np.sqrt(mse_original)
mae_original = mean_absolute_error(Fpk_actual, Fpk_pred)

print(f"\n=== 原始尺度拟合效果 ===")
print(f"R² (决定系数): {r2_original:.4f}")
print(f"RMSE (均方根误差): {rmse_original:.4f} kN")
print(f"MAE (平均绝对误差): {mae_original:.4f} kN")

# 计算相对误差
relative_error = np.abs(Fpk_actual - Fpk_pred) / Fpk_actual * 100
mean_relative_error = np.mean(relative_error)
print(f"平均相对误差: {mean_relative_error:.2f}%")

# 计算缓冲效果：缓冲后力相比缓冲前平均下降百分比
force_reduction = (F_before - Fpk_actual) / F_before * 100
mean_force_reduction = np.mean(force_reduction)
std_force_reduction = np.std(force_reduction)
min_force_reduction = np.min(force_reduction)
max_force_reduction = np.max(force_reduction)

print(f"\n=== 缓冲效果分析 ===")
print(f"平均力下降: {mean_force_reduction:.2f}%")
print(f"力下降标准差: {std_force_reduction:.2f}%")
print(f"最大力下降: {max_force_reduction:.2f}%")
print(f"最小力下降: {min_force_reduction:.2f}%")

# 缓冲效果评估
if mean_force_reduction > 80:
    print("缓冲效果: 优秀 (平均下降 > 80%)")
elif mean_force_reduction > 60:
    print("缓冲效果: 良好 (平均下降 > 60%)")
elif mean_force_reduction > 40:
    print("缓冲效果: 一般 (平均下降 > 40%)")
else:
    print("缓冲效果: 较差 (平均下降 < 40%)")

# 残差分析
residuals = Fpk_actual - Fpk_pred
print(f"\n=== 残差分析 ===")
print(f"残差均值: {np.mean(residuals):.4f} kN")
print(f"残差标准差: {np.std(residuals):.4f} kN")
print(f"最大正残差: {np.max(residuals):.4f} kN")
print(f"最大负残差: {np.min(residuals):.4f} kN")

# 拟合优度评估
print(f"\n=== 拟合优度评估 ===")
if r2_original > 0.9:
    print("拟合效果: 优秀 (R² > 0.9)")
elif r2_original > 0.8:
    print("拟合效果: 良好 (R² > 0.8)")
elif r2_original > 0.7:
    print("拟合效果: 一般 (R² > 0.7)")
else:
    print("拟合效果: 较差 (R² < 0.7)")

if mean_relative_error < 10:
    print("预测精度: 高 (相对误差 < 10%)")
elif mean_relative_error < 20:
    print("预测精度: 中等 (相对误差 < 20%)")
else:
    print("预测精度: 较低 (相对误差 > 20%)")

# 自助法估计参数95%置信区间
B = 2000
boot = []
for _ in range(B):
    Xb, yb = resample(X, y, replace=True, random_state=None)
    rb = LinearRegression().fit(Xb, yb)
    boot.append([np.exp(rb.intercept_), rb.coef_[0], rb.coef_[1], rb.coef_[2]])
boot = np.array(boot)
for name, col in zip(["C","alpha","beta","gamma"], [0,1,2,3]):
    lo, hi = np.percentile(boot[:,col], [2.5, 97.5])
    if name == 'C':
        print(f"{name} 95% CI: [{lo:.3e}, {hi:.3e}]")
    else:
        print(f"{name} 95% CI: [{lo:.3f}, {hi:.3f}]")

# 保存拟合参数到文件
import json

# 保存参数到JSON文件
params = {
    "C": float(C),
    "alpha": float(alpha),
    "beta": float(beta),
    "gamma": float(gamma)
}

with open("fitted_parameters.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

print(f"\n=== 参数已保存 ===")
print(f"拟合参数已保存到 fitted_parameters.json")
print(f"参数文件包含: C, alpha, beta, gamma")

# 设计反算函数
def t_required(p_val, F_val, Fpk_lim, C=C, alpha=alpha, beta=beta, gamma=gamma):
    """
    给定密度p、缓冲前冲击力F和目标缓冲后冲击力Fpk_lim，求所需厚度t
    """
    return (Fpk_lim / (C * (p_val**beta) * (F_val**gamma)))**(1.0/alpha)

# 举例计算
print(f"\n=== 设计反算示例 ===")
print(f"给定: p=0.3, F=20 kN, 目标Fpk=5 kN")
print(f"所需厚度: {t_required(0.3, 20.0, 5.0):.3f} m")