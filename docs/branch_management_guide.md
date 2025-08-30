# 分支管理指南

## 当前分支状态

### 你的功能分支
- **分支名**: `wang22-contact-csv-feature`
- **基于**: `community` 分支
- **最新提交**: `bccf79f` - 实现接触力CSV每帧保存功能

## 分支管理策略

### 1. 为什么使用功能分支？

✅ **优势**:
- 避免与主分支冲突
- 可以独立开发和测试
- 便于代码审查和合并
- 保持主分支的稳定性

### 2. 分支命名规范

```
{用户名}-{功能描述}-{类型}
```

例如：
- `wang22-contact-csv-feature` - 接触力CSV功能
- `wang22-bugfix-timestamp` - 时间戳修复
- `wang22-enhancement-analysis` - 分析功能增强

### 3. 日常操作

#### 查看当前分支
```bash
git branch
git status
```

#### 切换分支
```bash
# 切换到主分支
git checkout community

# 切换到你的功能分支
git checkout wang22-contact-csv-feature
```

#### 同步主分支更新
```bash
# 切换到主分支
git checkout community

# 拉取最新更新
git pull origin community

# 切换回功能分支
git checkout wang22-contact-csv-feature

# 合并主分支更新
git merge community
```

#### 提交新修改
```bash
# 添加修改
git add .

# 提交修改
git commit -m "feat: 新功能描述"

# 推送到远程（如果有权限）
git push origin wang22-contact-csv-feature
```

## 当前功能分支内容

### 主要功能
1. **接触力CSV保存**
   - 每帧保存（10kHz）
   - 可配置保存频率
   - 时间戳修复

2. **分析工具**
   - 接触力分析脚本
   - 快速分析脚本
   - 可视化工具

3. **文档**
   - 使用指南
   - 技术文档
   - 故障排除

### 文件结构
```
docs/
├── contact_csv_saving_guide.md      # CSV保存指南
├── contact_force_analysis_guide.md  # 分析指南
└── branch_management_guide.md       # 本文件

scripts/
├── analyze_contact_forces.py        # 分析脚本
├── quick_force_analysis.sh          # 快速分析
└── run_contact_csv_test.sh          # 测试脚本

src/simulation/mujoco/
├── src/ros_interface.cc             # CSV保存逻辑
└── include/ros_interface.h          # 接口定义
```

## 下一步计划

### 短期目标
1. **测试和优化**
   - 验证每帧保存性能
   - 优化文件I/O效率
   - 添加更多分析功能

2. **功能扩展**
   - 添加magnitude计算
   - 支持更多数据格式
   - 实时可视化

### 长期目标
1. **代码审查**
   - 准备Pull Request
   - 代码质量检查
   - 性能测试

2. **文档完善**
   - API文档
   - 示例代码
   - 最佳实践

## 注意事项

### 1. 数据文件管理
- CSV数据文件已添加到`.gitignore`
- 只提交代码和文档，不提交数据
- 使用`logs/.gitkeep`保持目录结构

### 2. 编译和测试
```bash
# 编译
./scripts/build_nodes.sh sim

# 测试
./scripts/run_contact_csv_test.sh
```

### 3. 备份策略
- 定期推送到远程分支（如果有权限）
- 本地备份重要数据
- 记录关键配置参数

## 常用命令速查

```bash
# 查看分支
git branch -a

# 查看提交历史
git log --oneline -10

# 查看文件状态
git status

# 查看修改内容
git diff

# 撤销修改
git checkout -- <file>

# 创建新分支
git checkout -b wang22-new-feature

# 删除分支
git branch -d wang22-old-feature
``` 