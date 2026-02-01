# CHR 与 rl_dance：观测计算与 RL 输出对比

rl_dance 为真机部署使用；CHR 为仿真/实验用。以下对比二者在**观测构建**与**RL 输出 → 关节指令**上的异同，便于对齐或排查差异。

---

## 1. 整体架构

| 项目 | rl_dance（真机） | CHR |
|------|------------------|-----|
| 观测管理 | **ObservationManager** 独立类：维护 history buffers、goal buffers，按 observation_type 从 param 读配置（time_steps、goal 维度等） | **无 ObservationManager**：所有 buffer 与观测拼接逻辑在 `RlBasicRunnerCHR` 内内联实现，配置来自 YAML 的 `observations`、`motion_states` profile |
| 控制流程 | `CalculateObservation()` → `UpdateObservation(...)` 只更新 buffer；`GetCurrentObservation()` 拼成最终 obs → `CalculateMotorCommand()` 里 `Inference(obs)`，再 `StepTrajectory()` | `CalculateObservationWalking()` / `CalculateObservationMimicFuture()` 只更新 history 与 goal_buffer；**最终 obs 在 `CalculateMotorCommand()` 内现场拼接** → `Inference(obs)` → `StepTrajectory()` |

---

## 2. 本体感知（Proprioceptive）观测

| 项目 | rl_dance | CHR |
|------|----------|-----|
| 内容 | q_actual - default_joint_q（q_diff）、qd、**上一帧 action**、w_real、projected_gravity（或 gravity_error / quat_error 若开启） | 同：q_diff_history_、qd_history_、action_history_、w_history_、gravity_history_；可选 quat_error_history_ |
| 历史步数 | 按 observation_type 从 `param_.observations` 的 `proprioceptive_time_steps` 取 | `num_include_obs_steps_`（如 5）从 YAML 读 |
| 拼接顺序 | 按类型再按时间：`[所有步的 q_diff, 所有步的 qd, 所有步的 action, 所有步的 w, 所有步的 gravity]`，再拼 quat_error（若有） | 与 rl_dance 一致 |
| 缩放与裁剪 | `BuildProprioceptiveScale()` + 统一 scale；最后 `observation_clip` | 从 `observations.observation_scale` 读各维 scale；最后 `observation_clip_` |

**结论**：本体部分结构和顺序与 rl_dance 一致，主要差异在配置来源（param vs YAML）和 obs 是在 Manager 里拼好再取，还是在 CHR 的 `CalculateMotorCommand()` 里拼。

---

## 3. Goal / 未来帧（mimic_future）

| 项目 | rl_dance | CHR |
|------|----------|-----|
| 帧数 | **10 帧**：当前帧 + 未来 9 帧，即 `traj_idx = trajectory_index_ + frame`，frame=0..9 | **9 帧**：仅未来 9 帧，即 `traj_idx = trajectory_index_ + frame + 1`，frame=0..8 |
| 每帧内容 | 每帧 2*num_joints（pos + vel） | 同：每帧 pos(24) + vel(24) |
| Goal 总维度 | 10×48 = **480**（24 关节） | 9×48 = **432** |
| Goal buffer 使用 | `goal_time_steps_[mimic_future]` 可为多步，GetCurrentObservation 里用 `goal_cols` 拼多列 | 单步：`goal_buffer_.col(0)` 一列，obs 里只拼这一列 |

**结论**：真机是「当前 + 9 个未来」共 10 帧；CHR 是「从下一帧起的 9 帧」。若 policy 是按 10 帧训练的，CHR 的 9 帧会与真机观测维度/语义不一致，需要统一帧数或训练配置。

**真机/部署侧配置**：goal = **1 当前帧 + 9 未来**（共 10 帧内容）；residual_control = **false**（见 §5）。此时语义上「当前帧 + 9 未来」与 mjlab/CHR 的 command（当前帧）+ future_frames（9 未来）一致；若 rl_dance 将 10 帧拼成 480 维 goal，内容等价，仅拼接方式不同（一坨 goal vs command+future_frames）。

### 对齐点 2（与 mjlab 训练侧一致性）✓

- **mjlab**（`tasks/tracking/mdp/commands.py`）：  
  - `command` = 当前帧 `time_steps` 的 pos+vel（`joint_pos` + `joint_vel`）。  
  - `future_frames_command` = **t+1 到 t+9** 共 9 帧，`future_steps = torch.arange(1, 10, ...)`，每帧 pos+vel，总维 9×48=432。  
- **CHR**：obs 中 `command` = 当前轨迹帧 `trajectory_index_` 的 pos+vel；`future_frames` = `trajectory_index_+1` 到 `trajectory_index_+9` 的 9 帧，432 维。  
- **结论**：**对齐点 2 正确**——CHR 的 command（当前帧）与 future_frames（未来 9 帧）的语义、索引和维度与 mjlab 训练侧一致。

---

## 4. IMU 与安装偏置

| 项目 | rl_dance | CHR |
|------|----------|-----|
| 旋转构造 | `math::RollPitchYawd(imu_install_bias_).ToRotationMatrix()`（Drake：roll-pitch-yaw 顺序） | `AngleAxisd(roll,X)*AngleAxisd(pitch,Y)*AngleAxisd(yaw,Z)` 再 `toRotationMatrix()`（固定轴 X→Y→Z） |
| 重力/角速度 | R_real = R_local * R_install^T；w_real、projected_gravity 用 R_real | 同样用 R_install、R_real 推导 w_real、projected_gravity |

**结论**：安装角到旋转矩阵的**欧拉顺序不同**（RollPitchYawd vs 固定轴 X-Y-Z）。若真机标定是按 rpyd 做的，CHR 用同一组 bias 会得到不同的 R_install，可能带来重力/角速度偏差。

---

## 5. RL 输出 → q_des（关节期望位置）

| 项目 | rl_dance | CHR |
|------|----------|-----|
| 公式 | `residual_control == true`：`q_des = GetCurrentReference() + action * action_scale`；`false`：`q_des = default_joint_q + action * action_scale` | mimic_future：**始终** `q_des = default_joint_q + action * action_scale`（无 residual 分支） |
| GetCurrentReference()（mimic_future） | 当前轨迹帧的**位置部分**：`current_traj_->row(trajectory_index_).head(num_joints)` | CHR 未使用；mimic 下没有「ref + residual」这一路 |

**结论**：真机在 mimic 类 policy 上可用「轨迹当前帧 + 残差」；CHR 的 mimic_future 始终是「default_joint_q + 动作」，与 rl_dance 的 `residual_control=false` 一致。若真机配置是 `residual_control=true`，则 CHR 与真机在 q_des 计算上不一致。

**真机/部署侧配置**：residual_control = **false** → q_des = default_joint_q + action × action_scale，与 mjlab/CHR 一致。

### 对齐点 3（与 mjlab 训练侧一致性）✓

- **mjlab**（`envs/mdp/actions/joint_actions.py` + `tasks/tracking/tracking_env_cfg.py`）：  
  - 使用 `JointPositionAction`，配置 `use_default_offset=True`。  
  - `processed_actions = raw_actions * scale + offset`，其中 `offset = asset.data.default_joint_pos[:, joint_ids]`。  
  - 即 **q_des = default_joint_pos + action × scale**（无「轨迹当前帧 + 残差」）。  
- **CHR**：`q_des_ = default_joint_q_ + mlp_net_action_.cwiseProduct(action_scale_)`。  
- **结论**：**对齐点 3 正确**——mjlab 训练时就是 default 偏移 + 动作缩放，CHR 的 q_des 计算与 mjlab 一致；若真机用 `residual_control=true`，则与 mjlab/CHR 不一致，需按真机配置单独对齐。

---

## 6. 轨迹步进与时机

| 项目 | rl_dance | CHR |
|------|----------|-----|
| 步进时机 | `CalculateMotorCommand()` 内：`Inference(obs)` 之后调用 `observation_manager_->StepTrajectory(false)` | 同：在 `CalculateMotorCommand()` 内 `Inference(obs)` 之后、计算 q_des 之前调用 `StepTrajectory()` |
| 步进逻辑 | ObservationManager 内：trajectory_index_ 自增，到末帧可停或按配置处理 | Runner 内：trajectory_index_ 自增；到末帧且未在 damping 时进入 **damping mode**（kp=0, kd=0.5），并清 mujoco_reset_received_ 等 |

**结论**：步进时机一致；CHR 多出「轨迹播完 → damping mode」的状态机，真机无此模式（或由别处实现）。

---

## 7. 扭矩限制

| 项目 | rl_dance | CHR |
|------|----------|-----|
| 调用位置 | `SendMotorCommand()` 内，在组包发送前 | `ControlCallback()` 内，在 `CalculateMotorCommand()` 之后、`SendMotorCommand()` 之前 |
| 实现思路 | 用 q_des、q_actual、qd、kp、kd 算 PD 扭矩，按 max_torque_joint 与下肢总扭矩限制做钳位，再**反解 q_des** 使下一拍 PD 输出落在限制内 | 相同：tau_des = Kp*(q_des - q_actual) - Kd*qd；逐关节与下肢总扭矩限制；反推 q_des 并做安全夹紧 |

**结论**：逻辑一致，仅调用位置不同（真机在 Send 前，CHR 在 CalculateMotorCommand 后）。

---

## 8. 其他差异（简要）

- **Walking + Mimic 状态机**：CHR 有 walking / mimic / damping、摔倒检测、test_force_fall_direction 等；rl_dance 是按 profile 切换，无内置「先走再 mimic 再 damping」的完整流程。
- **多 policy/多方向**：CHR 支持 8 方向 mimic policy + 轨迹；rl_dance 按 motion_states profile 切换，不区分 8 方向。
- **观测类型**：rl_dance 支持 mimic / control_mimic / mimic_future / mimic_tj / rl_locomotion / rl_saw_locomotion 等；CHR 主要用 walking（自建 obs）+ mimic_future。

---

## 9. 训练代码（mjlab）中的 future 拼维

训练在 `~/engineai/mjlab_wang/mjlab` 中：

- **future 帧数与索引**  
  - `tasks/tracking/mdp/commands.py`：`future_frames_command`  
    - `future_steps = torch.arange(1, 10, ...)` → 取 **t+1 到 t+9**，共 **9 帧**（不含当前帧 t）。  
  - `tasks/tracking/mdp/observations.py`：`future_frames_generated_commands_with_scale`  
    - 同样 `future_steps = torch.arange(1, 10, ...)`，索引 `time_steps + future_steps` → t+1..t+9。  
- **每帧内容**  
  - 每帧 `pos (num_joints) + vel (num_joints)`，再按 `pos_scale=1.0`、`vel_scale=0.05` 缩放。  
- **总维度**  
  - `9 * num_joints * 2` = **9×48 = 432**（24 关节）。  
- **拼接顺序**  
  - 按帧顺序：frame0(pos, vel), frame1(pos, vel), ..., frame8(pos, vel)。

结论：**训练侧是 9 个未来帧、432 维，与 CHR 一致**；与 rl_dance 的 10 帧（当前+9 未来、480 维）不同。CHR 用的 policy 若来自该 mjlab 训练，输入维度和 future 定义是对齐的；rl_dance 若用 10 帧格式，需单独训练或改 rl_dance 为 9 帧以匹配该训练与 CHR。

### 训练侧 policy 观测布局（toLeft_3.mnn 等，总维 900）

训练 / ObservationManager 打印的 policy 组观测为（**顺序与 CHR 必须一致**）：

| Index | Name                  | Shape        | 说明 |
|-------|-----------------------|-------------|------|
| 0     | joint_pos             | (120,) ← 5×(24,) | q - default_q，5 步历史 |
| 1     | joint_vel             | (120,) ← 5×(24,) | qd，5 步历史 |
| 2     | actions               | (120,) ← 5×(24,) | 上一步 action，5 步历史 |
| 3     | base_ang_vel          | (15,) ← 5×(3,)   | 机体角速度，5 步历史 |
| 4     | projected_gravity     | (15,) ← 5×(3,)   | 投影重力，5 步历史 |
| 5     | motion_anchor_ori_b   | (30,) ← 5×(6,)   | 锚点姿态前两列（6D），5 步历史；CHR 用 quat_error×5 |
| 6     | command               | (48,)            | 当前帧轨迹 pos(24)+vel(24) |
| 7     | future_frames         | (432,)           | 未来 9 帧 t+1..t+9，每帧 pos+vel，9×48 |

**总维**：120+120+120+15+15+30+48+432 = **900**。

CHR 在 `CalculateMotorCommand()` 中 mimic_future 分支的拼接顺序与此一致：`proprio_total(390) = q_diff×5 + qd×5 + action×5 + w×5 + gravity×5`，再 `quat_error×5(30)`、`command(48)`、`future_frames(432)`，总 900。部署时需保证 YAML 中 `use_quat_error: true` 与训练一致，否则 420→390、总维 870，会与 toLeft_3.mnn 等 900 维输入不匹配。

---

## 10. 建议对齐项（若希望 CHR 与真机一致）

- **与 mjlab 训练侧已对齐**（CHR 与 toLeft_3.mnn 等 900 维 policy 一致）：  
  - **对齐点 1**：mimic_future 未来帧数 = 9 帧（t+1..t+9），432 维。  
  - **对齐点 2**：Goal/command = 当前帧，future_frames = 未来 9 帧，索引与维度与 mjlab 一致。  
  - **对齐点 3**：q_des = default_joint_q + action × action_scale，与 mjlab 的 JointPositionAction（use_default_offset=True）一致。

- **真机配置**（residual_control=false，goal=1 当前帧+9 未来）：q_des 与 mjlab/CHR 一致；goal 内容为「当前+9 未来」，与 mjlab/CHR 的 command+future_frames 语义一致（若 rl_dance 将 10 帧拼成 480 维，内容等价）。  
- **若真机与上述不同**，可考虑：  
  1. **mimic_future 未来帧数**：若真机 goal 不是「1 当前+9 未来」，需与训练侧维度一致。  
  2. **residual_control**：若真机用 `residual_control=true`，CHR 可增加「q_des = GetCurrentReference() + action*scale」分支以对齐真机。  
  3. **IMU 安装角**：统一 R_install 的欧拉顺序（RollPitchYawd vs 固定轴 X-Y-Z），或在不同端用与各自标定一致的定义并在文档标明。

以上为代码层面的异同总结；具体是否改 CHR 以完全对齐真机，取决于训练/标定是在哪一侧完成的以及是否需要仿真-真机可互换 policy。
