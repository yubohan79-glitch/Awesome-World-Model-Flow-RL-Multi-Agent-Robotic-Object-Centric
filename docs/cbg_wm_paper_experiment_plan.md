# CBG-WM 论文级强化与正式实验执行方案

> 工作标题：**CBG-WM: Action-Conditioned Constraint-Graph Dynamics for Risk-Aware Multi-Robot Planning under Partial Observability**  
> 核心方法名：**Action-Conditioned Constraint Graph Dynamics (ACCGD)**  
> 方案冻结日期：2026-07-28  
> 当前状态：方案与代码缺口审计已完成；正式多随机种子训练和论文结果尚未完成。

## 1. 结论先行

现有 CBG-WM 的 belief token、手工类型图、ensemble、CVaR MPC 和 action shield 可以构成完整系统，但这些组件本身都有明确前作，不能作为论文的主要创新。论文必须收缩到一个可验证命题：

> 机器人动作不仅改变对象状态，还会生成、删除并延续约束关系；约束图拓扑的变化进一步改变路径可达性、视线、动作合法性和风险。CBG-WM 显式学习这种动作条件下的约束边生命周期，并在部分可观测条件下用其进行短视界风险规划。

正式论文不能把当前几何方向 smoke test 表述为因果识别。本文只主张：

1. 从传感器 belief 预测动作条件下的约束边转移和持续时间；
2. 使用同一初态、同一外生随机数的成对模拟器干预监督“干预效应一致性”；
3. 使用显式区分的奖励、物理代价、规则代价和认知不确定性进行 ego-only 规划；
4. 在满足交换性假设的冻结校准集上给出有限样本的边际风险校准，不把该保证外推到任意 OOD 或实机分布。

如果动态图模块、成对干预实验、公开任务族和强基线没有完成，目标应定位为 ICRA/IROS/RA-L 系统论文；完成后才有理由讨论 CoRL/AAAI，仍不应直接宣称达到 ICLR/NeurIPS 的通用性或理论强度。

## 2. 现有实现审计：正式实验前的硬阻塞项

| 项目 | 当前代码行为 | 为什么不能直接跑论文实验 | 必须修改 |
|---|---|---|---|
| 图结构 | `build_typed_edges` 每一步从预测坐标执行手写几何规则 | 没有学习边的生成、删除或持续时间，因而尚不存在 ACCGD | 增加 edge transition、onset/deletion 和 survival head；手工规则只生成训练标签和类型合法性 mask |
| 多步随机 rollout | 网络预测 `state_logvar`，但 rollout 用 `delta_mean` 推进 | aleatoric 不确定性没有进入轨迹分布 | 每个 ensemble member 至少采样 16 条 aleatoric 粒子，5 个成员形成 80 条轨迹粒子 |
| CVaR | 目前只在 5 个 ensemble 返回上取最差 25%，实际是最差 2 个成员 | 尾部样本太少，不能支撑风险结论 | 规划使用 80 条粒子；正式报告用 `CVaR_0.90`，即代价最高 10% 的均值 |
| 对抗规划 | 一个 joint candidate 同时为黄、蓝双方选第一步动作，并按双方均值选候选 | 测试时不能控制对手；存在集中式信息与动作泄漏 | 改为 ego-only MPC；对手动作只作为冻结策略/学习模型产生的假设，不由 ego 优化器决定 |
| OOD 策略评估 | `evaluate_cbg_world_model.py` 的轨迹由 deterministic actor 生成，没有走 checkpoint 中的 MPC | 得到的 win rate 不能证明规划器有效 | 统一策略和模型评估入口，显式记录 planner mode |
| 数据结构 | replay 只保存独立单步 transition | 无法训练 1/5/10 步、多步边持续时间和配对分支 | 增加 episode-aware sequence replay、`pair_id`、branch、exogenous seed 和边标签 |
| OOD 切分 | 当前只在评估时平移箱子/旋转目标，没有证明与训练支持集不重叠 | “held-out” 名称缺少可审计依据 | 生成并冻结 train/calibration/test 参数清单，运行时校验集合不相交 |
| low friction | 快速环境实际改变 drive/push/accuracy scale，不是真实摩擦系数 | 不能在论文中误写成物理摩擦 OOD | 快速环境命名为 low-traction dynamics；IsaacLab 正式评估直接修改 material friction |
| 统计 | 每次输出单个 JSON，无批量聚合、层级 CI 或多重比较修正 | 不能完成显著性分析 | 新增逐 episode 原始表、层级 bootstrap、成对置换、Holm 修正和结果表生成器 |
| legacy 消融 | YAML 指向一个已有 checkpoint | 不满足 3 个独立训练随机种子 | 在相同环境步数和数据协议下重新训练 3 次，不把旧 checkpoint 当三个重复实验 |

这些阻塞项未通过测试前，不启动 18 个正式训练；否则会花费大量 GPU 时间得到无法支持核心主张的结果。

## 3. 方法定义

### 3.1 状态与动态图

令 `B_t = {b_i,t}` 为对象 belief tokens，包含对象类型、位置、速度、可见性、时间戳、age、协方差、遮挡和存在概率。令：

```text
e^r_ij,t in {0, 1}
```

表示对象 `i -> j` 在关系类型 `r` 下是否存在边。主要动态约束边为：

- `CONTACTS(robot, box)`
- `BLOCKS_ROUTE(box, target_or_waypoint)`
- `PROTECTS_BASE(armor, base)`
- `LINE_OF_SIGHT(robot, target)`
- `THREATENS(opponent, robot_or_target)`
- `OBSERVES(robot, object)`

世界模型按下式分解：

```text
p(G_t+1, B_t+1, R_t, C_t, D_t | B_t, G_t, A_t)
  = p(G_t+1 | B_t, G_t, A_t)
    p(B_t+1 | B_t, G_t+1, A_t)
    p(R_t, C_t, D_t | B_t, G_t+1, A_t)
```

第一项是论文核心，显式预测每类边的：

- `presence_logit`：下一时刻是否存在；
- `event_logit`：`stay / add / delete`；
- `hazard`：已存在边在下一步结束的离散生存风险；
- `duration_bucket`：新边预计持续 `1 / 2-3 / 4-7 / 8+` 步。

节点类型 mask 禁止不合法的边，例如 armor 不能 `OBSERVES` robot。mask 是规则先验，不替代学习到的边概率。rollout 时使用预测的软边做 message passing，不再根据预测坐标重新调用手工几何函数生成运行时邻接矩阵。

![ACCGD pipeline](./figures/cbg_wm_accgd_pipeline.png)

Mermaid 源文件：[`figures/cbg_wm_accgd_pipeline.mmd`](./figures/cbg_wm_accgd_pipeline.mmd)；矢量版本：[`figures/cbg_wm_accgd_pipeline.svg`](./figures/cbg_wm_accgd_pipeline.svg)。

### 3.2 成对干预训练

模拟器需要提供可复现的 `clone_state/restore_state`。对同一状态 `s_t` 和同一外生随机数 `xi` 生成两条短轨迹：

```text
tau_a  = F(s_t, a,  xi)
tau_a' = F(s_t, a', xi)
```

干预只改变指定动作或对象条件，其他随机性相同。两类主干预为：

1. `push / no-push`：相同初态下是否推动指定箱子离开路线或射线；
2. `armor-hit / bypass`：相同初态下是否先移除护甲，再攻击 base。

建议 replay 构成为：70% 常规 self-play 序列、15% push 配对、15% armor 配对；配对 horizon 为 10。训练损失为：

```text
L = L_state + 0.5 L_reward + 0.5 L_risk
    + lambda_edge L_edge + lambda_surv L_survival
    + lambda_cf L_interventional_delta
```

其中 `L_interventional_delta` 比较两分支的预测差分与真实差分：

```text
(G_hat_a' - G_hat_a) vs (G_a' - G_a)
(R_hat_a' - R_hat_a) vs (R_a' - R_a)
(C_hat_a' - C_hat_a) vs (C_a' - C_a)
```

训练阶段允许使用模拟器真值生成 edge/risk teacher label，推理阶段只允许使用 belief tokens。这是 asymmetric privileged training，不是把训练环境真值泄漏给部署策略。

### 3.3 Ego-only 风险规划

对每个受控机器人独立执行规划：Flow actor 生成 ego 候选序列；冻结 opponent policy league 或 opponent dynamics model 生成对手动作假设。优化器只能选择 ego 动作，不能选择对手动作。

对代价通道 `k` 使用：

```text
score = E[return]
        - sum_k lambda_k * max(0, CVaR_0.90(cost_k) - budget_k)
        - lambda_u * epistemic_disagreement
        - lambda_cal * calibrated_upper_margin
```

四类代价保持分开：collision、penetration、illegal fire、LOS/range violation。`lambda_k` 采用验证集固定或 Lagrangian 更新，不能在测试集调参。action shield 保留为最后执行层，但主结果和 shadow audit 必须区分 planner 拒绝与 shield 拒绝，防止“零碰撞只是 shield 全部拦截”的错误结论。

### 3.4 校准与风险界的声明边界

1. 在冻结 nominal calibration episodes 上对每个风险 head 做 temperature scaling；类别极不平衡时使用 isotonic 仅作为敏感性分析。
2. 以 episode 为交换单元应用 split conformal / conformal risk control，输出给定风险阈值下的 marginal upper bound。
3. 序列窗口不能被当作独立校准样本；同一 episode 的窗口先聚合。
4. nominal 之外只报告经验校准和覆盖率下降。除非重新收集目标域 calibration set，否则不宣称 OOD/实机保证。

## 4. 六个核心训练版本 × 三个随机种子

正式种子固定为 `260707, 260708, 260709`。所有版本使用相同的 200,000 环境步、相同 train parameter manifest、相同 action shield、相同 checkpoint 选择规则和相同验证预算。

| ID | 训练版本 | 保留内容 | 移除/替换内容 | 回答的问题 |
|---|---|---|---|---|
| T0 | `legacy_sac_flow` | 原始 SAC Flow/self-play、shield | 无 belief-graph planning | 新系统是否优于当前已验证主线？必须重训 3 次 |
| T1 | `no_belief_uncertainty` | 动态图、ensemble、风险 MPC | covariance/age/occlusion uncertainty 字段置为常量；传感器 delay/dropout 仍保持相同 | 收益是否来自显式 belief uncertainty，而非更容易的传感器条件？ |
| T2 | `no_interaction_graph` | belief、ensemble、ego MPC | 所有 interaction message 和 edge state | 对象交互结构是否必要？ |
| T3 | `static_rule_graph` | 当前手工 typed graph、ensemble、ego MPC | 不学习 edge transition/survival；每步按预测几何重算 | 新的动态图学习是否优于当前手工静态图？这是核心机制基线 |
| T4 | `dynamic_graph_no_pairs` | 学习 edge transition/survival、ensemble、ego MPC | `lambda_cf = 0`，不使用 paired intervention loss | 成对干预监督是否真正贡献？ |
| T5 | `full_accgd_cbg_wm` | belief、动态图、paired loss、ensemble、calibrated risk MPC、shield | 无 | 完整方法 |

总计：`6 × 3 = 18` 个核心训练。每个 run 都从独立随机初始化开始，不允许复制 checkpoint 或只替换 seed 字段。

旧 YAML 中的 `single_world_model` 和 `no_mpc` 不再占用六个核心训练槽位，因为它们不能直接验证新的动态图主张。它们作为 T5 checkpoint 的推理时 planner diagnostics：

- P0 `actor_only`：关闭 MPC；
- P1 `expected_return_mpc`：关闭 risk、CVaR 和 uncertainty penalty；
- P2 `mean_risk_mpc`：期望代价，不取尾部；
- P3 `single_member_mpc`：只使用一个 ensemble member；
- P4 `uncalibrated_cvar_mpc`：CVaR 但无 calibration margin；
- P5 `full_calibrated_cvar_mpc`：完整规划。

这些推理消融共享完全相同的 T5 参数，因而比重新训练后比较更直接地隔离 planner 决策规则。若论文需要讨论 ensemble 对模型学习本身的影响，再增加一个预注册的 T6 单模型训练，不混入核心 18 runs。

## 5. 外部强基线与公开任务

### 5.1 自研竞技环境强基线

除六个内部版本外，至少训练以下官方方法各 3 个种子，环境交互预算同为 200,000 步：

| 基线 | 用途 | 公平性要求 |
|---|---|---|
| TD-MPC2 | 强 latent MPC 基线 | 使用官方实现的 episodic 支持；相同结构化观测和动作接口；不向其提供部署时不可得真值 |
| DreamerV3 | 通用 world-model RL 基线 | 使用 Nature 2025 对应实现；相同环境步数，单独报告 gradient steps 和 wall time |
| SafeDreamer | reward/cost 分离的安全 world-model 基线 | 四类规则事件映射为 cost；相同 shield 主结果，并增加 shield-off simulator audit |

这 9 个 run 不计入 18 个内部消融，因此自研环境论文级训练总量至少为 27 runs。超参数只可用 calibration/validation seeds 调整，测试集不得参与选择。

### 5.2 公开任务族

首选 **CausalWorld** 的 pushing、pick-and-place、stacking 三类任务，因为官方环境支持 `do_interventions`、counterfactual environment、质量/摩擦/初态干预和结构化对象状态。关系映射为：

- `CONTACTS(finger_or_robot, block)`
- `SUPPORTS(block, block)`
- `INSIDE_GOAL(block, goal)`
- `BLOCKS_REACH(block, goal)`

训练干预轴：初始位姿和目标位姿；测试干预轴：未见质量、摩擦、对象尺寸和组合。比较 T3、T4、T5、TD-MPC2、DreamerV3，每项 3 seeds，报告 task success、edge event F1 和 intervention effect error。

CausalWorld 依赖较老，必须放在独立 conda/container 中做 2 小时兼容性 gate。若无法在不修改环境语义的条件下复现，预注册 fallback 为 DINO-WM 官方 `PushT + Wall` 数据和规划接口；fallback 的原因和版本必须写入 manifest，不能看完结果后再换 benchmark。

## 6. 数据切分与六类场景

所有参数组合先写入 `scenario_splits.yaml`，再生成数据。训练、校准、预测测试、在线对战和干预测试使用互不重叠的 seed namespace：

| 用途 | seed 范围 |
|---|---|
| 训练 reset | 训练 seed 自身及 `+100000` 的内部序列 |
| nominal calibration | `410000-410127` |
| frozen prediction test | 每场景 `420000 + 1000 × scenario_id` 起的 128 episodes |
| online match | 每场景 `430000 + 1000 × scenario_id` 起的 32 world seeds |
| paired intervention | 每场景 `440000 + 1000 × scenario_id` 起的 128 pairs/机制 |

六类场景固定如下：

| ID | 场景 | Train support | Test/OOD 设置 | 要验证的泛化 |
|---|---|---|---|---|
| S0 | `nominal` | 训练参数范围 | 参数范围内、未见 episode seeds | IID 基线与校准 |
| S1 | `held_out_boxes` | 默认箱位附近平移不超过 0.08 normalized units | 两箱分别平移 `(+0.17,-0.13)`、`(-0.16,+0.14)`，并验证落在 train support 外 | 路径/射线拓扑改变 |
| S2 | `held_out_target_yaw` | yaw jitter 在 `[-10°, +10°]` | 正常目标额外旋转 `±25.7°`，方向按 world seed 配平 | 射线与射击姿态改变 |
| S3 | `delayed_occlusion` | delay `0-1` step、dropout `0-0.05` | delay `4` steps、dropout `0.25`、covariance growth `0.14` | belief 过期和遮挡 |
| S4 | `low_traction` | IsaacLab friction `0.70-1.10` | IsaacLab floor/wheel friction `0.35-0.45`；快速环境只作为 drive/push scale surrogate | 动力学变化导致推箱与制动误差 |
| S5 | `aggressive_opponent` | self-play snapshot league | 未参与训练的 aggressive scripted policy 与 aggression-shaped SAC snapshot 各占对手池一半 | 对手策略变化 |

每次生成后执行 split audit：检查参数区间、组合哈希和 episode seed 均无交集。S1/S2 不能只写“held out”，必须把实际参数随原始结果发布。

## 7. 统一评估矩阵

### 7.1 108 个正式单元

```text
18 trained checkpoints × 6 scenarios = 108 checkpoint-scenario cells
```

每个 cell 完成三部分：

1. 对同一个冻结 prediction dataset 做 1/5/10 步 open-loop rollout；
2. 对同一个冻结 intervention dataset 做 push/no-push 和 armor-hit/bypass；
3. 对冻结 opponent pool 做在线对战。

预测数据必须由固定 behavior mixture 生成，建议 40% legacy、30% scripted/intervention、20% T3 snapshot、10% random legal exploration。不能让每个模型只在自己的访问分布上测预测误差。

### 7.2 正式对战数量

每个 cell 使用：

```text
32 world seeds × 4 frozen opponents × 2 seat assignments = 256 matches
```

四个对手为：legacy seed-1、legacy seed-2、aggressive scripted、aggression-shaped SAC。每个 world seed 保留 side-swapped 两局；环境随机数、布局和 opponent identity 构成配对 block。

因此核心 108 cells 总计 `27,648` 场在线比赛。主结果报告 win/draw/loss 和 `win score = win + 0.5 × draw`。禁止用“同一个 joint policy 同时控制双方的自博弈黄方胜率”代替算法胜率。

### 7.3 Shield 协议

- 主表：所有方法 `shield=on`，符合实际部署安全契约；
- 诊断表：模拟器中 `shield=shadow`，记录“若不拦截将违规”的 proposed actions；
- 附录：小规模 `shield=off`，只在模拟器执行，用来判断 planner 自身风险，不用于实机；
- 同时报告 shield intervention rate，证明风险降低不是单纯因为动作全部被拦截。

## 8. 指标定义

### 8.1 1/5/10 步对象预测

每个 episode 先计算，再跨 episode 聚合，避免把重叠窗口伪装成独立样本：

- position RMSE，换算为米；
- velocity RMSE；
- physical-state Gaussian NLL；
- 50%/90% predictive interval coverage；
- epistemic/aleatoric variance 与 squared error 的 Spearman correlation；
- 按 robot/box/target/armor 类型分层结果。

T0 没有可比的 belief graph world model，其预测指标标记 `N/A`，不能填 0。T1-T5 使用完全相同的 token、action 和 target contract。

### 8.2 动态边指标

- edge presence macro/micro F1；
- `add/delete/stay` macro-F1；
- edge onset time MAE 和 deletion time MAE；
- duration bucket macro-F1；
- survival integrated Brier score；
- 每类边的 Brier、ECE 和 AUPRC；
- 只对发生边变化的 challenge subset 单独报告，防止大量 `stay absent` 掩盖失败。

### 8.3 风险校准

四个通道分别报告：

- Brier score；
- binary NLL；
- ECE，15 个 equal-mass bins；
- AUROC 和 AUPRC；
- nominal conformal upper-bound coverage；
- OOD coverage degradation。

某场景某通道少于 20 个 positive episodes 时，AUROC/AUPRC 标为低支持或 `N/A`，不能通过合并重叠窗口人为制造样本量。

### 8.4 CVaR

定义 episode cumulative cost：

```text
C_k = sum_t gamma^t c_k,t
m = ceil((1 - beta) * N)
CVaR_beta_hat(C_k) = mean of the m largest episode costs among N samples
```

该固定尾部样本数定义在大量零代价或分位点并列时仍然有效。规划时由 80 条 stochastic particles 估计，`beta=0.90` 时取最高代价 8 条；评估时每个 cell 有 256 场，取最高代价 26 场。报告：

- 每风险通道 realized `CVaR_0.90`；
- 总规则代价 `CVaR_0.90`；
- predicted-vs-realized CVaR absolute error；
- `beta = 0.80/0.95` 敏感性分析；
- reward-risk Pareto curve。

### 8.5 反事实/干预指标

对每种机制、每场景 128 对：

- edge-effect sign accuracy；
- edge-effect magnitude MAE；
- return-effect sign accuracy；
- rule-risk-effect sign accuracy；
- counterfactual action selection accuracy；
- intervention regret：选择错误分支相对真实最佳分支损失的 return；
- nuisance sensitivity：改变颜色/视角但保持物理参数时，预测效应的最大-最小差。

最后一项借鉴 CRONOS 的 matched intervention 思路，将“应保持不变的 nuisance intervention”和“应产生结构变化的 physical intervention”分开。这里仍称 controlled interventional consistency，不称为从观察数据识别了真实因果图。

### 8.6 任务结果和效率

- win/draw/loss、win score；
- score difference、normal targets cleared、base success；
- collision/penetration/illegal-fire/LOS violation per episode；
- task completion time；
- shield intervention rate；
- planner latency mean/P95/P99、environment steps/s、peak VRAM；
- 参数量和总 wall-clock training time。

必须同时画 win score 与 rule risk/完成时间的 Pareto 图，检验“安全提升是否只是极端保守”。

## 9. 统计方案与显著性边界

### 9.1 预注册主假设

- H1（机制主假设）：T5 相对 T3 在 edge change macro-F1 上提高；
- H2（决策主假设）：T5 相对 T3 在五个 OOD 场景聚合 win score 上提高；
- H3（风险主假设）：T5 相对 T3 降低 realized rule-cost `CVaR_0.90`，同时 nominal win score 的下降不超过 2 percentage points。

三项是 co-primary，使用 Holm-Bonferroni 控制 family-wise error rate `0.05`。T5 对 T0/T1/T2/T4 和外部基线为 secondary comparisons。

确认性检验中，H1/H2 使用预注册的单侧 superiority 方向；H3 使用 intersection-union 判据：rule-cost CVaR 的方法差 95% CI 上界低于 0，且 nominal win-score 方法差 95% CI 下界高于 `-0.02`。若任一条件不满足，不写“风险改善且性能非劣”。

### 9.2 层级和配对

独立训练 seed 是最高层实验单位；world seed、opponent、seat swap 是其下的配对 block。使用 10,000 次 hierarchical paired bootstrap：

1. 重采样训练 seed；
2. 在每个训练 seed 内重采样 world-seed block；
3. 每个 block 保留四个 opponent 和两个 seat；
4. 对完整方法与基线计算配对差。

报告 95% CI、绝对差、相对差和 probability of improvement。跨六场景的归一化结果额外使用 RLiable 风格的 IQM、stratified bootstrap、performance profile 和 optimality gap。

连续指标使用配对层级 bootstrap difference；win score 使用配对 block permutation 作为条件检验；稀有违规计数补充 exact Poisson/negative-binomial rate ratio。所有窗口级指标必须先聚合到 episode，禁止把窗口数当作自由度。

### 9.3 三个训练种子的限制

三个 seeds 可以完成规定的结果矩阵和区间估计，但不足以对“训练算法总体”做有力的 `p < 0.05` 声明。若只把 3 个训练 seeds 当独立单位，双侧精确符号检验甚至不可能达到 0.05；用数万 episode 代替训练重复会构成 pseudoreplication。

因此采用两阶段协议：

1. Stage A：完成全部 T0-T5 × 3 seeds，结果用于完整消融、失效诊断与效应量估计；Stage A 解盲后不得再把同一结果称为确认性证据；
2. Stage B：冻结 H1-H3、T5/T3/TD-MPC2 的代码、超参数和统计脚本，登记带时间戳的 manifest；各方法扩展到至少 10 个训练 seeds，并只在 Stage A 从未运行过的 confirmation world-seed namespace 上评估。任何根据 Stage B 结果做的方法修改都会使其失去确认性资格，必须再开新的保留集。

如果不执行 Stage B，论文只能写“在三个训练重复和冻结 episode 集上观察到改善”，不能写“统计显著地优于所有方法”。Stage B 的原始逐 episode 结果、预注册 manifest 和失败 runs 也必须发布。

## 10. 结果表与图的固定模板

### 主文表

1. Table 1：T0-T5 在 nominal + OOD aggregate 的 win score、rule CVaR、completion time；
2. Table 2：T1-T5 的 1/5/10 步 position RMSE、NLL、edge-change F1；
3. Table 3：风险 Brier/ECE/AUPRC、predicted-vs-realized CVaR error；
4. Table 4：与 TD-MPC2、DreamerV3、SafeDreamer 的任务结果与计算成本；
5. Table 5：CausalWorld 三个任务的 success、edge event F1、intervention effect error。

### 主文图

1. 方法图：belief 到动态图、stochastic rollout、ego-only MPC；
2. 1/5/10 步误差与 edge event timeline；
3. OOD win/risk Pareto；
4. reliability diagrams；
5. push/no-push 与 armor-hit/bypass 的成对 effect plot；
6. RLiable performance profile 和 probability of improvement。

所有表显示 `mean [95% CI]`，不使用只显示最好 seed 的曲线。训练曲线显示所有 seeds 的透明轨迹和聚合区间。

## 11. 需要新增的工程入口

建议新增而不是继续扩张单文件评估脚本：

```text
isaaclab_sim/rl/world_model/constraint_graph_dynamics.py
isaaclab_sim/rl/replay/sequence_replay.py
isaaclab_sim/rl/experiments/generate_frozen_datasets.py
isaaclab_sim/rl/experiments/run_paper_suite.py
isaaclab_sim/rl/experiments/evaluate_paper_suite.py
isaaclab_sim/rl/experiments/aggregate_paper_results.py
isaaclab_sim/rl/configs/cbg_wm_paper_suite.yaml
isaaclab_sim/rl/configs/cbg_wm_scenario_splits.yaml
tests/test_constraint_graph_dynamics.py
tests/test_paired_interventions.py
tests/test_paper_statistics.py
```

目标命令接口如下；这些命令是实施目标，不代表当前已经可运行：

```powershell
# 20k-step pilot：先测吞吐、显存和指标是否非退化
& C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe `
  isaaclab_sim/rl/experiments/run_paper_suite.py `
  --manifest isaaclab_sim/rl/configs/cbg_wm_paper_suite.yaml `
  --stage pilot --max-parallel 1

# 正式 18-run 核心训练，可断点续跑并跳过已校验 checkpoint
& C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe `
  isaaclab_sim/rl/experiments/run_paper_suite.py `
  --manifest isaaclab_sim/rl/configs/cbg_wm_paper_suite.yaml `
  --stage train --resume --max-parallel 1

# 生成一次冻结数据，然后完成 108 cells
& C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe `
  isaaclab_sim/rl/experiments/generate_frozen_datasets.py `
  --splits isaaclab_sim/rl/configs/cbg_wm_scenario_splits.yaml

& C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe `
  isaaclab_sim/rl/experiments/evaluate_paper_suite.py `
  --manifest isaaclab_sim/rl/configs/cbg_wm_paper_suite.yaml `
  --episodes-per-cell 256 --resume

# 汇总 CI、显著性、表格和图
& C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe `
  isaaclab_sim/rl/experiments/aggregate_paper_results.py `
  --manifest isaaclab_sim/rl/configs/cbg_wm_paper_suite.yaml `
  --bootstrap-reps 10000 --holm-alpha 0.05
```

## 12. 运行产物与可复现性

每个 run 目录必须包含：

```text
manifest.json              # variant, seed, exact CLI, config hash
environment.json           # git commit/diff hash, Python, Torch, CUDA, GPU
train_curve.parquet
checkpoint_best.pt
checkpoint_final.pt
checkpoint.sha256
training_summary.json
exit_status.json
```

每个 evaluation cell 必须包含逐 episode 的 Parquet/CSV，而不仅是 summary JSON。总目录建议：

```text
isaaclab_sim/output/paper/cbg_wm_2026/
  frozen_data/
  train/{variant}/seed_{seed}/
  eval/{variant}/seed_{seed}/{scenario}/
  aggregate/tables/
  aggregate/figures/
  aggregate/statistics/
```

正式 run 开始时若 worktree dirty，记录 diff SHA256；不要自动提交或覆盖用户改动。checkpoint 选择只允许依据 validation composite：

```text
validation_score = normalized_return - 2.0 * rule_cost - 0.5 * edge_nll
```

权重在 pilot 后、正式测试前冻结。不得按 test win rate 选择 checkpoint。

## 13. 计算资源与执行顺序

本机审计结果：RTX 4090 24 GB；`env_isaaclab` 使用 Python 3.11.14、PyTorch 2.7.0+cu128，`torch.cuda.is_available() = True`。这只证明 Torch CUDA 可用，正式开跑前仍需完成 IsaacLab headless、driver 和场景加载 pilot。

单卡不并发训练，`max-parallel=1`。不能根据 8/16-step CPU smoke 外推总时长；先运行 T3/T5 各 20,000 steps，记录稳态 steps/s、VRAM 和 eval latency，再用：

```text
estimated_train_hours = sum_variant(200000 / measured_steps_per_second_variant) / 3600
```

估计总时长。执行顺序：

1. 单元测试、动态图标签 audit、对手动作隔离测试；
2. T3/T5 20k pilot；
3. 冻结配置和数据 split；
4. 18 个核心 runs；
5. 108 cells 与统计汇总；
6. TD-MPC2/DreamerV3/SafeDreamer；
7. CausalWorld；
8. Stage B 额外 seeds；
9. 实机 paired safety/latency trial。

每一阶段只有在产物 schema、NaN 检查、checkpoint hash 和最小指标测试通过后进入下一阶段。

## 14. 论文结果验收门槛

### 工程完成门槛

- [ ] 18/18 核心训练成功且 checkpoint hash 唯一；
- [ ] 108/108 evaluation cells 完成，每 cell 256 对战；
- [ ] 六场景 frozen datasets 和 split audit 随结果保留；
- [ ] 1/5/10 步、edge、calibration、CVaR 和 win 原始数据均可由脚本重算；
- [ ] 三个外部基线和一个公开任务族完成；
- [ ] 无 test-set checkpoint/hyperparameter selection。

### 科学主张门槛

- [ ] T5 相对 T3 在 edge change F1 和 event-time MAE 上有稳定改善；
- [ ] T5 相对 T4 证明 paired intervention loss 有效，而不只是模型容量增加；
- [ ] T5 的 OOD win/risk Pareto 优于或不劣于 T3；
- [ ] 风险下降不能由 shield intervention rate 或更长完成时间完全解释；
- [ ] 公开任务至少在两类关系事件上复现动态图优势；
- [ ] 反事实措辞限定为 controlled intervention，不宣称未证明的 causal identification；
- [ ] 若写“统计显著”，完成 Stage B 的额外训练 seeds。

若 T5 只提升 win rate、却不能提升 edge-change prediction 或 intervention regret，说明新增方法没有被证实，论文应回退为系统集成稿，不应包装为动态图方法论文。

## 15. 2024-2026 顶会/顶刊与官方 GitHub 启发

| 工作 | 状态（截至 2026-07-28） | 可借鉴点 | 本项目不能照搬/夸大的部分 |
|---|---|---|---|
| LPWM | ICLR 2026 Oral；arXiv 2603.04553；官方 `taldatech/lpwm` | stochastic object particles、action-conditioned object dynamics | 其核心是从视频自监督发现对象；本项目使用检测/融合 token，不能声称同等视觉表示贡献 |
| FIOC-WM | NeurIPS 2025；arXiv 2511.02225 | self-transition 与 learned sparse interaction 分解、组合交互 primitive | 仍需证明本项目的 edge lifecycle 超越一般 interaction graph |
| STICA | AAAI 2026；本地 `14167-AAAI26...pdf` | token dependency 与任务相关对象交互 | 原论文的 causality 不是一般意义的可识别因果图，不能作为强因果声明背书 |
| Gamma-World | 2026 arXiv 预印本 2605.28816；官方 `nv-tlabs/Gamma-World` | permutation-equivalent agent identity、稀疏跨 agent 通信 | 重型视频生成和 24 FPS 结果与本项目无直接可比性 |
| CRONOS | 2026 arXiv 预印本 2605.23699 | matched intervention、区分 nuisance stability 与 physical response、报告 sensitivity | 它评估视频物理一致性，不证明本项目的策略因果性 |
| PIGDreamer | ICML 2025（arXiv comment）；arXiv 2508.02159 | privileged training / belief-only execution 的非对称设计 | GitHub README 仍显示 AsymDreamer，整合前必须审计版本，不直接复制 |
| SafeDreamer | ICLR 2024；官方 `PKU-Alignment/SafeDreamer` | reward/cost 分离、Lagrangian safety-reward planning | nearly-zero cost 结果来自 Safety-Gymnasium，不能移植为本任务保证 |
| TD-MPC2 | ICLR 2024；官方 `nicklashansen/tdmpc2` | episodic latent MPC、候选轨迹组织、强连续控制基线 | 不能只“参考代码”而不做正式同预算基线 |
| DreamerV3 | Nature 2025，DOI `10.1038/s41586-025-08744-2`；`danijar/dreamerv3` | 强通用 world-model baseline、固定超参数与 imagined actor | 与对象关系主张不同，主要用于检验整体性能 |
| DINO-WM | arXiv 2411.04983；官方 `gaoyuezhou/dino_wm` | 冻结离线轨迹、test-time planning、PushT/Wall 公共接口 | 未核验到正式会议信息，本文按预印本引用 |
| Forking Uncertainties | IEEE JSAIT 2024，DOI `10.1109/JSAIT.2024.3368229` | sequence model + conformal risk control 对 MPC 的校准思路 | 保证依赖校准分布假设，不能无条件外推 OOD |
| RLiable | NeurIPS 2021 Outstanding Paper；`google-research/rliable` | IQM、stratified bootstrap、probability of improvement、performance profile | 仍不能用大量 episode 掩盖只有 3 个训练 seeds |
| CausalWorld | 官方 `rr-learning/CausalWorld` 公共 benchmark | `do_interventions`、counterfactual env、质量/摩擦/任务分布轴 | 工程依赖较老，必须先做可复现兼容性 gate |

另外检索到 Dyn-O（arXiv 2507.03298）和 OC-STORM（arXiv 2501.16443），它们支持“复杂视觉下对象中心 world model 有价值”，但截至检索时未从元数据核验正式顶会状态，因此只作为 related-work 线索，不写成已录用顶会成果。

## 16. 检索记录与证据等级

检索日期：2026-07-28。

### 数据库与端点

- arXiv API：`https://export.arxiv.org/api/query?id_list=...`，核验 10 个 arXiv 条目的标题、作者、版本、摘要和 author comment；
- OpenAlex：`https://api.openalex.org/works?search=...`，检索 action-conditioned graph、object-centric WM、safe WM、counterfactual planning 和 conformal risk；
- Crossref：`/works/10.1038%2Fs41586-025-08744-2` 与 `/works/10.1109%2FJSAIT.2024.3368229`，核验正式 DOI 元数据；
- GitHub REST search 与官方 raw README，核验 SafeDreamer、DINO-WM、Gamma-World、TD-MPC2、DreamerV3、RLiable、CausalWorld；
- Semantic Scholar 公共 API 返回 HTTP 429，未用其结果；
- OpenReview API 返回 challenge-required HTTP 403，LPWM/SafeDreamer 会议信息改由 author-maintained README、arXiv comment 和公开 forum URL 交叉核对；
- GitHub 未认证搜索在后续请求达到 rate limit，之后仅访问已确认的官方仓库 README，没有用搜索排序代替论文证据。

### 证据等级

1. DOI/Crossref 或正式会议页面；
2. author-maintained official repository + OpenReview/arXiv comment；
3. arXiv 预印本；
4. 聚合列表或第三方复现，仅用于发现线索。

论文 related work 必须在投稿前再次核验 2026 预印本是否已有正式发表版本，并优先引用正式版本。

## 17. 关键链接

- LPWM: <https://openreview.net/forum?id=lTaPtGiUUc>, <https://github.com/taldatech/lpwm>
- FIOC-WM: <https://arxiv.org/abs/2511.02225>
- Gamma-World: <https://arxiv.org/abs/2605.28816>, <https://github.com/nv-tlabs/Gamma-World>
- CRONOS: <https://arxiv.org/abs/2605.23699>, <https://genintel.github.io/CRONOS/>
- PIGDreamer: <https://arxiv.org/abs/2508.02159>, <https://github.com/hggforget/PIGDreamer>
- SafeDreamer: <https://openreview.net/forum?id=tsE5HLYtYg>, <https://github.com/PKU-Alignment/SafeDreamer>
- TD-MPC2: <https://arxiv.org/abs/2310.16828>, <https://github.com/nicklashansen/tdmpc2>
- DreamerV3: <https://doi.org/10.1038/s41586-025-08744-2>, <https://github.com/danijar/dreamerv3>
- DINO-WM: <https://arxiv.org/abs/2411.04983>, <https://github.com/gaoyuezhou/dino_wm>
- Conformal MPC: <https://doi.org/10.1109/JSAIT.2024.3368229>
- RLiable: <https://arxiv.org/abs/2108.13264>, <https://github.com/google-research/rliable>
- CausalWorld: <https://arxiv.org/abs/2010.04296>, <https://github.com/rr-learning/CausalWorld>

---

本方案的完成标准不是“脚本运行结束”，而是核心方法主张、数据切分、对抗协议、统计单位和公开任务都能被独立复核。任何未通过的门槛必须在论文中降级为限制或未来工作，不能用展示视频或单个最好 seed 代替。
