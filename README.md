# 面向自动驾驶感知系统的多维度对抗样本攻击研究与实验

天津大学《物联网安全》课程结课作业（设计类）

## 环境要求

- Python 3.10+
- Conda（推荐）
- 无需GPU（CPU可运行，有GPU更快）

## 安装步骤

### 1. 创建并激活Conda虚拟环境

```bash
conda create -n IoT-Security python=3.10 -y
conda activate IoT-Security
```

### 2. 安装PyTorch（CPU版本）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

如果有NVIDIA GPU，安装GPU版本：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 4. 配置LLM API（用于AI辅助分析模块）

在 `config.py` 中设置 `GPT_API_KEY`，或通过环境变量：
```bash
# Windows PowerShell
$env:GPT_API_KEY="your-api-key-here"

# Linux/Mac
export GPT_API_KEY=your-api-key-here
```

默认使用 DeepSeek API（`deepseek-v4-flash` 模型），可在 `config.py` 中修改。

## 项目结构

```
IoT/
├── config.py                       # 全局配置参数
├── requirements.txt                # 依赖列表
├── README.md                       # 本文件
│
├── models/
│   └── train_classifier.py         # 训练交通标志分类器
│
├── attacks/                        # 攻击脚本
│   ├── attack_fgsm.py              # FGSM攻击（一步梯度攻击）
│   ├── attack_pgd.py               # PGD攻击（迭代梯度攻击）
│   ├── attack_patch.py             # 对抗补丁攻击（物理世界攻击）
│   ├── attack_yolo.py              # YOLO检测逃逸攻击（让目标消失）
│   └── auto_attack_eval.py         # AI评判的自动攻击评估系统（整合脚本）
│
├── gpt_assistant/
│   └── gpt_analyzer.py             # DeepSeek辅助攻击分析
│
├── evaluation/
│   └── evaluate_attacks.py         # 综合攻击效果评估
│
├── data/                           # GTSRB数据集（自动下载，位于上级目录）
└── results/
    ├── images/                     # 对抗样本图片
    ├── figures/                    # 实验图表 + AI分析报告
    └── logs/                       # 训练日志和实验数据
```

## 运行步骤

**重要：所有命令都需在 `IoT-Security` 环境中运行！**

```bash
conda activate IoT-Security
```

### Step 1: 训练交通标志分类器

```bash
cd IoT/models
python train_classifier.py
```

- 自动下载GTSRB数据集（约300MB）到上级目录的 `data/` 文件夹
- 使用ResNet18迁移学习训练（ImageNet预训练 + GTSRB微调）
- CPU约30-60分钟，GPU约5-10分钟
- 训练完成后模型保存在 `models/classifier_model.pth`
- 预期准确率 > 95%（实际达到98.61%）

### Step 2: 运行攻击实验

```bash
cd IoT/attacks
```

按顺序运行：

```bash
python attack_fgsm.py      # FGSM攻击（秒级，建议先跑）
python attack_pgd.py        # PGD攻击（分钟级）
python attack_patch.py      # 对抗补丁攻击（10-30分钟生成补丁）
python attack_yolo.py       # YOLO检测逃逸攻击（分钟级）
```

### Step 3: 综合评估

```bash
cd IoT/evaluation
python evaluate_attacks.py
```

- 评估模型在干净数据上的准确率
- 统计各类别FGSM攻击脆弱性
- 生成脆弱性排名图和方法对比图

### Step 4: DeepSeek辅助分析（需配置API Key）

```bash
cd IoT/gpt_assistant
python gpt_analyzer.py
```

生成3份AI分析报告：
1. 最危险的交通标志误判组合分析
2. 针对性攻击策略建议
3. 语义危害评估报告（ASR × 后果严重度）

### Step 5: AI评判的自动攻击评估系统

```bash
cd IoT/attacks
python auto_attack_eval.py
```

这是整合脚本，自动完成：
1. 从GTSRB选取一张交通标志
2. 运行7种攻击变体（FGSM×3 + PGD×3 + Patch×1）
3. 收集攻击指标（成功率、扰动大小、耗时）
4. 调用DeepSeek从5个维度（有效性、隐蔽性、效率、危险度、物理可行性）评分
5. 输出推荐排名和最佳攻击方法

## 实验结果

所有实验结果保存在 `results/` 目录下：

### 攻击实验图表
| 文件名 | 内容 |
|--------|------|
| `fgsm_asr_vs_epsilon.png` | FGSM攻击成功率随扰动强度变化曲线 |
| `fgsm_examples.png` | FGSM对抗样本对比（原图 vs 对抗图 vs 扰动放大） |
| `pgd_asr_vs_steps.png` | PGD攻击成功率随迭代步数变化 |
| `pgd_examples.png` | PGD对抗样本对比 |
| `adversarial_patch.png` | 对抗补丁 + 训练损失曲线 + 攻击成功率 |
| `patch_on_signs.png` | 补丁贴在不同交通标志上的效果 |
| `yolo_evasion_bus.png` | YOLO检测逃逸效果对比（公交车场景） |
| `yolo_evasion_zidane.png` | YOLO检测逃逸效果对比（人物场景） |
| `yolo_evasion_vs_epsilon.png` | YOLO检测数量随扰动强度变化 |

### 评估图表
| 文件名 | 内容 |
|--------|------|
| `vulnerability_ranking.png` | 各交通标志类别脆弱性排名 |
| `accuracy_vs_epsilon.png` | 模型准确率随攻击强度下降曲线 |
| `method_comparison.png` | 攻击方法三维对比（成功率/扰动/速度） |

### AI分析报告
| 文件名 | 内容 |
|--------|------|
| `gpt_dangerous_pairs.txt` | 10个最危险的交通标志误判组合 |
| `gpt_attack_strategy.txt` | 针对危险组合的攻击策略与防御建议 |
| `gpt_semantic_danger.txt` | 语义危害评估报告（综合风险评级） |
| `auto_eval_report.txt` | AI评判的自动攻击评估报告（推荐最佳攻击方法） |
| `auto_eval_comparison.png` | 单张图片上7种攻击的效果对比 |

### 其他
| 文件名 | 内容 |
|--------|------|
| `patch_sticker.png` | 单独的对抗补丁图片（可打印） |
| `training_log.txt` | 模型训练日志（15个epoch） |
| `attack_summary.json` | 综合评估实验数据（JSON格式） |

## 攻击方法说明

| 攻击方法 | 文件 | 原理 | 速度 | 特点 |
|---------|------|------|------|------|
| FGSM | attack_fgsm.py | 一步梯度方向加扰动 | 极快(秒级) | 最基础的梯度攻击 |
| PGD | attack_pgd.py | 多步迭代梯度攻击 | 较快(分钟级) | FGSM的强化版，成功率更高 |
| Patch | attack_patch.py | 训练生成可打印的对抗贴纸 | 中等 | 唯一可在物理世界实施的攻击 |
| YOLO逃逸 | attack_yolo.py | 降低检测置信度使目标消失 | 快 | 攻击检测任务而非分类任务 |

## 关键实验数据

- **模型**：ResNet18，ImageNet预训练 + GTSRB微调
- **干净数据准确率**：98.61%
- **最脆弱类别**：Speed limit (100km/h)，FGSM攻击成功率80%
- **最危险类别**：Stop，语义危害分数300（ASR 60% × 严重度5.0）
- **AI综合评级**：高风险（High Risk）

## 常见问题

**Q: 训练模型时内存不足怎么办？**
A: 修改 `config.py` 中的 `BATCH_SIZE` 为更小的值（如32或16）。

**Q: GTSRB数据集下载失败？**
A: 可以手动从 https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 下载数据集，放到 `data/` 目录下。

**Q: YOLO攻击时图片下载失败？**
A: 脚本会自动使用ultralytics自带的本地测试图片，无需下载。

**Q: 没有GPU可以用吗？**
A: 可以。所有实验在CPU上都能运行。FGSM和PGD在CPU上很快，Patch生成稍慢。

**Q: DeepSeek API调用超时？**
A: 已设置120秒超时。如果仍然超时，检查网络连接或更换API服务。
