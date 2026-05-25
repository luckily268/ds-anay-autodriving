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

### 4. 配置GPT API（用于创新实验模块,也可以使用其他Ai）

设置环境变量：
```bash
# Windows CMD
set GPT_API_KEY=your-api-key-here
set GPT_BASE_URL=your-base-url-here
set GPT_MODEL=gpt-5.4-mini

# Windows PowerShell
$env:GPT_API_KEY="your-api-key-here"
$env:GPT_BASE_URL="your-base-url-here"
$env:GPT_MODEL="gpt-5.4-mini"

# Linux/Mac
export GPT_API_KEY=your-api-key-here
export GPT_BASE_URL=your-base-url-here
export GPT_MODEL=gpt-5.4-mini
```

## 项目结构

```
.
├── config.py                    # 全局配置参数
├── requirements.txt             # 依赖列表
├── README.md                    # 本文件
│
├── models/
│   └── train_classifier.py      # 训练交通标志分类器
│
├── attacks/
│   ├── attack_fgsm.py           # FGSM攻击
│   ├── attack_pgd.py            # PGD攻击
│   ├── attack_patch.py          # 对抗补丁攻击
│   └── attack_yolo.py           # YOLO检测逃逸攻击
│
├── gpt_assistant/
│   └── gpt_analyzer.py          # GPT辅助攻击分析
│
├── evaluation/
│   └── evaluate_attacks.py      # 综合攻击效果评估
│
├── data/                        # GTSRB数据集（自动下载）
├── results/
│   ├── images/                  # 对抗样本图片
│   ├── figures/                 # 实验图表
│   └── logs/                    # 运行日志
└── docs/                        # 文档
```

## 运行步骤

**重要：所有命令都需在 `IoT-Security` 环境中运行！**

```bash
conda activate IoT-Security
```

### Step 1: 训练交通标志分类器

```bash
cd models
python train_classifier.py
```

- 自动下载GTSRB数据集（约300MB）
- 使用ResNet18迁移学习训练
- CPU约30-60分钟，GPU约5-10分钟
- 训练完成后模型保存在 `models/classifier_model.pth`
- 预期准确率 > 95%

### Step 2: 运行攻击实验

**FGSM攻击（最快，建议先运行）：**
```bash
cd attacks
python attack_fgsm.py
```

**PGD攻击：**
```bash
python attack_pgd.py
```

**对抗补丁攻击：**
```bash
python attack_patch.py
```

**YOLO检测逃逸攻击：**
```bash
python attack_yolo.py
```

### Step 3: 综合评估

```bash
cd evaluation
python evaluate_attacks.py
```

### Step 4: GPT辅助分析（需配置API Key）

```bash
cd gpt_assistant
python gpt_analyzer.py
```

## 实验结果

所有实验结果保存在 `results/` 目录下：

- `results/figures/` - 可视化图表（PNG格式）
  - `fgsm_asr_vs_epsilon.png` - FGSM攻击成功率随扰动强度变化曲线
  - `fgsm_examples.png` - FGSM对抗样本对比图
  - `pgd_asr_vs_steps.png` - PGD攻击成功率随迭代步数变化
  - `pgd_examples.png` - PGD对抗样本对比图
  - `adversarial_patch.png` - 对抗补丁及训练损失
  - `patch_on_signs.png` - 补丁贴在交通标志上的效果
  - `yolo_evasion_*.png` - YOLO检测逃逸效果对比
  - `vulnerability_ranking.png` - 各类别脆弱性排名
  - `method_comparison.png` - 攻击方法综合对比
  - `gpt_*.png` - GPT分析报告

- `results/images/` - 对抗样本图片
- `results/logs/` - 训练日志和实验数据

## 攻击方法说明

| 攻击方法 | 文件 | 速度 | 攻击效果 | 特点 |
|---------|------|------|---------|------|
| FGSM | attack_fgsm.py | 极快(秒级) | 良好 | 最基础的梯度攻击 |
| PGD | attack_pgd.py | 较快(分钟级) | 很强 | FGSM的迭代强化版 |
| Patch | attack_patch.py | 中等 | 良好 | 物理世界可实施 |
| YOLO逃逸 | attack_yolo.py | 快 | 良好 | 让检测器"看不见"目标 |

## 常见问题

**Q: 训练模型时内存不足怎么办？**
A: 修改 `config.py` 中的 `BATCH_SIZE` 为更小的值（如32或16）。

**Q: GTSRB数据集下载失败？**
A: 可以手动从 https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 下载数据集，放到 `data/` 目录下。

**Q: YOLO攻击时图片下载失败？**
A: 可以使用自己的包含行人/车辆的图片，修改 `attack_yolo.py` 中的图片路径。

**Q: 没有GPU可以用吗？**
A: 可以。所有实验在CPU上都能运行，只是速度较慢。FGSM和PGD在CPU上很快，Patch训练稍慢。
