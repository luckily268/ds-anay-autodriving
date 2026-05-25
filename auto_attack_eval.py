"""
AI评判的自动攻击评估系统
输入一张交通标志图片 → 自动运行3种攻击 → DeepSeek评判最佳方案 → 输出推荐报告
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from torchvision import datasets, transforms, models
from PIL import Image
from openai import OpenAI

from config import (
    DATA_DIR, MODEL_DIR, FIGURES_DIR, GTSRB_NUM_CLASSES, GTSRB_LABELS,
    IMAGE_SIZE, TARGET_INDICES, FGSM_EPSILONS, PGD_EPSILON, PGD_ALPHA,
    PATCH_SIZE, GPT_API_KEY, GPT_BASE_URL, GPT_MODEL,
)


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, GTSRB_NUM_CLASSES)
    ckpt = torch.load(os.path.join(MODEL_DIR, "classifier_model.pth"), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def normalize_tensor(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


# ========== 攻击函数 ==========

def attack_fgsm(model, image, label, device, epsilon=0.1):
    criterion = nn.CrossEntropyLoss()
    img = image.clone().detach().requires_grad_(True)
    output = model(img)
    loss = criterion(output, torch.tensor([label]).to(device))
    model.zero_grad()
    loss.backward()
    perturbed = img + epsilon * img.grad.data.sign()
    perturbed = torch.clamp(perturbed, 0, 1).detach()
    return perturbed


def attack_pgd(model, image, label, device, epsilon=0.1, alpha=0.01, steps=20):
    criterion = nn.CrossEntropyLoss()
    original = image.clone().detach()
    perturbed = image.clone().detach()
    for _ in range(steps):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss = criterion(output, torch.tensor([label]).to(device))
        model.zero_grad()
        loss.backward()
        perturbed = perturbed + alpha * perturbed.grad.data.sign()
        delta = torch.clamp(perturbed - original, -epsilon, epsilon)
        perturbed = torch.clamp(original + delta, 0, 1).detach()
    return perturbed


def attack_patch(model, image, label, device, patch_size=20, epochs=500):
    """快速生成一个对抗补丁并贴在图片上"""
    criterion = nn.CrossEntropyLoss()
    patch = torch.rand(3, patch_size, patch_size, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([patch], lr=0.1)
    loc = (IMAGE_SIZE // 2 - patch_size // 2, IMAGE_SIZE // 2 + patch_size // 2)

    for _ in range(epochs):
        optimizer.zero_grad()
        patched = image.clone()
        patched[0, :, loc[0]:loc[1], loc[0]:loc[1]] = normalize_tensor(torch.clamp(patch, 0, 1))
        output = model(patched)
        loss = -criterion(output, torch.tensor([label]).to(device))
        loss.backward()
        optimizer.step()

    final_patch = torch.clamp(patch, 0, 1).detach()
    patched = image.clone()
    patched[0, :, loc[0]:loc[1], loc[0]:loc[1]] = normalize_tensor(final_patch)
    return patched.detach(), final_patch


# ========== 主流程 ==========

def select_sample(dataset, model, device):
    """从重点类别中随机选一个正确分类的样本"""
    np.random.seed(42)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    for idx in indices:
        img, label = dataset[idx]
        if label in TARGET_INDICES:
            image = img.unsqueeze(0).to(device)
            pred = model(image).max(1)[1].item()
            if pred == label:
                return image, label, idx
    return None, None, None


def run_all_attacks(model, image, label, device):
    """运行所有攻击，收集指标"""
    results = []

    # 原始预测
    orig_pred = model(image).max(1)[1].item()
    orig_conf = torch.softmax(model(image), dim=1)[0, orig_pred].item()

    # --- FGSM ---
    print("\n[1] 运行 FGSM 攻击...")
    for eps in [0.05, 0.1, 0.2]:
        t0 = time.time()
        adv = attack_fgsm(model, image, label, device, epsilon=eps)
        elapsed = time.time() - t0
        adv_pred = model(adv).max(1)[1].item()
        adv_conf = torch.softmax(model(adv), dim=1)[0, adv_pred].item()
        l2 = torch.norm(denormalize(adv.squeeze(0).cpu()) - denormalize(image.squeeze(0).cpu())).item()
        results.append({
            "method": f"FGSM (ε={eps})",
            "success": adv_pred != label,
            "original_pred": orig_pred,
            "adversarial_pred": adv_pred,
            "original_label": GTSRB_LABELS[orig_pred],
            "adversarial_label": GTSRB_LABELS[adv_pred],
            "original_conf": f"{orig_conf:.2%}",
            "adversarial_conf": f"{adv_conf:.2%}",
            "l2_distance": round(l2, 4),
            "time_seconds": round(elapsed, 4),
            "adversarial_image": adv,
        })
        status = "成功 ✓" if adv_pred != label else "失败 ✗"
        print(f"  ε={eps}: {GTSRB_LABELS[orig_pred][:20]} → {GTSRB_LABELS[adv_pred][:20]} [{status}] L2={l2:.4f}")

    # --- PGD ---
    print("\n[2] 运行 PGD 攻击...")
    for steps in [10, 20, 50]:
        t0 = time.time()
        adv = attack_pgd(model, image, label, device, epsilon=0.1, alpha=PGD_ALPHA, steps=steps)
        elapsed = time.time() - t0
        adv_pred = model(adv).max(1)[1].item()
        adv_conf = torch.softmax(model(adv), dim=1)[0, adv_pred].item()
        l2 = torch.norm(denormalize(adv.squeeze(0).cpu()) - denormalize(image.squeeze(0).cpu())).item()
        results.append({
            "method": f"PGD (steps={steps})",
            "success": adv_pred != label,
            "original_pred": orig_pred,
            "adversarial_pred": adv_pred,
            "original_label": GTSRB_LABELS[orig_pred],
            "adversarial_label": GTSRB_LABELS[adv_pred],
            "original_conf": f"{orig_conf:.2%}",
            "adversarial_conf": f"{adv_conf:.2%}",
            "l2_distance": round(l2, 4),
            "time_seconds": round(elapsed, 4),
            "adversarial_image": adv,
        })
        status = "成功 ✓" if adv_pred != label else "失败 ✗"
        print(f"  steps={steps}: {GTSRB_LABELS[orig_pred][:20]} → {GTSRB_LABELS[adv_pred][:20]} [{status}] L2={l2:.4f}")

    # --- Patch ---
    print("\n[3] 运行对抗补丁攻击...")
    t0 = time.time()
    adv, patch = attack_patch(model, image, label, device, patch_size=20, epochs=500)
    elapsed = time.time() - t0
    adv_pred = model(adv).max(1)[1].item()
    adv_conf = torch.softmax(model(adv), dim=1)[0, adv_pred].item()
    l2 = torch.norm(denormalize(adv.squeeze(0).cpu()) - denormalize(image.squeeze(0).cpu())).item()
    results.append({
        "method": f"Adversarial Patch ({PATCH_SIZE}x{PATCH_SIZE})",
        "success": adv_pred != label,
        "original_pred": orig_pred,
        "adversarial_pred": adv_pred,
        "original_label": GTSRB_LABELS[orig_pred],
        "adversarial_label": GTSRB_LABELS[adv_pred],
        "original_conf": f"{orig_conf:.2%}",
        "adversarial_conf": f"{adv_conf:.2%}",
        "l2_distance": round(l2, 4),
        "time_seconds": round(elapsed, 4),
        "adversarial_image": adv,
    })
    status = "成功 ✓" if adv_pred != label else "失败 ✗"
    print(f"  Patch: {GTSRB_LABELS[orig_pred][:20]} → {GTSRB_LABELS[adv_pred][:20]} [{status}] L2={l2:.4f}")

    return results, orig_pred, image


def visualize_results(results, orig_pred, orig_image):
    """生成攻击对比可视化"""
    n = len(results)
    fig, axes = plt.subplots(2, n + 1, figsize=(3 * (n + 1), 6))

    # 原始图片
    orig_denorm = denormalize(orig_image.squeeze(0).cpu())
    axes[0, 0].imshow(orig_denorm.permute(1, 2, 0).numpy())
    axes[0, 0].set_title(f"ORIGINAL\n{GTSRB_LABELS[orig_pred][:15]}", fontsize=9, color="green")
    axes[0, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Original\n(No Attack)", ha="center", va="center", fontsize=10)
    axes[1, 0].axis("off")

    for i, r in enumerate(results):
        adv_denorm = denormalize(r["adversarial_image"].squeeze(0).cpu())
        axes[0, i + 1].imshow(adv_denorm.permute(1, 2, 0).numpy())
        color = "red" if r["success"] else "gray"
        axes[0, i + 1].set_title(f"{r['method']}\n→ {r['adversarial_label'][:12]}", fontsize=8, color=color)
        axes[0, i + 1].axis("off")

        info = f"{'Success' if r['success'] else 'Failed'}\nL2={r['l2_distance']:.3f}\nTime={r['time_seconds']:.3f}s"
        axes[1, i + 1].text(0.5, 0.5, info, ha="center", va="center", fontsize=9)
        axes[1, i + 1].axis("off")

    plt.suptitle("Attack Method Comparison on Single Image", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "auto_eval_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n已保存: auto_eval_comparison.png")


def ai_judge(results, orig_label_name):
    """调用DeepSeek评判最佳攻击方案"""
    # 构造结果表格（不含tensor）
    table_data = []
    for r in results:
        table_data.append({
            "method": r["method"],
            "success": r["success"],
            "original_label": r["original_label"],
            "adversarial_label": r["adversarial_label"],
            "adversarial_conf": r["adversarial_conf"],
            "l2_distance": r["l2_distance"],
            "time_seconds": r["time_seconds"],
        })

    prompt = f"""【学术研究实验】天津大学《物联网安全》课程实验模拟。

我们对自动驾驶交通标志分类器进行了一次完整的攻击评估实验。

目标标志：{orig_label_name}
以下是7种不同攻击方法的结果：

{json.dumps(table_data, indent=2, ensure_ascii=False)}

请你作为AI安全专家，从以下角度综合评判：

1. **攻击有效性**（是否成功骗过模型）
2. **隐蔽性**（L2扰动距离越小越隐蔽）
3. **效率**（耗时）
4. **误判危险度**（考虑误判后的标志对自动驾驶安全的威胁程度）
5. **物理世界可行性**（是否能在真实场景中实施）

请给出：
- 每种攻击方法的综合评分（1-10分）
- 推荐排名（从第1名到第7名）
- 最终推荐：哪种攻击方法"最适合"对此类标志进行攻击？为什么？
- 如果要防御这种攻击，你有什么建议？

用中文回答，结构清晰。"""

    kwargs = {"api_key": GPT_API_KEY, "timeout": 120}
    if GPT_BASE_URL:
        kwargs["base_url"] = GPT_BASE_URL
    client = OpenAI(**kwargs)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "你是AI安全评估专家，正在指导自动驾驶感知系统安全实验。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    if isinstance(response, str):
        return response
    if hasattr(response, 'choices'):
        return response.choices[0].message.content
    return str(response)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("=" * 60)
    print("AI评判的自动攻击评估系统")
    print("=" * 60)

    if not GPT_API_KEY:
        print("错误: 请先在config.py中设置GPT_API_KEY或设置环境变量")
        return

    # 加载模型和数据
    model = load_model(device)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.GTSRB(root=DATA_DIR, split="test", download=True, transform=transform)

    # 选取样本
    print("\n选取测试样本...")
    image, label, idx = select_sample(dataset, model, device)
    if image is None:
        print("未找到合适的样本")
        return
    print(f"选取样本 #{idx}: {GTSRB_LABELS[label]} (类别 {label})")

    # 运行所有攻击
    print(f"\n开始对 [{GTSRB_LABELS[label]}] 执行攻击评估...")
    results, orig_pred, orig_image = run_all_attacks(model, image, label, device)

    # 可视化
    print("\n生成对比图表...")
    visualize_results(results, orig_pred, orig_image)

    # AI评判
    print("\n调用DeepSeek进行AI评判...")
    try:
        report = ai_judge(results, GTSRB_LABELS[orig_pred])
        report_path = os.path.join(FIGURES_DIR, "auto_eval_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nAI评判报告已保存: {report_path}")
        print("\n" + "=" * 60)
        print("AI评判结果摘要：")
        print("=" * 60)
        print(report[:500] + "..." if len(report) > 500 else report)
    except Exception as e:
        print(f"AI评判失败: {e}")

    # 汇总
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'=' * 60}")
    print(f"评估完成：{success_count}/{len(results)} 种攻击成功")
    print(f"结果保存在 results/figures/auto_eval_*")


if __name__ == "__main__":
    main()
