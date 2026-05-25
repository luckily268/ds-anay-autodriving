"""
PGD (Projected Gradient Descent) 攻击
FGSM 的多步迭代强化版本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from config import (
    DATA_DIR, MODEL_DIR, IMAGES_DIR, FIGURES_DIR,
    GTSRB_NUM_CLASSES, GTSRB_LABELS, IMAGE_SIZE,
    PGD_EPSILON, PGD_ALPHA, PGD_STEPS_LIST, TARGET_INDICES,
)


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, GTSRB_NUM_CLASSES)
    ckpt_path = os.path.join(MODEL_DIR, "classifier_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return datasets.GTSRB(root=DATA_DIR, split="test", download=True, transform=transform)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def pgd_attack(model, image, label, epsilon, alpha, num_steps, device):
    """执行PGD攻击"""
    criterion = nn.CrossEntropyLoss()
    original = image.clone().detach()
    perturbed = image.clone().detach()

    for _ in range(num_steps):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss = criterion(output, torch.tensor([label]).to(device))
        model.zero_grad()
        loss.backward()
        gradient = perturbed.grad.data

        perturbed = perturbed + alpha * gradient.sign()
        # 投影到 epsilon 球内
        delta = torch.clamp(perturbed - original, -epsilon, epsilon)
        perturbed = torch.clamp(original + delta, 0, 1).detach()

    return perturbed


def run_pgd_experiment(model, dataset, device):
    """测试不同迭代步数的PGD攻击"""
    # 收集样本
    samples = {}
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label in TARGET_INDICES and label not in samples:
            image = img.unsqueeze(0).to(device)
            pred = model(image).max(1)[1].item()
            if pred == label:
                samples[label] = (image, label)
        if len(samples) >= len(TARGET_INDICES):
            break

    results = {}
    best_examples = []

    for num_steps in PGD_STEPS_LIST:
        success_count = 0
        total_count = 0

        for true_label, (image, label) in samples.items():
            adv = pgd_attack(model, image, label, PGD_EPSILON, PGD_ALPHA, num_steps, device)
            adv_pred = model(adv).max(1)[1].item()

            total_count += 1
            if adv_pred != label:
                success_count += 1

            if num_steps == PGD_STEPS_LIST[-1]:
                best_examples.append({
                    "original": denormalize(image.squeeze(0).cpu()),
                    "adversarial": denormalize(adv.squeeze(0).cpu()),
                    "true_label": label,
                    "adversarial_pred": adv_pred,
                })

        asr = 100.0 * success_count / total_count if total_count > 0 else 0
        results[num_steps] = {"asr": asr, "success": success_count, "total": total_count}
        print(f"PGD steps={num_steps}: ASR={asr:.1f}% ({success_count}/{total_count})")

    return results, best_examples


def visualize_pgd_results(results, examples):
    """生成PGD攻击可视化"""
    # 1. 迭代步数 vs 攻击成功率
    steps = sorted(results.keys())
    asrs = [results[s]["asr"] for s in steps]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(steps)), asrs, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"])
    plt.xticks(range(len(steps)), [f"Steps={s}" for s in steps])
    plt.ylabel("Attack Success Rate (%)", fontsize=12)
    plt.title(f"PGD Attack: ASR vs Iteration Steps (ε={PGD_EPSILON})", fontsize=14)
    plt.ylim(0, 105)
    for i, v in enumerate(asrs):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(os.path.join(FIGURES_DIR, "pgd_asr_vs_steps.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存: pgd_asr_vs_steps.png")

    # 2. FGSM vs PGD 对比（需要读取FGSM结果）
    # 对抗样本展示
    n = min(len(examples), 5)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i, ex in enumerate(examples[:n]):
        axes[0, i].imshow(ex["original"].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"Original\n{GTSRB_LABELS[ex['true_label']][:15]}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(ex["adversarial"].permute(1, 2, 0).numpy())
        adv_name = GTSRB_LABELS[ex["adversarial_pred"]][:15]
        axes[1, i].set_title(f"PGD Adversarial\n→ {adv_name}", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle(f"PGD Adversarial Examples (ε={PGD_EPSILON}, steps={PGD_STEPS_LIST[-1]})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pgd_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存: pgd_examples.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_model(device)
    dataset = get_test_loader()

    print(f"\n开始 PGD 攻击实验 (ε={PGD_EPSILON}, α={PGD_ALPHA})")
    print(f"测试迭代步数: {PGD_STEPS_LIST}")

    results, examples = run_pgd_experiment(model, dataset, device)

    print("\n生成可视化图表...")
    visualize_pgd_results(results, examples)

    print("\nPGD 攻击实验完成！")


if __name__ == "__main__":
    main()
