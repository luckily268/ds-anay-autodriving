"""
FGSM (Fast Gradient Sign Method) 攻击
对交通标志分类器生成对抗样本
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
    FGSM_EPSILONS, TARGET_INDICES,
)


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, GTSRB_NUM_CLASSES)
    ckpt_path = os.path.join(MODEL_DIR, "classifier_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"模型加载成功，训练准确率: {checkpoint['best_acc']:.2f}%")
    return model


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.GTSRB(root=DATA_DIR, split="test", download=True, transform=transform)
    return dataset


def fgsm_attack(image, epsilon, gradient):
    perturbed = image + epsilon * gradient.sign()
    return torch.clamp(perturbed, 0, 1)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def run_fgsm_on_samples(model, dataset, device, num_samples=20):
    """对指定类别样本执行FGSM攻击，测试不同epsilon"""
    criterion = nn.CrossEntropyLoss()

    # 收集目标类别的样本
    samples = {}
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label in TARGET_INDICES and label not in samples:
            samples[label] = (img.unsqueeze(0).to(device), label)
        if len(samples) >= len(TARGET_INDICES):
            break

    # 如果没收集够，从数据集前面找
    for idx in range(len(dataset)):
        if len(samples) >= len(TARGET_INDICES):
            break
        img, label = dataset[idx]
        if label in TARGET_INDICES and label not in samples:
            samples[label] = (img.unsqueeze(0).to(device), label)

    results = {}

    for epsilon in FGSM_EPSILONS:
        success_count = 0
        total_count = 0
        adv_examples = []

        for true_label, (image, label) in samples.items():
            image.requires_grad = True
            output = model(image)
            original_pred = output.max(1, keepdim=True)[1].item()

            if original_pred != label:
                continue

            loss = criterion(output, torch.tensor([label]).to(device))
            model.zero_grad()
            loss.backward()
            gradient = image.grad.data

            perturbed = fgsm_attack(image, epsilon, gradient)
            perturbed = perturbed.detach()
            perturbed.requires_grad = False

            # 反归一化用于保存
            perturbed_denorm = denormalize(perturbed.squeeze(0).cpu())
            original_denorm = denormalize(image.squeeze(0).detach().cpu())

            output_adv = model(perturbed)
            adv_pred = output_adv.max(1, keepdim=True)[1].item()

            total_count += 1
            if adv_pred != label:
                success_count += 1

            adv_examples.append({
                "original": original_denorm,
                "adversarial": perturbed_denorm,
                "true_label": label,
                "original_pred": original_pred,
                "adversarial_pred": adv_pred,
                "epsilon": epsilon,
            })

        asr = 100.0 * success_count / total_count if total_count > 0 else 0
        results[epsilon] = {
            "asr": asr,
            "success": success_count,
            "total": total_count,
            "examples": adv_examples,
        }
        print(f"epsilon={epsilon:.2f}: 攻击成功率={asr:.1f}% ({success_count}/{total_count})")

    return results


def visualize_fgsm_results(results):
    """生成FGSM攻击可视化"""
    # 1. 攻击成功率 vs epsilon 曲线
    epsilons = sorted(results.keys())
    asrs = [results[e]["asr"] for e in epsilons]

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, asrs, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Perturbation Strength (ε)", fontsize=12)
    plt.ylabel("Attack Success Rate (%)", fontsize=12)
    plt.title("FGSM Attack: Success Rate vs Perturbation Strength", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "fgsm_asr_vs_epsilon.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存: fgsm_asr_vs_epsilon.png")

    # 2. 对抗样本对比图（选取中间epsilon的结果）
    mid_eps = epsilons[len(epsilons) // 2]
    examples = results[mid_eps]["examples"]
    n = min(len(examples), 5)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i, ex in enumerate(examples[:n]):
        # 原始图片
        axes[0, i].imshow(ex["original"].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"Original\n{GTSRB_LABELS[ex['true_label']][:15]}", fontsize=8)
        axes[0, i].axis("off")

        # 对抗样本
        axes[1, i].imshow(ex["adversarial"].permute(1, 2, 0).numpy())
        adv_name = GTSRB_LABELS[ex["adversarial_pred"]][:15]
        axes[1, i].set_title(f"Adversarial (ε={mid_eps})\n→ {adv_name}", fontsize=8)
        axes[1, i].axis("off")

        # 扰动放大
        diff = (ex["adversarial"] - ex["original"]).permute(1, 2, 0).numpy()
        diff_vis = np.clip(np.abs(diff) * 10, 0, 1)
        axes[2, i].imshow(diff_vis)
        axes[2, i].set_title("Perturbation (10x)", fontsize=8)
        axes[2, i].axis("off")

    plt.suptitle("FGSM Adversarial Examples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fgsm_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存: fgsm_examples.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_model(device)
    dataset = get_test_loader()

    print(f"\n开始 FGSM 攻击实验...")
    print(f"测试 epsilon 值: {FGSM_EPSILONS}")

    results = run_fgsm_on_samples(model, dataset, device)

    print("\n生成可视化图表...")
    visualize_fgsm_results(results)

    print("\nFGSM 攻击实验完成！")


if __name__ == "__main__":
    main()
