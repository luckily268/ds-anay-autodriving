"""
Adversarial Patch 攻击
生成一个可打印的对抗补丁，贴在交通标志附近导致误判
最贴近物理世界的攻击方式
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
    PATCH_SIZE, PATCH_LEARNING_RATE, PATCH_EPOCHS, TARGET_INDICES,
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


def normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def generate_adversarial_patch(model, dataset, device, target_label=None):
    """
    训练生成一个对抗补丁
    target_label: 目标误判类别（None表示无目标攻击，只要不是正确类别即可）
    """
    # 收集训练图片
    images_list = []
    labels_list = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label in TARGET_INDICES:
            pred = model(img.unsqueeze(0).to(device)).max(1)[1].item()
            if pred == label:
                images_list.append(img)
                labels_list.append(label)
        if len(images_list) >= 20:
            break

    # 初始化补丁（随机噪声）
    patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([patch], lr=PATCH_LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()
    patch_location = (IMAGE_SIZE // 2 - PATCH_SIZE // 2, IMAGE_SIZE // 2 + PATCH_SIZE // 2)

    print(f"生成对抗补丁 (size={PATCH_SIZE}x{PATCH_SIZE}, epochs={PATCH_EPOCHS})...")
    losses = []

    for epoch in range(PATCH_EPOCHS):
        # 随机选一张图
        idx = np.random.randint(0, len(images_list))
        image = images_list[idx].clone().to(device)
        label = labels_list[idx]

        # 将补丁贴到图片上
        clamped_patch = torch.clamp(patch, 0, 1)
        patched_image = image.clone()
        patched_image[
            :, patch_location[0]:patch_location[1], patch_location[0]:patch_location[1]
        ] = normalize(clamped_patch)

        patched_image = patched_image.unsqueeze(0)

        output = model(patched_image)

        if target_label is not None:
            # 有目标攻击
            loss = criterion(output, torch.tensor([target_label]).to(device))
        else:
            # 无目标攻击：最大化正确类别的损失
            loss = -criterion(output, torch.tensor([label]).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            pred = output.max(1)[1].item()
            losses.append(loss.item())
            print(f"  epoch {epoch+1}/{PATCH_EPOCHS}, loss={loss.item():.4f}, pred={pred}")

    final_patch = torch.clamp(patch, 0, 1).detach()
    return final_patch, losses


def test_patch_effectiveness(model, dataset, patch, device):
    """测试对抗补丁在测试集上的攻击效果"""
    patch_location = (IMAGE_SIZE // 2 - PATCH_SIZE // 2, IMAGE_SIZE // 2 + PATCH_SIZE // 2)
    clamped_patch = torch.clamp(patch, 0, 1)

    results = []
    for idx in range(min(200, len(dataset))):
        img, label = dataset[idx]
        if label not in TARGET_INDICES:
            continue

        image = img.unsqueeze(0).to(device)
        original_pred = model(image).max(1)[1].item()
        if original_pred != label:
            continue

        # 贴上补丁
        patched = image.clone()
        patched[
            0, :, patch_location[0]:patch_location[1], patch_location[0]:patch_location[1]
        ] = normalize(clamped_patch)

        adv_pred = model(patched).max(1)[1].item()

        results.append({
            "true_label": label,
            "original_pred": original_pred,
            "adversarial_pred": adv_pred,
            "success": adv_pred != label,
        })

    total = len(results)
    success = sum(1 for r in results if r["success"])
    asr = 100.0 * success / total if total > 0 else 0

    print(f"\n补丁攻击成功率: {asr:.1f}% ({success}/{total})")

    # 按类别统计
    for label in TARGET_INDICES:
        label_results = [r for r in results if r["true_label"] == label]
        if label_results:
            label_success = sum(1 for r in label_results if r["success"])
            label_total = len(label_results)
            print(f"  {GTSRB_LABELS[label][:20]}: {100*label_success/label_total:.1f}% ({label_success}/{label_total})")

    return results, asr


def visualize_patch_results(patch, losses, results, asr):
    """生成对抗补丁攻击可视化"""
    # 1. 补丁本身
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(patch.cpu().permute(1, 2, 0).numpy())
    axes[0].set_title(f"Adversarial Patch ({PATCH_SIZE}x{PATCH_SIZE})", fontsize=12)
    axes[0].axis("off")

    # 2. 训练损失曲线
    axes[1].plot(losses, "b-", linewidth=1)
    axes[1].set_xlabel("Training Step (x100)")
    axes[2] = axes[1]
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Patch Training Loss")
    axes[1].grid(True, alpha=0.3)

    # 3. 攻击成功率
    axes[2].bar(["Attack Success\nRate"], [asr], color="#e74c3c", width=0.5)
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_title(f"Overall ASR: {asr:.1f}%")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adversarial_patch.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存: adversarial_patch.png")

    # 保存补丁图片
    plt.figure(figsize=(3, 3))
    plt.imshow(patch.cpu().permute(1, 2, 0).numpy())
    plt.title("Adversarial Patch")
    plt.axis("off")
    plt.savefig(os.path.join(IMAGES_DIR, "patch_sticker.png"), dpi=150, bbox_inches="tight")
    plt.close()


def apply_patch_to_samples(model, dataset, patch, device, num_samples=5):
    """展示补丁贴在图片上的效果"""
    patch_location = (IMAGE_SIZE // 2 - PATCH_SIZE // 2, IMAGE_SIZE // 2 + PATCH_SIZE // 2)
    clamped_patch = torch.clamp(patch, 0, 1)

    samples = []
    for idx in range(len(dataset)):
        if len(samples) >= num_samples:
            break
        img, label = dataset[idx]
        if label not in TARGET_INDICES:
            continue

        image = img.unsqueeze(0).to(device)
        pred = model(image).max(1)[1].item()
        if pred != label:
            continue

        patched = image.clone()
        patched[
            0, :, patch_location[0]:patch_location[1], patch_location[0]:patch_location[1]
        ] = normalize(clamped_patch)

        adv_pred = model(patched).max(1)[1].item()

        samples.append({
            "original": denormalize(image.squeeze(0).cpu()),
            "patched": denormalize(patched.squeeze(0).cpu()),
            "true_label": label,
            "adv_pred": adv_pred,
        })

    if not samples:
        return

    fig, axes = plt.subplots(2, len(samples), figsize=(3 * len(samples), 6))
    for i, s in enumerate(samples):
        axes[0, i].imshow(s["original"].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"Original\n{GTSRB_LABELS[s['true_label']][:15]}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(s["patched"].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f"With Patch\n→ {GTSRB_LABELS[s['adv_pred']][:12]}", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle("Adversarial Patch Attack on Traffic Signs", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "patch_on_signs.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存: patch_on_signs.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_model(device)
    dataset = get_test_loader()

    # 生成对抗补丁（无目标攻击）
    patch, losses = generate_adversarial_patch(model, dataset, device, target_label=None)

    # 测试补丁效果
    results, asr = test_patch_effectiveness(model, dataset, patch, device)

    # 可视化
    print("\n生成可视化图表...")
    visualize_patch_results(patch, losses, results, asr)
    apply_patch_to_samples(model, dataset, patch, device)

    # 保存补丁张量
    torch.save(patch.cpu(), os.path.join(MODEL_DIR, "adversarial_patch.pth"))
    print(f"\n对抗补丁已保存到 {MODEL_DIR}/adversarial_patch.pth")
    print("对抗补丁攻击实验完成！")


if __name__ == "__main__":
    main()
