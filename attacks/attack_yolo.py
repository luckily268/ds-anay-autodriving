"""
YOLO 目标检测逃逸攻击
攻击 YOLOv8 使其无法检测到行人或车辆
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

from config import IMAGES_DIR, FIGURES_DIR

YOLO_IMG_SIZE = 640


def download_sample_images():
    """获取ultralytics自带的本地测试图片"""
    import glob
    try:
        import ultralytics
        pkg_dir = os.path.dirname(ultralytics.__file__)
        assets_dir = os.path.join(pkg_dir, "assets")
        if os.path.exists(assets_dir):
            imgs = glob.glob(os.path.join(assets_dir, "*.jpg")) + glob.glob(os.path.join(assets_dir, "*.png"))
            if imgs:
                print(f"找到 {len(imgs)} 张ultralytics自带图片")
                return imgs[:3]
    except Exception:
        pass
    return []


def preprocess_for_yolo(img_path, device):
    """加载图片并预处理为YOLO输入格式 (1,3,640,640)"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((YOLO_IMG_SIZE, YOLO_IMG_SIZE), Image.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor


def yolo_fgsm_attack(yolo_model, image_tensor, epsilon=0.05):
    """
    使用FGSM攻击YOLO底层模型
    通过直接调用PyTorch模型获取梯度
    """
    raw_model = yolo_model.model
    raw_model.eval()

    image_tensor = image_tensor.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        # YOLO前向传播，获取原始输出
        preds = raw_model(image_tensor)[0]  # shape: (1, 84, 8400)
        # 84 = 4(bbox) + 80(classes)
        # 取所有预测的类别置信度
        class_scores = preds[0, 4:, :]  # (80, 8400)

        # 损失：最大化所有检测分数（让模型在加扰动后更不确定）
        loss = -class_scores.max(dim=0)[0].sum()

        loss.backward()

    gradient = image_tensor.grad.data
    perturbed = image_tensor + epsilon * gradient.sign()
    perturbed = torch.clamp(perturbed, 0, 1).detach()

    return perturbed


def save_tensor_as_image(tensor, path):
    """将tensor保存为图片"""
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def run_yolo_attack():
    """执行YOLO检测逃逸攻击实验"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载 YOLOv8n 模型...")
    yolo_model = YOLO("yolov8n.pt")

    print("准备测试图片...")
    image_paths = download_sample_images()

    if not image_paths:
        print("没有找到测试图片，跳过YOLO攻击实验")
        return

    for img_path in image_paths:
        print(f"\n处理图片: {os.path.basename(img_path)}")

        # 原始检测结果（用原始尺寸的图片）
        original_results = yolo_model(img_path, conf=0.25, verbose=False)
        original_img = original_results[0].plot()
        orig_count = len(original_results[0].boxes)
        print(f"  原始检测: {orig_count} 个目标")

        # 预处理为640x640
        img_tensor = preprocess_for_yolo(img_path, device)

        # FGSM攻击
        print("  执行FGSM逃逸攻击...")
        perturbed_tensor = yolo_fgsm_attack(yolo_model, img_tensor, epsilon=0.05)

        # 保存扰动后的图片
        perturbed_path = os.path.join(IMAGES_DIR, f"perturbed_{os.path.basename(img_path)}")
        save_tensor_as_image(perturbed_tensor, perturbed_path)

        # 攻击后检测
        attacked_results = yolo_model(perturbed_path, conf=0.25, verbose=False)
        attacked_img = attacked_results[0].plot()
        attacked_count = len(attacked_results[0].boxes)

        print(f"  攻击后检测: {attacked_count} 个目标")
        print(f"  逃逸目标数: {orig_count - attacked_count}")

        # 可视化对比
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(original_img[..., ::-1])
        axes[0].set_title(f"Original Detection: {orig_count} objects", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(attacked_img[..., ::-1])
        axes[1].set_title(f"After FGSM Attack: {attacked_count} objects", fontsize=14)
        axes[1].axis("off")

        plt.suptitle("YOLO Object Detection Evasion Attack", fontsize=16)
        plt.tight_layout()

        save_name = f"yolo_evasion_{os.path.basename(img_path).replace('.jpg', '.png')}"
        plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {save_name}")

        # 额外：不同epsilon的对比
        print("  测试不同扰动强度...")
        epsilons = [0.01, 0.03, 0.05, 0.1]
        eps_results = []
        for eps in epsilons:
            p_tensor = yolo_fgsm_attack(yolo_model, img_tensor, epsilon=eps)
            p_path = os.path.join(IMAGES_DIR, f"perturbed_eps{eps}_{os.path.basename(img_path)}")
            save_tensor_as_image(p_tensor, p_path)
            p_results = yolo_model(p_path, conf=0.25, verbose=False)
            p_count = len(p_results[0].boxes)
            eps_results.append((eps, p_count))
            print(f"    ε={eps}: {p_count} 个目标")

        # 画epsilon对比图
        fig, ax = plt.subplots(figsize=(8, 5))
        eps_list = [0] + [e for e, _ in eps_results]
        count_list = [orig_count] + [c for _, c in eps_results]
        ax.plot(eps_list, count_list, "ro-", linewidth=2, markersize=8)
        ax.set_xlabel("Perturbation Strength (ε)", fontsize=12)
        ax.set_ylabel("Detected Objects", fontsize=12)
        ax.set_title("YOLO Detection vs Adversarial Perturbation", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "yolo_evasion_vs_epsilon.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  已保存: yolo_evasion_vs_epsilon.png")


def main():
    print("YOLO 目标检测逃逸攻击实验")
    print("=" * 50)
    run_yolo_attack()
    print("\nYOLO 攻击实验完成！")


if __name__ == "__main__":
    main()
