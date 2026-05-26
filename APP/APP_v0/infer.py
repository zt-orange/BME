import onnxruntime as ort
import numpy as np
from PIL import Image
# Force 'Agg' backend for writing to files without GUI pop-ups
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF


def sigmoid(x):
    """NumPy implementation of the Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def preprocess_image(image_path, img_size=224):
    """
    Image preprocessing: Must strictly match the training pipeline.
    """
    # 1. Read image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Save a copy of the original image for visualization
    image_viz = image.resize((img_size, img_size), Image.BILINEAR)

    # 2. Resize to strictly 224x224
    image_tensor = TF.resize(image, (img_size, img_size))

    # 3. Convert to tensor (C, H, W) and scale to [0, 1]
    image_tensor = TF.to_tensor(image_tensor)

    # 4. ImageNet normalization
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 5. Add batch dimension (1, C, H, W) and convert to NumPy array
    input_numpy = image_tensor.unsqueeze(0).numpy()

    return input_numpy, image_viz


def main():
    # ================= Path Configuration =================
    onnx_path = "./models/model.onnx"
    img_path = "./bus_0341-r.png"
    class_map = {0: 'Benign', 1: 'Malignant'}

    print("1. Loading and preprocessing image...")
    input_data, img_viz = preprocess_image(img_path, img_size=224)

    print("2. Starting ONNX Runtime session...")
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    print("3. Executing model inference...")
    ort_inputs = {'input_image': input_data}
    ort_outs = ort_session.run(None, ort_inputs)

    # Extract outputs and remove dummy dimensions (Batch_size=1, Channel=1)
    seg_stage1_logits = ort_outs[0][0, 0, :, :]
    seg_final_logits = ort_outs[1][0, 0, :, :]
    class_output = ort_outs[2][0]

    print("4. Post-processing and calculating edge attention...")
    # --- A. Classification Result ---
    pred_class_idx = np.argmax(class_output)
    pred_class_name = class_map[pred_class_idx]

    # --- B. Coarse Segmentation to Probability ---
    mask_prob_stage1 = sigmoid(seg_stage1_logits)

    # --- C. Extract Edge Attention Map (Stage 1) ---
    attention_map = (mask_prob_stage1 < 0.95).astype(np.float32) * mask_prob_stage1

    # --- D. Fine Segmentation to Probability ---
    mask_prob_final = sigmoid(seg_final_logits)

    # Binarize masks (Threshold = 0.5) to strict 0 and 1
    binary_mask_stage1 = (mask_prob_stage1 > 0.5).astype(np.uint8)
    binary_mask_final = (mask_prob_final > 0.5).astype(np.uint8)

    # --- E. Extract Fine Edge Map (Stage 2 - Binary) ---
    # 提取最终精细图的边缘，并二值化为黑白图
    fine_edge_prob = (mask_prob_final < 0.95).astype(np.float32) * mask_prob_final
    binary_fine_edge = (fine_edge_prob > 0.5).astype(np.uint8)

    print(f"Prediction: {pred_class_name}")
    print("5. Generating and saving 6 separate visualization images...")

    # ================= Visualization (Saved Separately) =================
    title_color = 'red' if pred_class_idx == 1 else 'green'

    # ---------------- 1. Save Original Image ----------------
    fig1 = plt.figure(figsize=(5, 5))
    plt.imshow(img_viz)
    plt.title(f"Original Image\nPred: {pred_class_name}", fontsize=14, color=title_color)
    plt.axis('off')
    plt.savefig("1_original_image.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ---------------- 2. Save Coarse Mask (Strict Binary) ----------------
    fig2 = plt.figure(figsize=(5, 5))
    plt.imshow(binary_mask_stage1, cmap='gray')
    plt.title(f"Coarse Mask (Stage 1)\nPred: {pred_class_name}", fontsize=14, color=title_color)
    plt.axis('off')
    plt.savefig("2_coarse_mask.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # ---------------- 3. Save Edge Attention Map ----------------
    fig3 = plt.figure(figsize=(6, 5))
    im3 = plt.imshow(attention_map, cmap='jet')
    plt.title(f"Edge Attention Map\nPred: {pred_class_name}", fontsize=14, color=title_color)
    plt.axis('off')
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    plt.savefig("3_edge_attention_map.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # ---------------- 4. Save Final Fine Mask (Strict Binary) ----------------
    fig4 = plt.figure(figsize=(5, 5))
    plt.imshow(binary_mask_final, cmap='gray')
    plt.title(f"Final Fine Mask (Stage 2)\nPred: {pred_class_name}", fontsize=14, color=title_color)
    plt.axis('off')
    plt.savefig("4_final_fine_mask.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # ---------------- 5. Save Final Fine Edge Map (Strict Binary) ----------------
    fig5 = plt.figure(figsize=(5, 5))
    plt.imshow(binary_fine_edge, cmap='gray')
    plt.title(f"Final Fine Edge (Binary)\nPred: {pred_class_name}", fontsize=14, color=title_color)
    plt.axis('off')
    plt.savefig("5_final_fine_edge_mask.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # ---------------- 6. Save Overlay (Original + Red Contour) ----------------
    # 【新增代码部分】：将最终的精细掩码作为轮廓叠加上去
    fig6 = plt.figure(figsize=(5, 5))
    # 首先绘制原图作为底图
    plt.imshow(img_viz)
    # 使用 contour 函数提取 binary_mask_final (0和1的矩阵) 的等高线
    # levels=[0.5] 表示在 0 和 1 之间的分界处画线，即边缘
    plt.contour(binary_mask_final, levels=[0.5], colors='red', linewidths=2)
    plt.title(f"Segmentation Overlay\nPred: {pred_class_name}", fontsize=14, color=title_color)
    plt.axis('off')
    plt.savefig("6_overlay_contour.png", dpi=300, bbox_inches='tight')
    plt.close(fig6)

