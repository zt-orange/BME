"""
Core inference engine — 100% equivalent to infer.py but without torch/torchvision.
Uses pure PIL + NumPy for preprocessing to minimize package size.
"""
import numpy as np
from PIL import Image
import onnxruntime as ort
from dataclasses import dataclass


@dataclass
class InferenceResult:
    original_image: Image.Image
    pred_class: str
    pred_class_idx: int
    confidence: float
    binary_mask_stage1: np.ndarray
    binary_mask_final: np.ndarray
    binary_fine_edge: np.ndarray
    attention_map: np.ndarray


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


class InferenceEngine:
    def __init__(self, onnx_path: str):
        self.ort_session = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider']
        )
        self.class_map = {0: 'Benign', 1: 'Malignant'}

    def preprocess(self, image_path: str, img_size: int = 224):
        image = Image.open(image_path).convert("RGB")

        # Visualization copy (resized original)
        image_viz = image.resize((img_size, img_size), Image.BILINEAR)

        # PIL resize (bilinear, same as torchvision TF.resize default)
        img_resized = image.resize((img_size, img_size), Image.BILINEAR)

        # To tensor: HWC [0,255] -> CHW [0.0, 1.0] float32
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std

        # Add batch dimension
        input_numpy = np.expand_dims(img_array, axis=0).astype(np.float32)

        return input_numpy, image_viz

    def infer(self, input_array: np.ndarray):
        ort_inputs = {'input_image': input_array}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs

    def postprocess(self, ort_outputs):
        seg_stage1_logits = ort_outputs[0][0, 0, :, :]
        seg_final_logits = ort_outputs[1][0, 0, :, :]
        class_output = ort_outputs[2][0]

        # --- Classification ---
        pred_class_idx = int(np.argmax(class_output))
        pred_class_name = self.class_map[pred_class_idx]
        class_probs = softmax(class_output)
        confidence = float(class_probs[pred_class_idx])

        # --- Stage 1 mask ---
        mask_prob_stage1 = sigmoid(seg_stage1_logits)
        attention_map = (mask_prob_stage1 < 0.95).astype(np.float32) * mask_prob_stage1
        binary_mask_stage1 = (mask_prob_stage1 > 0.5).astype(np.uint8)

        # --- Final (Stage 2) mask ---
        mask_prob_final = sigmoid(seg_final_logits)
        binary_mask_final = (mask_prob_final > 0.5).astype(np.uint8)

        # --- Fine edge ---
        fine_edge_prob = (mask_prob_final < 0.95).astype(np.float32) * mask_prob_final
        binary_fine_edge = (fine_edge_prob > 0.5).astype(np.uint8)

        return InferenceResult(
            original_image=None,  # set by caller
            pred_class=pred_class_name,
            pred_class_idx=pred_class_idx,
            confidence=confidence,
            binary_mask_stage1=binary_mask_stage1,
            binary_mask_final=binary_mask_final,
            binary_fine_edge=binary_fine_edge,
            attention_map=attention_map,
        )

    def run(self, image_path: str):
        input_array, image_viz = self.preprocess(image_path)
        ort_outputs = self.infer(input_array)
        result = self.postprocess(ort_outputs)
        result.original_image = image_viz
        return result
