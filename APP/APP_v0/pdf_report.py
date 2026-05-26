"""
PDF report generation using reportlab with Chinese font support.
"""
import io
import os
from datetime import datetime
from PIL import Image
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from inference_engine import InferenceResult
from widgets.image_viewer import numpy_mask_to_pil, generate_overlay

# ---- Chinese font registration (Windows) ----
# Try multiple common Chinese fonts in order of preference.
# We need a TrueType font that has CJK glyphs; Helvetica does not.

_CHINESE_FONT_PATHS = [
    ("C:/Windows/Fonts/msyh.ttc", 0),   # 微软雅黑 Regular (TTC index 0)
    ("C:/Windows/Fonts/simhei.ttf", None),  # 黑体
    ("C:/Windows/Fonts/simsun.ttc", 0),  # 宋体 (TTC index 0)
    ("C:/Windows/Fonts/STSONG.TTF", None),  # 华文宋体
    ("C:/Windows/Fonts/simfang.ttf", None),  # 仿宋
]

_CJK_FONT_NAME = None  # resolved font name usable with canvas.setFont()


def _register_cjk_font():
    """Find and register a Chinese-capable font. Called once on first PDF export."""
    global _CJK_FONT_NAME
    if _CJK_FONT_NAME is not None:
        return _CJK_FONT_NAME

    for path, ttc_index in _CHINESE_FONT_PATHS:
        if not os.path.exists(path):
            continue
        try:
            kwargs = {}
            if ttc_index is not None:
                kwargs["subfontIndex"] = ttc_index
            pdfmetrics.registerFont(TTFont("CJK", path, **kwargs))
            _CJK_FONT_NAME = "CJK"
            return _CJK_FONT_NAME
        except Exception:
            continue

    # No Chinese font found — fall back to Helvetica (text will be garbled)
    _CJK_FONT_NAME = "Helvetica"
    return _CJK_FONT_NAME


def generate_pdf_report(
    output_path: str,
    result: InferenceResult,
    doctor_id: str,
    doctor_name: str,
    patient_id: str,
    patient_name: str,
):
    font_name = _register_cjk_font()

    page_w, page_h = A4
    margin = 20 * mm
    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle("超声图像诊断报告")

    # ---- Header ----
    c.setFont(font_name, 18)
    c.drawString(margin, page_h - 30 * mm, "超声图像智能诊断报告")

    c.setFont(font_name, 10)
    y = page_h - 40 * mm
    c.drawString(margin, y, f"报告日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 7 * mm
    c.drawString(margin, y, f"医生: {doctor_name} (编号: {doctor_id})")
    y -= 7 * mm
    c.drawString(margin, y, f"患者: {patient_name} (编号: {patient_id})")
    y -= 7 * mm

    # Separator line
    c.setStrokeColor(colors.grey)
    c.line(margin, y, page_w - margin, y)
    y -= 10 * mm

    # ---- Images (2x2 grid) ----
    img_size = 65 * mm
    gap = 5 * mm

    original_img = result.original_image.convert("RGB")
    fine_mask_img = numpy_mask_to_pil(result.binary_mask_final).convert("RGB")
    fine_edge_img = numpy_mask_to_pil(result.binary_fine_edge).convert("RGB")
    overlay_img = generate_overlay(result.original_image, result.binary_mask_final).convert("RGB")

    images = [
        ("原始图像", original_img),
        ("精细分割图 (Stage 2)", fine_mask_img),
        ("边缘轮廓", fine_edge_img),
        ("重叠图", overlay_img),
    ]

    positions = [
        (margin,                    y - img_size),
        (margin + img_size + gap,   y - img_size),
        (margin,                    y - img_size * 2 - gap),
        (margin + img_size + gap,   y - img_size * 2 - gap),
    ]

    for (title, img), (px, py) in zip(images, positions):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        reader = ImageReader(buf)
        c.drawImage(reader, px, py, img_size, img_size, preserveAspectRatio=True)
        c.setFont(font_name, 8)
        c.drawString(px, py - 5 * mm, title)

    # ---- Diagnosis Result ----
    y = positions[3][1] - 17 * mm

    c.setFont(font_name, 14)
    c.drawString(margin, y, "诊断结果:")

    y -= 10 * mm
    if result.pred_class_idx == 1:
        c.setFillColor(colors.red)
        diag_text = "恶性 (Malignant)"
    else:
        c.setFillColor(colors.green)
        diag_text = "良性 (Benign)"
    c.setFont(font_name, 18)
    c.drawString(margin, y, diag_text)

    c.setFillColor(colors.black)
    c.setFont(font_name, 11)
    y -= 9 * mm
    c.drawString(margin, y, f"置信度: {result.confidence * 100:.1f}%")

    # ---- Footer ----
    c.setFont(font_name, 7)
    c.setFillColor(colors.grey)
    c.drawString(margin, 10 * mm, "本报告由 AI 辅助诊断系统自动生成，仅供临床参考。")

    c.save()
