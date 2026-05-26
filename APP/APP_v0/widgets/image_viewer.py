"""
Zoomable/pannable image viewer with multi-view switching.
"""
import numpy as np
from PIL import Image, ImageDraw
import cv2
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
    QButtonGroup, QHBoxLayout, QPushButton, QLabel,
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QImage, QPainter


def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    """Convert PIL Image to QPixmap."""
    if pil_image.mode == 'L':
        data = pil_image.tobytes('raw', 'L')
        qimg = QImage(data, pil_image.width, pil_image.height,
                      pil_image.width, QImage.Format.Format_Grayscale8)
    elif pil_image.mode == 'RGBA':
        data = pil_image.tobytes('raw', 'RGBA')
        qimg = QImage(data, pil_image.width, pil_image.height,
                      pil_image.width * 4, QImage.Format.Format_RGBA8888)
    else:
        rgb = pil_image.convert('RGB')
        data = rgb.tobytes('raw', 'RGB')
        qimg = QImage(data, rgb.width, rgb.height,
                      rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def numpy_mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convert a uint8 numpy mask (0/1) to a grayscale PIL image (0/255)."""
    return Image.fromarray(mask * 255, mode='L')


def generate_overlay(original: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Draw red contours of binary mask over the original image.
    Equivalent to matplotlib's:
        plt.contour(binary_mask_final, levels=[0.5], colors='red', linewidths=2)
    """
    binary = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_array = np.array(original.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_bgr, contours, -1, (0, 0, 255), 2)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView with mouse wheel zoom support."""
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #1e1e1e; border: none;")

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.scale(factor, factor)


class ImageViewer(QWidget):
    VIEW_ORIGINAL = 0
    VIEW_FINE_MASK = 1
    VIEW_FINE_EDGE = 2
    VIEW_OVERLAY = 3
    VIEW_QUAD = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original = None
        self._fine_mask = None
        self._fine_edge = None
        self._overlay = None
        self._current_view = self.VIEW_ORIGINAL

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # View toggle buttons
        btn_layout = QHBoxLayout()
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)

        views = [
            ("原始图像", self.VIEW_ORIGINAL),
            ("精细分割图", self.VIEW_FINE_MASK),
            ("边缘轮廓", self.VIEW_FINE_EDGE),
            ("重叠图", self.VIEW_OVERLAY),
            ("四图同列", self.VIEW_QUAD),
        ]
        for label, vid in views:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, v=vid: self._switch_view(v))
            self._view_group.addButton(btn, vid)
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

        # Graphics view
        self._scene = QGraphicsScene()
        self._view = ZoomableGraphicsView(self._scene, self)
        layout.addWidget(self._view)

        # Placeholder
        self._placeholder = QLabel("请导入超声图像")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 18px;")
        layout.addWidget(self._placeholder)

        self._view_group.button(self.VIEW_ORIGINAL).setChecked(True)

    def clear(self):
        self._original = None
        self._fine_mask = None
        self._fine_edge = None
        self._overlay = None
        self._scene.clear()
        self._placeholder.show()

    def set_result(self, original: Image.Image, fine_mask: np.ndarray,
                   fine_edge: np.ndarray, overlay_img: Image.Image):
        self._original = original
        self._fine_mask = fine_mask
        self._fine_edge = fine_edge
        self._overlay = overlay_img
        self._placeholder.hide()
        self._switch_view(self._current_view)

    def _switch_view(self, view_idx: int):
        self._current_view = view_idx
        self._scene.clear()

        if not self._original:
            return

        if view_idx == self.VIEW_QUAD:
            pix = self._build_quad_view()
        elif view_idx == self.VIEW_ORIGINAL:
            pix = pil_to_qpixmap(self._original)
        elif view_idx == self.VIEW_FINE_MASK and self._fine_mask is not None:
            pix = pil_to_qpixmap(numpy_mask_to_pil(self._fine_mask))
        elif view_idx == self.VIEW_FINE_EDGE and self._fine_edge is not None:
            pix = pil_to_qpixmap(numpy_mask_to_pil(self._fine_edge))
        elif view_idx == self.VIEW_OVERLAY and self._overlay:
            pix = pil_to_qpixmap(self._overlay)
        else:
            return

        self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _build_quad_view(self) -> QPixmap:
        if not self._original:
            return QPixmap()

        size = 224
        gap = 4
        canvas = Image.new('RGB', (size * 2 + gap, size * 2 + gap), color=(30, 30, 30))

        imgs = [
            ("原始图像", self._original.convert('RGB')),
            ("精细分割图", numpy_mask_to_pil(self._fine_mask).convert('RGB')),
            ("边缘轮廓", numpy_mask_to_pil(self._fine_edge).convert('RGB')),
            ("重叠图", (self._overlay or Image.new('RGB', (size, size), (30, 30, 30))).convert('RGB')),
        ]

        positions = [
            (0, 0), (size + gap, 0),
            (0, size + gap), (size + gap, size + gap),
        ]

        for (title, img), (x, y) in zip(imgs, positions):
            canvas.paste(img.resize((size, size), Image.LANCZOS), (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 4, y + 4), title, fill='white')

        return pil_to_qpixmap(canvas)
