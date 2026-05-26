"""
Left-side file list panel with thumbnail previews.
"""
import os
from PIL import Image
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def make_thumbnail(image_path: str, size: int = 64) -> QPixmap:
    try:
        img = Image.open(image_path)
        img.thumbnail((size, size), Image.LANCZOS)
        from widgets.image_viewer import pil_to_qpixmap
        return pil_to_qpixmap(img.convert('RGB'))
    except Exception:
        return QPixmap()


class FileListPanel(QWidget):
    file_selected = pyqtSignal(str)
    files_imported = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("图像列表")
        title.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setIconSize(QSize(48, 48))
        self._list.setSpacing(2)
        self._list.setStyleSheet("QListWidget::item { padding: 4px; }")
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        self._file_paths: list[str] = []
        self._status: dict[str, str] = {}  # file_path -> 'pending' | 'done'

    def add_single(self, file_path: str):
        if file_path not in self._file_paths:
            self._file_paths.append(file_path)
            self._status[file_path] = 'pending'
            self._add_item(file_path)
        self._list.setCurrentRow(len(self._file_paths) - 1)
        self.file_selected.emit(file_path)

    def add_folder(self, folder_path: str):
        new_files = []
        for fname in os.listdir(folder_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(folder_path, fname)
                if full_path not in self._file_paths:
                    self._file_paths.append(full_path)
                    self._status[full_path] = 'pending'
                    new_files.append(full_path)
        for fp in new_files:
            self._add_item(fp)
        if new_files:
            self._list.setCurrentRow(0)
            self.file_selected.emit(self._file_paths[0])
        if self._file_paths:
            self.files_imported.emit(self._file_paths)

    def mark_done(self, file_path: str):
        self._status[file_path] = 'done'
        idx = self._file_paths.index(file_path) if file_path in self._file_paths else -1
        if idx >= 0:
            item = self._list.item(idx)
            current_text = item.text()
            if not current_text.endswith(' ✓'):
                item.setText(current_text + ' ✓')

    def clear(self):
        self._file_paths.clear()
        self._status.clear()
        self._list.clear()

    def current_file(self) -> str | None:
        items = self._list.selectedItems()
        if items:
            return items[0].data(Qt.ItemDataRole.UserRole)
        if self._file_paths:
            return self._file_paths[0]
        return None

    def _add_item(self, file_path: str):
        fname = os.path.basename(file_path)
        thumb = make_thumbnail(file_path)
        item = QListWidgetItem()
        item.setText(fname)
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        if not thumb.isNull():
            item.setIcon(QIcon(thumb))
        self._list.addItem(item)

    def _on_item_clicked(self, item: QListWidgetItem):
        fp = item.data(Qt.ItemDataRole.UserRole)
        if fp:
            self.file_selected.emit(fp)
