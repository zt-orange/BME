"""
Main application window for the ultrasound image inference tool.
"""
import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSplitter, QFileDialog, QMessageBox, QStatusBar,
    QLabel, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QFont

from inference_engine import InferenceEngine, InferenceResult
from widgets.image_viewer import ImageViewer, generate_overlay
from widgets.file_list import FileListPanel
from widgets.result_panel import ResultPanel
from pdf_report import generate_pdf_report


class InferenceWorker(QObject):
    finished = pyqtSignal(str, object)  # file_path, InferenceResult
    error = pyqtSignal(str, str)        # file_path, error_message
    _process_signal = pyqtSignal(str)   # internal: queued invocation

    def __init__(self, engine: InferenceEngine):
        super().__init__()
        self._engine = engine
        self._process_signal.connect(self._do_process)

    def request_process(self, image_path: str):
        """Call from main thread — delivered via queued connection to worker thread."""
        self._process_signal.emit(image_path)

    def _do_process(self, image_path: str):
        try:
            result = self._engine.run(image_path)
            self.finished.emit(image_path, result)
        except Exception as e:
            self.error.emit(image_path, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("超声图像智能诊断系统")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 820)

        # ---- Model / Engine ----
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "model", "model.onnx")
        self._engine: InferenceEngine | None = None
        self._results_cache: dict[str, InferenceResult] = {}

        # ---- Inference thread ----
        self._thread = QThread()
        self._worker: InferenceWorker | None = None

        # ---- Build UI ----
        self._setup_ui()
        self._setup_statusbar()

        # ---- Load model after UI is ready ----
        QTimer.singleShot(100, lambda: self._load_model(model_path))

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # ---- Top toolbar ----
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(6)

        self._btn_import_single = QPushButton("导入单张图像")
        self._btn_import_single.clicked.connect(self._import_single)
        self._btn_import_single.setEnabled(False)
        toolbar_layout.addWidget(self._btn_import_single)

        self._btn_import_folder = QPushButton("导入文件夹（批量）")
        self._btn_import_folder.clicked.connect(self._import_folder)
        self._btn_import_folder.setEnabled(False)
        toolbar_layout.addWidget(self._btn_import_folder)

        toolbar_layout.addWidget(self._separator())

        self._btn_export = QPushButton("导出当前报告")
        self._btn_export.clicked.connect(self._export_report)
        self._btn_export.setEnabled(False)
        toolbar_layout.addWidget(self._btn_export)

        toolbar_layout.addStretch()

        # Style toolbar buttons
        for btn in [self._btn_import_single, self._btn_import_folder, self._btn_export]:
            btn.setMinimumHeight(32)
            btn.setStyleSheet(
                "QPushButton { padding: 6px 16px; font-size: 13px; }"
                "QPushButton:disabled { color: #999; }"
            )

        main_layout.addLayout(toolbar_layout)

        # ---- Main splitter (left | center | right) ----
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: File list
        self._file_list = FileListPanel()
        self._file_list.setMinimumWidth(160)
        self._file_list.setMaximumWidth(280)
        self._file_list.file_selected.connect(self._on_file_selected)
        splitter.addWidget(self._file_list)

        # Center: Image viewer
        self._image_viewer = ImageViewer()
        splitter.addWidget(self._image_viewer)

        # Right: Result panel
        self._result_panel = ResultPanel()
        self._result_panel.setMinimumWidth(260)
        self._result_panel.setMaximumWidth(320)
        self._result_panel.export_requested.connect(self._export_report)
        self._result_panel.reset_requested.connect(self._reset_all)
        splitter.addWidget(self._result_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([200, 800, 280])

        main_layout.addWidget(splitter)

    def _setup_statusbar(self):
        self._status = QStatusBar()
        self._status_label = QLabel("正在加载模型...")
        self._status.addWidget(self._status_label)
        self.setStatusBar(self._status)

    def _separator(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet("color: #ccc;")
        return f

    # ===================== Model Loading =====================

    def _load_model(self, model_path: str):
        try:
            self._engine = InferenceEngine(model_path)
            self._status_label.setText("模型加载完成 — 请导入图像")
            self._btn_import_single.setEnabled(True)
            self._btn_import_folder.setEnabled(True)

            # Set up worker thread
            self._worker = InferenceWorker(self._engine)
            self._worker.moveToThread(self._thread)
            self._worker.finished.connect(self._on_inference_done)
            self._worker.error.connect(self._on_inference_error)
            self._thread.start()
        except Exception as e:
            self._status_label.setText(f"模型加载失败: {e}")
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")

    # ===================== Import =====================

    def _import_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择超声图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if path:
            self._file_list.add_single(path)

    def _import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self._file_list.add_folder(folder)

    # ===================== File Selection → Inference =====================

    def _on_file_selected(self, file_path: str):
        if file_path in self._results_cache:
            self._display_result(file_path, self._results_cache[file_path])
            return

        if self._worker and self._engine:
            self._status_label.setText(f"正在推理: {os.path.basename(file_path)} ...")
            self._btn_export.setEnabled(False)
            self._worker.request_process(file_path)

    # ===================== Inference Callbacks =====================

    def _on_inference_done(self, file_path: str, result: InferenceResult):
        self._results_cache[file_path] = result
        self._file_list.mark_done(file_path)
        self._display_result(file_path, result)
        self._status_label.setText(f"推理完成: {os.path.basename(file_path)}")
        self._btn_export.setEnabled(True)

    def _on_inference_error(self, file_path: str, error_msg: str):
        self._status_label.setText(f"推理失败: {error_msg}")
        QMessageBox.warning(self, "推理错误",
                            f"图像 {os.path.basename(file_path)} 推理失败:\n{error_msg}")

    # ===================== Display =====================

    def _display_result(self, file_path: str, result: InferenceResult):
        overlay = generate_overlay(result.original_image, result.binary_mask_final)
        self._image_viewer.set_result(
            result.original_image,
            result.binary_mask_final,
            result.binary_fine_edge,
            overlay,
        )
        self._result_panel.set_result(result)

    # ===================== Export PDF =====================

    def _export_report(self):
        current = self._file_list.current_file()
        if not current or current not in self._results_cache:
            QMessageBox.information(self, "提示", "请先完成图像推理后再导出报告。")
            return

        doctor_id, doctor_name = self._result_panel.get_doctor_info()
        patient_id, patient_name = self._result_panel.get_patient_info()

        if not doctor_id or not doctor_name:
            QMessageBox.information(self, "提示", "请先填写医生信息。")
            return
        if not patient_id or not patient_name:
            QMessageBox.information(self, "提示", "请先填写患者信息。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存PDF报告", f"超声诊断报告_{patient_name}.pdf",
            "PDF文件 (*.pdf)"
        )
        if not path:
            return

        try:
            result = self._results_cache[current]
            generate_pdf_report(
                path, result,
                doctor_id, doctor_name,
                patient_id, patient_name,
            )
            self._status_label.setText(f"报告已导出: {os.path.basename(path)}")
            QMessageBox.information(self, "成功", f"报告已保存至:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出报告失败:\n{e}")

    # ===================== Reset =====================

    def _reset_all(self):
        self._results_cache.clear()
        self._image_viewer.clear()
        self._file_list.clear()
        self._result_panel.reset()
        self._btn_export.setEnabled(False)
        self._status_label.setText("模型就绪 — 请导入图像")

    # ===================== Cleanup =====================

    def closeEvent(self, event):
        self._thread.quit()
        self._thread.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Global font
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # Dark-ish stylesheet
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QPushButton {
            background-color: #fff; border: 1px solid #ccc;
            border-radius: 4px; padding: 5px 12px;
        }
        QPushButton:hover { background-color: #e8f0fe; border-color: #5b9cf5; }
        QPushButton:checked { background-color: #d0e4fc; border-color: #3b7cf5; }
        QPushButton:pressed { background-color: #c0d8f8; }
        QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 6px;
                     margin-top: 8px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QLineEdit { border: 1px solid #ccc; border-radius: 3px; padding: 4px 6px; }
        QLineEdit:focus { border-color: #5b9cf5; }
        QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; }
        QProgressBar::chunk { border-radius: 3px; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
