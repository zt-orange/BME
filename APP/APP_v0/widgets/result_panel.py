"""
Right-side result panel showing classification, confidence, and controls.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QHBoxLayout,
    QCheckBox, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ResultPanel(QWidget):
    export_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ---- Doctor Info ----
        doctor_group = QGroupBox("医生信息")
        doc_form = QFormLayout(doctor_group)
        self._doctor_id = QLineEdit()
        self._doctor_id.setPlaceholderText("请输入医生编号")
        self._doctor_name = QLineEdit()
        self._doctor_name.setPlaceholderText("请输入医生姓名")
        doc_form.addRow("编号:", self._doctor_id)
        doc_form.addRow("姓名:", self._doctor_name)
        layout.addWidget(doctor_group)

        # ---- Patient Info ----
        patient_group = QGroupBox("患者信息")
        pat_form = QFormLayout(patient_group)
        self._patient_id = QLineEdit()
        self._patient_id.setPlaceholderText("请输入患者编号")
        self._patient_name = QLineEdit()
        self._patient_name.setPlaceholderText("请输入患者姓名")
        pat_form.addRow("编号:", self._patient_id)
        pat_form.addRow("姓名:", self._patient_name)
        layout.addWidget(patient_group)

        # ---- Result ----
        result_group = QGroupBox("诊断结果")
        result_layout = QVBoxLayout(result_group)

        self._result_label = QLabel("等待推理...")
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; padding: 12px;"
        )
        result_layout.addWidget(self._result_label)

        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setValue(0)
        self._confidence_bar.setFormat("置信度: %p%")
        self._confidence_bar.setVisible(False)
        result_layout.addWidget(self._confidence_bar)

        layout.addWidget(result_group)

        # ---- Export option ----
        export_layout = QHBoxLayout()
        self._export_check = QCheckBox("导出PDF报告")
        self._export_check.setChecked(False)
        export_layout.addWidget(self._export_check)
        self._export_btn = QPushButton("导出当前报告")
        self._export_btn.clicked.connect(self.export_requested.emit)
        export_layout.addWidget(self._export_btn)
        layout.addLayout(export_layout)

        # ---- Reset ----
        self._reset_btn = QPushButton("清除重置")
        self._reset_btn.setStyleSheet("QPushButton { color: #888; }")
        self._reset_btn.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self._reset_btn)

        layout.addStretch()

        self._result = None

    def set_result(self, result):
        self._result = result
        if result.pred_class_idx == 1:
            self._result_label.setText("● 恶性 (Malignant)")
            self._result_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; padding: 12px;"
                "color: #fff; background-color: #d32f2f; border-radius: 8px;"
            )
        else:
            self._result_label.setText("● 良性 (Benign)")
            self._result_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; padding: 12px;"
                "color: #fff; background-color: #2e7d32; border-radius: 8px;"
            )

        conf_pct = int(round(result.confidence * 100))
        self._confidence_bar.setValue(conf_pct)
        self._confidence_bar.setVisible(True)

        # Color the progress bar
        if result.pred_class_idx == 1:
            bar_color = "#d32f2f"
        else:
            bar_color = "#2e7d32"
        self._confidence_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {bar_color}; }}"
        )

    def clear_result(self):
        self._result = None
        self._result_label.setText("等待推理...")
        self._result_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; padding: 12px;"
        )
        self._confidence_bar.setValue(0)
        self._confidence_bar.setVisible(False)

    def reset(self):
        self._doctor_id.clear()
        self._doctor_name.clear()
        self._patient_id.clear()
        self._patient_name.clear()
        self._export_check.setChecked(False)
        self.clear_result()

    def get_doctor_info(self) -> tuple[str, str]:
        return (self._doctor_id.text().strip(), self._doctor_name.text().strip())

    def get_patient_info(self) -> tuple[str, str]:
        return (self._patient_id.text().strip(), self._patient_name.text().strip())

    def is_export_checked(self) -> bool:
        return self._export_check.isChecked()
