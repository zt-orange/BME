import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
MODEL = ROOT / "model" / "model.onnx"

print("=" * 50)
print("  超声图像智能诊断系统 - PyInstaller 打包")
print("=" * 50)

# 清理旧构建
for d in (DIST, BUILD):
    if d.exists():
        shutil.rmtree(d)
        print(f"  已清理: {d}")

if not MODEL.exists():
    print(f"  错误: 模型文件不存在: {MODEL}")
    sys.exit(1)

print("\n[1/2] 打包中...")

cmd = [
    "pyinstaller",
    "--name=超声图像智能诊断系统",
    "--onedir",
    "--noconsole",
    "--clean",
    "--add-data", "model/model.onnx;model",
    "--hidden-import=sklearn",
    "--hidden-import=PIL",
    "--hidden-import=PIL.Image",
    "--hidden-import=cv2",
    "--hidden-import=reportlab",
    "--hidden-import=onnxruntime",
    "--hidden-import=onnxruntime.capi",
    "gui_main.py",
]

result = subprocess.run(cmd, cwd=str(ROOT))
if result.returncode != 0:
    print("\n打包失败! 请检查错误信息。")
    input("按任意键退出...")
    sys.exit(1)

print("\n[2/2] 打包完成!")
print(f"输出目录: {DIST / '超声图像智能诊断系统'}")

# 写使用说明
readme = DIST / "超声图像智能诊断系统" / "使用说明.txt"
readme.write_text(
    "1. 双击 \"超声图像智能诊断系统.exe\" 启动程序\n"
    "2. 启动时自动加载 AI 模型（约10-30秒），请耐心等待\n"
    "3. 点击\"导入单张图像\"或\"导入文件夹\"加载超声图像\n"
    "4. 填写医生和患者信息后可导出 PDF 报告\n"
    "5. 点击\"清除重置\"可清空所有数据，方便换医生使用\n",
    encoding="utf-8",
)
print(f"  已生成: {readme}")

input("按任意键退出...")
