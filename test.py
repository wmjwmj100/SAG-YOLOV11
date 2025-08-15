import os
import sys

# 构建正确的repo_root路径，假设当前脚本在ultralytics目录同级
repo_root = os.path.abspath(os.path.join(__file__, ".."))
package_root = os.path.abspath(os.path.join(__file__, "..", "ultralytics-main"))
# 将repo_root添加到系统路径，确保优先导入本地版本
sys.path.insert(0, package_root)  # 使用insert而不是append，确保路径在最前面

# 验证ultralytics包的导入路径
try:
    import ultralytics

    actual_path = ultralytics.__path__[0]

    if not package_root in actual_path:
        raise ImportError(
            f"❌ 正在使用系统 ultralytics：{actual_path}\n"
            f"✅ 你需要使用本地版本：{repo_root}\n"
            f"请检查sys.path顺序或移除系统安装的ultralytics包"
        )
    print(f"✅ 使用本地ultralytics包: {actual_path}")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    pth_path = r"/data/ultralytics-main/ultralytics-main/runs/detect/train10/weights/best.pt"
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category