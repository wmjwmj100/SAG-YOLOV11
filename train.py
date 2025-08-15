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

# 加载模型
#model = YOLO(os.path.join(repo_root, "runs/detect/train2/weights/best.pt"))
model = YOLO(os.path.join(repo_root, "ultralytics-main/ultralytics/cfg/models/v8/yolov8s.yaml"))
#从0开始？用上面注释掉的代码
#model = YOLO(os.path.join(repo_root, "ultralytics-main/ultralytics/cfg/models/11/yolo11s.yaml"))
# 训练模型 - 修正数据集路径
data_path = os.path.join(repo_root, "datasets/gc10-det yolo/data.yaml")
#data_path = os.path.join(repo_root, "datasets/ImageSets/sets.yaml")
results = model.train(
    data=data_path,
    epochs=400,
    batch=16,
    imgsz=600,
    optimizer="AdamW",
    lr0=0.001,

)