工业检测项目介绍：

1.基本信息：
使用论文中的backbone和neck部分，使用yolo的predict部分（论文中写的用的yolo的检测模块）

2.快速开始：
IDE打开项目时会提示安装包,安装所需要的包，但是不要安装ultralytics包。
因为我借用这个包的预测模块，改动了其他模块，并把他放在./ultralytics-main/ultralytics下
所以，如果你用了官方的包，会报错的
训练只需要运行train.py即可
训练结果在./runs/detect/train20目录下（为什么是20?因为我之前测试了20次）
训练好的模型结果在./runs/detect/train20/weights下
model = YOLO(os.path.join(repo_root, "runs/detect/train20/weights/best.pt"))
model = YOLO(os.path.join(repo_root, "ultralytics-main/ultralytics/cfg/models/11/detection-s.yaml")
上面两种写法一个是用训练了一点的模型继续训练，一个是从0开始


3.如果你想知道模型架构可以继续看：
论文中DFEB SAG之类的模块在ultralytics-main/ultralytics/nn/modules下面
架构设计在ultralytics-main/ultralytics/cfg/models/11/detection-s.yaml里面
为什么是detection-s s?s代表small是一个参数少模型
如果想体验大的参数（前提是12g以上显存)->ultralytics-main/ultralytics/cfg/models/11/defect-l.yaml
架构设计如何与模块联系起来？
ultralytics-main/ultralytics/nn/tasks.py中的get_parse()函数connects everything

总之，你如果想改动架构，只需要看3个文件
ultralytics-main/ultralytics/nn/modules
ultralytics-main/ultralytics/cfg/models/11/detection-s.yaml
ultralytics-main/ultralytics/nn/tasks.py中的get_parse()函数

其他：
项目里有很多测试代码没删除，有一些没用的你可以删除
