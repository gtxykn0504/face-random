# Face Random

## 1.软件说明
FaceRandomSelector 是一款人脸随机选择工具。软件采用opencv的人脸检测算法，能够快速准确地识别摄像头中的人脸，并提供高效的随机选择功能，帮助教师轻松实现课堂互动。

## 2.打包指令
2.4版本之前：

` nuitka --standalone --include-package=cv2 --enable-plugin=pyside6 --windows-console-mode=disable --windows-icon-from-ico=./text.ico --include-data-dir=F:\python310\lib\site-packages\cv2\data=cv2\data --output-dir=dist3c --remove-output 3-control.py `

2.4版本之后：

` nuitka --standalone --include-package=cv2 --enable-plugin=pyside6 --windows-console-mode=disable --windows-icon-from-ico=./text.ico --include-data-files=model/face_detection_yunet_2023mar.onnx=model/face_detection_yunet_2023mar.onnx --output-dir=dist34 --remove-output 3-5.py` 
## 3.版本说明

1. 1.py 2.py：非稳定版本
2. 3-1.py：存在UI不美观 窗口大小 摄像头清晰度等问题
3. 3-2.py：解决3-1.py的相关问题
4. 3-3.py：添加加载进度条
5. 3-test.py：由3-3.py改编 给出具体的人脸识别模型无法获取的相关报错
6. 3-control.py：由3-3.py改编 用户可以修改相关参数
7. 3-4.py：使用YuNet模型 加强对小人脸识别准确度
8. 3-5.py：优化加载性能
9. 3-6.py：优化加载性能 添加静止模式 删去置信度与选中绿框
10. 3-7.py：解决静止模式下 资源无效使用的问题
> 软件务必保存在纯英文路径中！