import sys
import os
import random
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout, QProgressBar


# ======================================================
# 加载动画界面
# ======================================================
class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Random Selector")
        self.setFixedSize(300, 150)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("""
            background-color: #1E90FF;
            border-radius: 10px;
        """)

        # 创建布局
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        # 加载提示文字
        self.loading_label = QLabel("正在加载模型...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            color: white; 
            font-size: 18px; 
            font-weight: bold;
        """)
        layout.addWidget(self.loading_label)

        # 进度条 - 设置为不确定模式，显示加载动画
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 设置为0,0表示不确定模式
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid rgba(255, 255, 255, 100);
                border-radius: 5px;
                background-color: rgba(255, 255, 255, 50);
            }
            QProgressBar::chunk {
                background-color: white;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)


# ======================================================
# 模型加载线程
# ======================================================
class ModelLoader(QThread):
    loaded = Signal(object, str)  # 模型加载完成信号
    progress = Signal(str)  # 进度更新信号

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            # 模拟进度更新
            self.progress.emit("正在初始化模型...")

            # 启用 OpenCV 的优化并设置线程数，减少模型加载/运算开销
            try:
                cv2.setUseOptimized(True)
            except Exception:
                pass

            # 延迟加载检测器。
            detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 240),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )

            self.progress.emit("模型加载完成")
            self.loaded.emit(detector, "")
        except Exception as e:
            self.loaded.emit(None, str(e))


# ======================================================
# 主窗口
# ======================================================
class FaceRandomApp(QWidget):
    def __init__(self):
        super().__init__()

        # 先显示加载页面
        self.loading_screen = LoadingScreen()
        self.loading_screen.show()

        # 强制立即处理GUI事件，确保加载页面能立即显示
        QApplication.processEvents()

        # 延迟初始化主界面
        QTimer.singleShot(100, self.initialize_app)

    def initialize_app(self):
        """初始化应用程序"""
        # 状态
        self.state = "normal"
        self.selected_face_index = -1
        self.static_frame = None
        self.faces_snapshot = []
        self.faces = []
        self.detection_confidence = 0.6
        self.is_static_mode = False
        self.last_black_frame = None  # 缓存黑屏帧

        # 初始化摄像头 - 保持常开
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        # UI
        self.setup_ui()

        # 启动模型加载
        self.start_model_loading()

    def start_model_loading(self):
        """启动模型加载"""
        model_path = self.get_yunet_model_path()
        if model_path:
            self.loader = ModelLoader(model_path)
            self.loader.loaded.connect(self.on_model_loaded)
            self.loader.progress.connect(self.on_loading_progress)
            self.loader.start()
        else:
            # 如果没有模型，跳过模型加载
            self.loading_screen.close()
            self.show()
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

    def get_yunet_model_path(self):
        """获取模型路径"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "face_detection_yunet_2023mar.onnx")
        return model_path if os.path.exists(model_path) else None

    def on_loading_progress(self, message):
        """加载进度更新"""
        self.loading_screen.loading_label.setText(message)

    def on_model_loaded(self, detector, error):
        """模型加载完成回调"""
        if detector and not error:
            self.detector = detector
            # 更新置信度
            self.detector.setScoreThreshold(self.detection_confidence)
        else:
            print(f"模型加载失败: {error}")

        # 关闭加载界面并显示主窗口
        self.loading_screen.close()
        self.show()

        # 启动视频更新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ======================================================
    # UI
    # ======================================================
    def setup_ui(self):
        self.setWindowTitle("Face Random Selector")
        self.resize(1280, 720)

        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 0, self.width(), self.height())
        self.video_label.setStyleSheet("background: black;")

        # 随机按钮
        self.btn = QPushButton("随机", self)
        self.btn.setFixedSize(140, 55)
        self.btn.move(20, self.height() - 75)
        self.btn.clicked.connect(self.on_random_clicked)
        self.btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                font-size: 20px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(30, 30, 30, 200);
            }
            QPushButton:pressed {
                background-color: rgba(50, 50, 50, 220);
            }
        """)

        # 置信度滑条
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(30, 90)
        self.slider.setValue(int(self.detection_confidence * 100))
        self.slider.setFixedWidth(200)
        self.slider.move(120, 20)
        self.slider.valueChanged.connect(self.on_confidence_change)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(0, 0, 0, 120);
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: white;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }
        """)

        # 静止按钮 (圆形半透明)
        self.static_btn = QPushButton("静", self)
        self.static_btn.setFixedSize(55, 55)
        self.static_btn.move(self.btn.x() + self.btn.width() + 20, self.height() - 75)
        self.static_btn.clicked.connect(self.on_static_clicked)
        self.static_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 140);
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 27px;
            }
            QPushButton:hover {
                background-color: rgba(30, 30, 30, 180);
            }
            QPushButton:pressed {
                background-color: rgba(50, 50, 50, 200);
            }
        """)

    def resizeEvent(self, event):
        self.video_label.setGeometry(0, 0, self.width(), self.height())
        self.btn.move(20, self.height() - 75)
        if hasattr(self, 'static_btn'):
            self.static_btn.move(self.btn.x() + self.btn.width() + 20, self.height() - 75)
        # 重置缓存的黑屏帧
        self.last_black_frame = None
        super().resizeEvent(event)

    def on_confidence_change(self, value):
        self.detection_confidence = value / 100.0
        if hasattr(self, 'detector'):
            self.detector.setScoreThreshold(self.detection_confidence)

    def on_random_clicked(self):
        if self.state == "normal":
            self.state = "random"
            self.btn.setText("重置")

            if len(self.faces) > 0:
                self.selected_face_index = random.randint(0, len(self.faces) - 1)

                # 捕获静态帧
                if self.cap and self.cap.isOpened():
                    ret, img = self.cap.read()
                    if ret:
                        img = cv2.flip(img, 1)
                        img = self.resize_cover(img, self.video_label.width(), self.video_label.height())
                        self.static_frame = img.copy()
                        self.faces_snapshot = self.faces.copy()
        else:
            self.state = "normal"
            self.btn.setText("随机")
            self.selected_face_index = -1
            self.static_frame = None

    def on_static_clicked(self):
        """ 静止模式 - 暂停计算但不关闭摄像头，切换更快 """
        if not self.is_static_mode:
            # 进入静止模式
            self.is_static_mode = True
            self.static_btn.setText("恢复")
            
            # 显示静态黑屏图像，只显示一次
            self.show_static_black_screen()
        else:
            # 退出静止模式 - 立即恢复
            self.is_static_mode = False
            self.static_btn.setText("静")
            
            # 直接进入下一帧更新，无需重新初始化摄像头
    
    def show_static_black_screen(self):
        """显示静态黑屏，缓存结果避免重复计算"""
        h = self.video_label.height()
        w = self.video_label.width()
        
        # 缓存黑屏帧，避免重复创建
        if self.last_black_frame is None or self.last_black_frame.shape[:2] != (h, w):
            self.last_black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        self.display(self.last_black_frame)

    def resize_cover(self, img, target_w, target_h):
        h, w = img.shape[:2]
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        x_start = (new_w - target_w) // 2
        y_start = (new_h - target_h) // 2
        return resized[y_start:y_start + target_h, x_start:x_start + target_w]

    def draw_face_with_confidence(self, img, face, color, thickness=2):
        """绘制人脸框和置信度，增加对无效坐标的检查"""
        try:
            # 检查坐标是否有效（不是inf或nan）
            x, y, w, h = face[:4]
            
            # 检查是否为有限数值
            if (not np.isfinite(x) or not np.isfinite(y) or 
                not np.isfinite(w) or not np.isfinite(h)):
                return
            
            # 检查坐标是否在合理范围内
            img_h, img_w = img.shape[:2]
            if (x < -img_w or x > img_w * 2 or 
                y < -img_h or y > img_h * 2 or
                w <= 0 or h <= 0 or w > img_w * 2 or h > img_h * 2):
                return
            
            # 转换为整数
            x, y, w, h = map(int, (x, y, w, h))
            
            # 确保坐标在图像范围内
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))
            
            # 绘制人脸矩形框
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        except (ValueError, OverflowError) as e:
            # 如果转换失败，跳过这个检测框
            print(f"绘制人脸框时出错: {e}")
            return

    def update_frame(self):
        # 如果处于静止模式，只显示黑屏，不进行任何计算
        if self.is_static_mode:
            # 检查是否已经有缓存的黑屏帧
            if self.last_black_frame is None:
                self.show_static_black_screen()
            return

        # 检查摄像头是否可用
        if not self.cap or not self.cap.isOpened():
            return

        ret, img = self.cap.read()
        if not ret:
            # 如果读取失败，尝试重新获取一帧，最多尝试3次
            for _ in range(3):
                ret, img = self.cap.read()
                if ret:
                    break
            if not ret:
                return

        # 镜像翻转
        img = cv2.flip(img, 1)

        # resize_cover方法，保持视野大小
        img = self.resize_cover(img, self.video_label.width(), self.video_label.height())

        if self.state == "normal":
            if hasattr(self, 'detector') and self.detector is not None:
                # 设置检测器输入大小
                self.detector.setInputSize((img.shape[1], img.shape[0]))

                # 检测人脸
                try:
                    _, detected = self.detector.detect(img)
                    
                    # 过滤无效的检测结果
                    if detected is not None:
                        valid_faces = []
                        for face in detected:
                            # 检查是否包含无效值
                            if np.all(np.isfinite(face[:4])):
                                valid_faces.append(face)
                        self.faces = valid_faces
                    else:
                        self.faces = []
                        
                except Exception as e:
                    print(f"人脸检测出错: {e}")
                    self.faces = []

                # 绘制检测结果（正常模式绘制全部绿色框）
                for face in self.faces:
                    self.draw_face_with_confidence(img, face, (0, 255, 0), 2)
            else:
                # 如果没有模型，只显示原始图像
                pass
        else:  # random 模式
            if self.static_frame is not None:
                img = self.static_frame.copy()
                # 仅绘制被选中的红色框，其他不显示
                if self.selected_face_index != -1 and len(self.faces_snapshot) > self.selected_face_index:
                    face = self.faces_snapshot[self.selected_face_index]
                    self.draw_face_with_confidence(img, face, (0, 0, 255), 3)

        self.display(img)

    def display(self, img):
        # 图像显示
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        # 停止定时器
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # 释放摄像头
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        
        # 终止线程
        if hasattr(self, 'loader') and self.loader.isRunning():
            self.loader.quit()
            self.loader.wait()
        
        event.accept()


# ======================================================
def main():
    app = QApplication(sys.argv)
    window = FaceRandomApp()
    # 注意：这里不再调用 window.show()，因为 FaceRandomApp 内部会处理显示逻辑
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
