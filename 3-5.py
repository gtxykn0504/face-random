import sys
import os
import random
import cv2
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

            # 延迟加载检测器
            detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (640, 480),
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

        # 初始化摄像头 - 使用原代码的高分辨率设置
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # 提高图像质量设置
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

        # 置信度标签
        self.conf_label = QLabel("置信度", self)
        self.conf_label.move(20, 20)
        self.conf_label.setStyleSheet("""
            color: white; 
            font-size: 16px; 
            font-weight: bold;
            background: transparent;
        """)

    def resizeEvent(self, event):
        self.video_label.setGeometry(0, 0, self.width(), self.height())
        self.btn.move(20, self.height() - 75)
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
        """绘制人脸框和置信度"""
        x, y, w, h = map(int, face[:4])
        score = face[4]

        # 绘制人脸框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        # 绘制置信度文本
        confidence_text = f"{score:.2f}"
        text_y = max(y - 10, 20)

        # 创建半透明背景
        overlay = img.copy()
        text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, (x, text_y - text_size[1] - 5),
                      (x + text_size[0] + 10, text_y + 5), color, -1)

        # 应用透明度
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 绘制文本
        cv2.putText(img, confidence_text, (x + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, img = self.cap.read()
        if not ret:
            return

        # 使用原代码的镜像翻转
        img = cv2.flip(img, 1)

        # 使用原代码的resize_cover方法，保持视野大小
        img = self.resize_cover(img, self.video_label.width(), self.video_label.height())

        if self.state == "normal":
            if hasattr(self, 'detector'):
                # 设置检测器输入大小
                self.detector.setInputSize((img.shape[1], img.shape[0]))

                # 检测人脸
                _, detected = self.detector.detect(img)
                self.faces = detected if detected is not None else []

                # 绘制检测结果
                for face in self.faces:
                    self.draw_face_with_confidence(img, face, (0, 255, 0), 2)
            else:
                # 如果没有模型，只显示原始图像
                pass
        else:  # random 模式
            if self.static_frame is not None:
                img = self.static_frame.copy()
                for i, face in enumerate(self.faces_snapshot):
                    color = (0, 0, 255) if i == self.selected_face_index else (0, 255, 0)
                    thickness = 3 if i == self.selected_face_index else 2
                    self.draw_face_with_confidence(img, face, color, thickness)

        self.display(img)

    def display(self, img):
        # 保持原代码的图像显示方式
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
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