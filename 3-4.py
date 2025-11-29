import sys
import os
import random
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
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
# YuNet 模型路径
# ======================================================
def get_yunet_model_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "face_detection_yunet_2023mar.onnx")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YuNet 模型不存在: {model_path}")
    return model_path


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
        
        # 延迟初始化主界面，让加载页面先显示
        QTimer.singleShot(100, self.initialize_app)

    def initialize_app(self):
        """初始化应用程序"""
        # 状态
        self.state = "normal"
        self.selected_face_index = -1
        self.static_frame = None
        self.faces_snapshot = []

        # 参数
        self.detection_confidence = 0.6
        self.faces = []

        # 摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # YuNet 加载
        model_path = get_yunet_model_path()
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (640, 480),
            score_threshold=self.detection_confidence,
            nms_threshold=0.3,
            top_k=5000
        )

        # UI
        self.setup_ui()
        
        # 关闭加载界面并显示主窗口
        self.loading_screen.close()
        self.show()

        # Timer
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

        # 按钮 - 简约黑色半透明样式，无边框
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

        # 置信度滑条 - 简约半透明样式
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

        # 置信度标签 - 简约白色文字
        self.conf_label = QLabel("置信度", self)
        self.conf_label.move(20, 20)
        self.conf_label.setStyleSheet("""
            color: white; 
            font-size: 16px; 
            font-weight: bold;
            background: transparent;
        """)

    # ======================================================
    def resizeEvent(self, event):
        self.video_label.setGeometry(0, 0, self.width(), self.height())
        self.btn.move(20, self.height() - 75)
        super().resizeEvent(event)

    # ======================================================
    # 置信度
    # ======================================================
    def on_confidence_change(self, value):
        self.detection_confidence = value / 100.0
        self.detector.setScoreThreshold(self.detection_confidence)

    # ======================================================
    # 随机按钮
    # ======================================================
    def on_random_clicked(self):
        if self.state == "normal":
            self.state = "random"
            self.btn.setText("重置")

            if len(self.faces) > 0:
                self.selected_face_index = random.randint(0, len(self.faces) - 1)

                # 捕获静态帧
                ret, img = self.cap.read()
                img = cv2.flip(img, 1)
                img = self.resize_cover(img, self.video_label.width(), self.video_label.height())

                self.static_frame = img.copy()
                self.faces_snapshot = self.faces.copy()

        else:
            self.state = "normal"
            self.btn.setText("随机")
            self.selected_face_index = -1
            self.static_frame = None

    # ======================================================
    # 等比例覆盖（无黑边）
    # ======================================================
    def resize_cover(self, img, target_w, target_h):
        """等比例缩放 + 居中裁切 = 无黑边"""
        h, w = img.shape[:2]

        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # 居中裁切
        x_start = (new_w - target_w) // 2
        y_start = (new_h - target_h) // 2

        cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]
        return cropped

    # ======================================================
    # 绘制人脸框和置信度 - 优化样式
    # ======================================================
    def draw_face_with_confidence(self, img, face, color, thickness=2):
        """绘制人脸框和置信度"""
        x, y, w, h = map(int, face[:4])
        score = face[4]  # 置信度分数，范围0-1

        # 绘制人脸框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        # 绘制置信度文本
        confidence_text = f"{score:.2f}"
        text_y = max(y - 10, 20)  # 确保文本不超出图像顶部

        # 绘制半透明文本背景
        text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # 创建半透明覆盖层
        overlay = img.copy()
        cv2.rectangle(overlay, (x, text_y - text_size[1] - 5),
                      (x + text_size[0] + 10, text_y + 5), color, -1)
        
        # 应用半透明效果
        alpha = 0.7  # 透明度
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 绘制文本
        cv2.putText(img, confidence_text, (x + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ======================================================
    # 主循环
    # ======================================================
    def update_frame(self):
        ret, img = self.cap.read()
        if not ret:
            return

        img = cv2.flip(img, 1)
        img = self.resize_cover(img, self.video_label.width(), self.video_label.height())

        if self.state == "normal":
            # 设置 YuNet 输入大小
            self.detector.setInputSize((img.shape[1], img.shape[0]))

            # 检测
            _, detected = self.detector.detect(img)
            self.faces = detected if detected is not None else []

            # 画框和置信度
            for face in self.faces:
                self.draw_face_with_confidence(img, face, (0, 255, 0), 2)

        else:  # random 模式
            img = self.static_frame.copy()

            for i, face in enumerate(self.faces_snapshot):
                color = (0, 0, 255) if i == self.selected_face_index else (0, 255, 0)
                thickness = 3 if i == self.selected_face_index else 2
                self.draw_face_with_confidence(img, face, color, thickness)

        self.display(img)

    # ======================================================
    def display(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # ======================================================
    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


# ======================================================
def main():
    app = QApplication(sys.argv)
    window = FaceRandomApp()
    # 注意：这里不再调用 window.show()，因为 FaceRandomApp 内部会处理显示逻辑
    sys.exit(app.exec())


if __name__ == "__main__":
    main()