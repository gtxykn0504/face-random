import sys
import random
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QProgressBar


class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("加载中...")
        self.setFixedSize(300, 150)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #1E90FF;")  # 蓝色背景
        
        # 创建布局
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)
        
        # 加载提示文字
        self.loading_label = QLabel("人脸模型加载中")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
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


class FaceRandomApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # 先显示加载页面
        self.loading_screen = LoadingScreen()
        self.loading_screen.show()
        
        # 延迟初始化主界面，让加载页面先显示
        QTimer.singleShot(100, self.initialize_app)

    def initialize_app(self):
        # 设置窗口标题
        self.setWindowTitle("Face Random Selector")
        
        # 状态：normal → random
        self.state = "normal"
        self.selected_face_index = -1
        self.static_frame = None  # 保存随机状态下的静态画面
        self.all_faces_snapshot = []  # 保存所有人脸信息快照

        # ---------------- 摄像头 ----------------
        self.cap = cv2.VideoCapture(0)
        
        # 设置摄像头分辨率
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
        # 设置窗口初始大小
        self.resize(1280, 720)
        
        # 使用长距离模型和更低的人脸检测阈值以提高检测率
        # 这里会加载模型，可能需要一些时间
        try:
            self.detector = FaceDetector(minDetectionCon=0.25, modelSelection=1)
        except Exception as e:
            print(f"初始化人脸检测器时出错: {e}")
            # 如果初始化失败，使用一个空的检测器
            self.detector = None

        # ---------------- 创建界面 ----------------
        self.setup_ui()

        self.faces = []
        
        # 模拟加载完成
        QTimer.singleShot(2000, self.finish_loading)

    def finish_loading(self):
        # 关闭加载页面
        self.loading_screen.close()
        # 显示主窗口
        self.show()
        
        # 启动定时器更新视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def setup_ui(self):
        # 视频标签 - 填充整个窗口
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background:#000;")
        self.video_label.setGeometry(0, 0, self.width(), self.height())

        # 按钮 - 叠加在视频上
        self.btn = QPushButton("随机", self)
        self.btn.setFixedSize(140, 55)
        self.btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 120);
                color: white;
                font-size: 20px;
                padding: 10px 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(30, 30, 30, 180);
            }
        """)
        self.btn.clicked.connect(self.button_clicked)
        
        # 初始按钮位置
        self.update_button_position()

    def resizeEvent(self, event):
        """窗口大小改变时自动调整视频和按钮位置"""
        self.video_label.setGeometry(0, 0, self.width(), self.height())
        self.update_button_position()
        super().resizeEvent(event)

    def update_button_position(self):
        """更新按钮位置"""
        btn_w = 140
        btn_h = 55
        margin = 20
        self.btn.setGeometry(
            margin,
            self.height() - btn_h - margin,
            btn_w,
            btn_h
        )

    # ---------------------------------------------------------
    # 逻辑按钮：随机 ↔ 重置
    # ---------------------------------------------------------
    def button_clicked(self):
        if self.state == "normal":
            # 切换到随机状态
            self.state = "random"
            self.btn.setText("重置")

            # 随机选择一个人脸
            if len(self.faces) > 0:
                self.selected_face_index = random.randint(0, len(self.faces) - 1)

                # 保存当前帧作为静态画面
                ret, img = self.cap.read()
                if ret:
                    img = cv2.flip(img, 1)
                    # 调整图像大小以适应窗口
                    img = cv2.resize(img, (self.video_label.width(), self.video_label.height()))
                    self.static_frame = img.copy()
                    self.all_faces_snapshot = self.faces.copy()
        else:  # self.state == "random"
            # 切换回初始状态
            self.state = "normal"
            self.selected_face_index = -1
            self.static_frame = None
            self.all_faces_snapshot = []
            self.btn.setText("随机")

    # ---------------------------------------------------------
    # 刷视频帧
    # ---------------------------------------------------------
    def update_frame(self):
        # ============================
        # 状态：normal（检测所有人脸并显示绿框）
        # ============================
        if self.state == "normal":
            ret, img = self.cap.read()
            if not ret:
                return

            img = cv2.flip(img, 1)
            
            # 调整图像大小以适应窗口
            img = cv2.resize(img, (self.video_label.width(), self.video_label.height()))

            # 检测人脸
            if self.detector is not None:
                img, faces = self.detector.findFaces(img, draw=False)
                self.faces = faces if faces else []
            else:
                self.faces = []

            # 绘制所有人脸为绿色
            for face in self.faces:
                x, y, w, h = face["bbox"]
                score = face["score"][0]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, f"{score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示图像
            self.display_image(img)

        # ============================
        # 状态：random（显示静态画面，选中的人脸为红色，其他为绿色）
        # ============================
        elif self.state == "random" and self.static_frame is not None:
            img = self.static_frame.copy()

            # 绘制所有人脸，选中的为红色，其他为绿色
            for i, face in enumerate(self.all_faces_snapshot):
                x, y, w, h = face["bbox"]
                score = face["score"][0]

                if i == self.selected_face_index:
                    color = (0, 0, 255)  # 红色
                else:
                    color = (0, 255, 0)  # 绿色

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, f"{score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 显示图像
            self.display_image(img)

    # ---------------------------------------------------------
    # 显示图像到 QLabel
    # ---------------------------------------------------------
    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # ---------------------------------------------------------
    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 直接创建主应用窗口，它内部会处理加载页面
    window = FaceRandomApp()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()