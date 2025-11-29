import sys
import random
import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                               QVBoxLayout, QProgressBar, QHBoxLayout, 
                               QSlider, QGroupBox, QDoubleSpinBox)


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
        
        # 强制立即处理GUI事件，确保加载页面能立即显示
        QApplication.processEvents()
        
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
        
        # 检测参数
        self.detection_confidence = 0.15  # 降低置信度阈值以提高检测率
        self.model_selection = 1  # 长距离模型
        self.control_panel_visible = False  # 控制面板显示状态
        self.multi_scale_enabled = True  # 启用多尺度检测

        # ---------------- 摄像头 ----------------
        self.cap = cv2.VideoCapture(0)
        
        # 设置更高分辨率以捕捉更多细节
        self.cap.set(3, 1920)  # 宽度
        self.cap.set(4, 1080)  # 高度
        
        # 设置窗口初始大小
        self.resize(1280, 720)
        
        # 更新加载页面文字，提示正在进行人脸模型初始化
        self.loading_screen.loading_label.setText("正在初始化人脸检测模型...")
        QApplication.processEvents()
        
        # 使用优化的参数
        try:
            self.detector = FaceDetector(
                minDetectionCon=self.detection_confidence, 
                modelSelection=self.model_selection
            )
            self.loading_screen.loading_label.setText("人脸模型加载完成!")
        except Exception as e:
            print(f"初始化人脸检测器时出错: {e}")
            # 如果初始化失败，使用一个空的检测器
            self.detector = None
            self.loading_screen.loading_label.setText("人脸模型加载失败，使用基础模式")

        # ---------------- 创建界面 ----------------
        self.setup_ui()

        self.faces = []
        
        # 模拟加载完成
        QTimer.singleShot(1500, self.finish_loading)

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

        # 随机按钮 - 叠加在视频上
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
        
        # 控制按钮 - 新增
        self.control_btn = QPushButton("控制", self)
        self.control_btn.setFixedSize(80, 40)
        self.control_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 120);
                color: white;
                font-size: 16px;
                padding: 5px 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: rgba(30, 30, 30, 180);
            }
        """)
        self.control_btn.clicked.connect(self.toggle_control_panel)
        
        # 控制面板 - 新增
        self.control_panel = QWidget(self)
        self.control_panel.setFixedSize(300, 180)  # 减小高度
        self.control_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 180);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 100);
            }
        """)
        self.control_panel.hide()  # 初始隐藏
        
        # 控制面板布局
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(10)
        
        # 置信度控制
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("置信度:")
        confidence_label.setStyleSheet("color: white; font-size: 14px;")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(5, 50)  # 调整范围以适应教室环境
        self.confidence_slider.setValue(15)  # 默认0.15
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        self.confidence_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(255, 255, 255, 100);
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
        """)
        
        self.confidence_value = QLabel("0.15")
        self.confidence_value.setStyleSheet("color: white; font-size: 14px; min-width: 40px;")
        
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_value)
        control_layout.addLayout(confidence_layout)
        
        # 多尺度检测选项
        multiscale_layout = QHBoxLayout()
        multiscale_label = QLabel("多尺度检测:")
        multiscale_label.setStyleSheet("color: white; font-size: 14px;")
        
        self.multiscale_check = QPushButton("启用")
        self.multiscale_check.setCheckable(True)
        self.multiscale_check.setChecked(True)
        self.multiscale_check.setFixedHeight(30)
        self.multiscale_check.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 50);
                color: white;
                border: 1px solid rgba(255, 255, 255, 100);
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: rgba(30, 144, 255, 200);
                border: 1px solid rgba(255, 255, 255, 200);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 80);
            }
        """)
        self.multiscale_check.clicked.connect(self.on_multiscale_changed)
        
        multiscale_layout.addWidget(multiscale_label)
        multiscale_layout.addWidget(self.multiscale_check)
        multiscale_layout.addStretch()
        control_layout.addLayout(multiscale_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("检测模型:")
        model_label.setStyleSheet("color: white; font-size: 14px;")
        
        self.model_short_btn = QPushButton("短距离")
        self.model_long_btn = QPushButton("长距离")
        
        for btn in [self.model_short_btn, self.model_long_btn]:
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 50);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 100);
                    border-radius: 4px;
                    font-size: 12px;
                }
                QPushButton:checked {
                    background-color: rgba(30, 144, 255, 200);
                    border: 1px solid rgba(255, 255, 255, 200);
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 80);
                }
            """)
        
        self.model_short_btn.clicked.connect(lambda: self.on_model_changed(0))
        self.model_long_btn.clicked.connect(lambda: self.on_model_changed(1))
        
        # 默认选择长距离模型
        self.model_long_btn.setChecked(True)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_short_btn)
        model_layout.addWidget(self.model_long_btn)
        model_layout.addStretch()
        control_layout.addLayout(model_layout)
        
        control_layout.addStretch()
        
        # 初始按钮位置
        self.update_button_position()

    def toggle_control_panel(self):
        """切换控制面板显示状态"""
        self.control_panel_visible = not self.control_panel_visible
        if self.control_panel_visible:
            self.control_panel.show()
        else:
            self.control_panel.hide()

    def on_confidence_changed(self, value):
        """置信度滑块值改变"""
        confidence = value / 100.0
        self.confidence_value.setText(f"{confidence:.2f}")
        self.detection_confidence = confidence
        if self.detector:
            self.detector.minDetectionCon = confidence

    def on_multiscale_changed(self):
        """多尺度检测选项改变"""
        self.multi_scale_enabled = self.multiscale_check.isChecked()

    def on_model_changed(self, model_selection):
        """模型选择改变"""
        self.model_selection = model_selection
        if model_selection == 0:
            self.model_short_btn.setChecked(True)
            self.model_long_btn.setChecked(False)
        else:
            self.model_short_btn.setChecked(False)
            self.model_long_btn.setChecked(True)
        
        # 重新初始化检测器
        try:
            self.detector = FaceDetector(
                minDetectionCon=self.detection_confidence,
                modelSelection=model_selection
            )
        except Exception as e:
            print(f"重新初始化检测器失败: {e}")

    def multi_scale_detection(self, img):
        """多尺度人脸检测，提高小尺寸人脸的检测率"""
        if not self.multi_scale_enabled or self.detector is None:
            return self.detector.findFaces(img, draw=False) if self.detector else (img, [])
        
        faces = []
        scales = [1.0, 0.75, 0.5]  # 多尺度因子
        
        for scale in scales:
            # 缩放图像
            if scale != 1.0:
                h, w = img.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                scaled_img = cv2.resize(img, (new_w, new_h))
            else:
                scaled_img = img.copy()
            
            # 在当前尺度下检测人脸
            _, scaled_faces = self.detector.findFaces(scaled_img, draw=False)
            
            # 将检测到的人脸坐标转换回原始尺度
            if scaled_faces:
                for face in scaled_faces:
                    bbox = face["bbox"]
                    # 调整边界框坐标
                    face["bbox"] = [
                        int(bbox[0] / scale),
                        int(bbox[1] / scale),
                        int(bbox[2] / scale),
                        int(bbox[3] / scale)
                    ]
                    # 只添加新检测到的人脸（避免重复）
                    if not self.is_duplicate_face(face, faces):
                        faces.append(face)
        
        return img, faces

    def is_duplicate_face(self, new_face, existing_faces, threshold=0.5):
        """检查是否重复检测到同一个人脸"""
        if not existing_faces:
            return False
            
        new_x, new_y, new_w, new_h = new_face["bbox"]
        new_center = (new_x + new_w/2, new_y + new_h/2)
        
        for face in existing_faces:
            x, y, w, h = face["bbox"]
            center = (x + w/2, y + h/2)
            
            # 计算两个边界框中心点的距离
            distance = np.sqrt((new_center[0] - center[0])**2 + (new_center[1] - center[1])**2)
            
            # 如果距离小于阈值，认为是同一个人脸
            if distance < max(new_w, new_h) * threshold:
                return True
                
        return False

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
        
        # 随机按钮位置（左下角）
        self.btn.setGeometry(
            margin,
            self.height() - btn_h - margin,
            btn_w,
            btn_h
        )
        
        # 控制按钮位置（左上角）
        control_btn_w = 80
        control_btn_h = 40
        self.control_btn.setGeometry(
            margin,
            margin,
            control_btn_w,
            control_btn_h
        )
        
        # 控制面板位置（控制按钮下方）
        panel_width = 300
        panel_height = 180
        self.control_panel.setGeometry(
            margin,
            margin + control_btn_h + 10,
            panel_width,
            panel_height
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

            # 检测人脸 - 使用多尺度检测
            if self.detector is not None:
                if self.multi_scale_enabled:
                    img, self.faces = self.multi_scale_detection(img)
                else:
                    img, self.faces = self.detector.findFaces(img, draw=False)
                self.faces = self.faces if self.faces else []
            else:
                self.faces = []

            # 绘制所有人脸为绿色
            for face in self.faces:
                x, y, w, h = face["bbox"]
                score = face["score"][0]
                # 根据置信度调整边框粗细
                thickness = max(1, int(3 * score))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
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
                    thickness = 4
                else:
                    color = (0, 255, 0)  # 绿色
                    thickness = 2

                cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
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