import cv2
import numpy as np
import random
import os
import sys
from collections import deque

class FaceSelector:
    def __init__(self):
        # 获取模型文件的正确路径
        cascade_path = self.get_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 检查模型是否加载成功
        if self.face_cascade.empty():
            fallback_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(fallback_path)
            
            if self.face_cascade.empty():
                sys.exit(1)
        
        # 状态变量
        self.random_mode = False
        self.selected_face = None
        self.static_frame = None
        self.static_faces = None
        
        # 用于平滑检测的队列
        self.face_queue = deque(maxlen=5)
        
        # 创建窗口
        self.window_name = "Face Selector"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 添加窗口关闭事件处理
        self.running = True
        
    def get_cascade_path(self):
        """获取模型文件的正确路径"""
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        possible_paths = [
            os.path.join(base_path, 'haarcascade_frontalface_default.xml'),
            os.path.join(base_path, 'cv2', 'data', 'haarcascade_frontalface_default.xml'),
            os.path.join(base_path, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml'),
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                return path
        
        return possible_paths[0]
    
    def detect_faces(self, frame):
        """检测人脸并返回位置"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def draw_faces(self, frame, faces):
        """在帧上绘制人脸框"""
        # 在随机模式下，只绘制选中的红框
        if self.random_mode and self.selected_face is not None and self.static_faces is not None:
            if self.selected_face < len(self.static_faces):
                x, y, w, h = self.static_faces[self.selected_face]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)  # 选中的为红色
        else:
            # 在正常模式下，绘制所有人脸框
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色
        
        return frame
    
    def draw_button(self, frame):
        """绘制按钮"""
        button_text = "Reset" if self.random_mode else "Random"
        button_color = (0, 0, 255) if self.random_mode else (0, 255, 0)
        
        # 绘制按钮背景
        cv2.rectangle(frame, (10, 10), (150, 50), button_color, -1)
        cv2.rectangle(frame, (10, 10), (150, 50), (255, 255, 255), 2)
        
        # 绘制按钮文字
        cv2.putText(frame, button_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def toggle_random_mode(self, frame, faces):
        """切换随机模式"""
        if not self.random_mode:
            # 进入随机模式
            self.random_mode = True
            if len(faces) > 0:
                self.selected_face = random.randint(0, len(faces)-1)
                self.static_frame = frame.copy()
                self.static_faces = faces.copy()  # 保存静态状态下的人脸位置
            else:
                self.selected_face = None
                self.static_frame = None
                self.static_faces = None
        else:
            # 退出随机模式
            self.random_mode = False
            self.selected_face = None
            self.static_frame = None
            self.static_faces = None
    
    def process_frame(self, frame):
        """处理每一帧"""
        # 检测人脸
        faces = self.detect_faces(frame)
        
        # 更新人脸队列
        if len(faces) > 0:
            self.face_queue.append(faces)
        
        # 使用队列中最新的人脸数据
        current_faces = faces
        if self.face_queue:
            current_faces = self.face_queue[-1]
        
        # 处理随机模式
        if self.random_mode:
            if self.static_frame is not None:
                # 使用静态帧，只绘制选中的红框
                display_frame = self.static_frame.copy()
                display_frame = self.draw_faces(display_frame, current_faces)
            else:
                display_frame = frame.copy()
        else:
            # 在实时帧上绘制所有人脸
            display_frame = self.draw_faces(frame.copy(), current_faces)
        
        # 绘制按钮
        display_frame = self.draw_button(display_frame)
        
        return display_frame
    
    def run(self):
        """运行主程序"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return
        
        # 获取摄像头分辨率并设置窗口大小
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            cv2.resizeWindow(self.window_name, width, height)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cv2.waitKey(1000)
                continue
            
            # 处理帧
            display_frame = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow(self.window_name, display_frame)
            
            # 处理鼠标点击
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # 检查是否点击了按钮区域
                    if 10 <= x <= 150 and 10 <= y <= 50:
                        faces = self.detect_faces(frame)
                        self.toggle_random_mode(frame, faces)
            
            cv2.setMouseCallback(self.window_name, mouse_callback)
            
            # 处理键盘输入和窗口关闭事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                self.running = False
                break
            
            # 检查窗口是否被关闭
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) <= 0:
                    self.running = False
                    break
            except:
                self.running = False
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    app = FaceSelector()
    app.run()
