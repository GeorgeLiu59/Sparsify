import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QSlider, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #cccccc; }")

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)

        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedHeight(40)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.load_button.clicked.connect(self.load_image)

        percentile_widget = QWidget()
        percentile_layout = QVBoxLayout(percentile_widget)
        self.percentile_label = QLabel("Intensity Percentile: 0%")
        self.percentile_label.setStyleSheet("font-weight: bold;")
        self.percentile_slider = QSlider(Qt.Horizontal)
        self.percentile_slider.setMinimum(0)
        self.percentile_slider.setMaximum(99)
        self.percentile_slider.setValue(0)
        self.percentile_slider.valueChanged.connect(self.process_image)
        percentile_layout.addWidget(self.percentile_label)
        percentile_layout.addSpacing(10)
        percentile_layout.addWidget(self.percentile_slider)

        control_layout.addWidget(self.load_button)
        control_layout.addSpacing(20)
        control_layout.addWidget(percentile_widget)
        control_layout.addStretch()

        main_layout.addWidget(self.image_label, stretch=1)
        main_layout.addWidget(control_panel)

        self.original_image = None
        self.processed_image = None

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.original_image = cv2.imread(file_name)
            self.process_image()

    def process_image(self):
        if self.original_image is None:
            return

        percentile_value = self.percentile_slider.value()
        self.percentile_label.setText(f"Intensity Percentile: {percentile_value}%")

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        white_mask = gray >= 245
        
        non_white_pixels = gray[~white_mask]
        
        if len(non_white_pixels) > 0:
            threshold_value = np.percentile(non_white_pixels, 100 - percentile_value)
        else:
            threshold_value = 245
        
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        binary[white_mask] = 255
        
        rgb_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
