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

        threshold_widget = QWidget()
        threshold_layout = QVBoxLayout(threshold_widget)
        self.threshold_label = QLabel("Threshold: 70")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.process_image)
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)

        curve_widget = QWidget()
        curve_layout = QVBoxLayout(curve_widget)
        self.curve_label = QLabel("Curve Removal: 50%")
        self.curve_slider = QSlider(Qt.Horizontal)
        self.curve_slider.setMinimum(0)
        self.curve_slider.setMaximum(100)
        self.curve_slider.setValue(50)
        self.curve_slider.valueChanged.connect(self.process_image)
        curve_layout.addWidget(self.curve_label)
        curve_layout.addWidget(self.curve_slider)

        control_layout.addWidget(self.load_button)
        control_layout.addSpacing(20)
        control_layout.addWidget(threshold_widget)
        control_layout.addSpacing(20)
        control_layout.addWidget(curve_widget)
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

        self.threshold_label.setText(f"Threshold: {self.threshold_slider.value()}")
        self.curve_label.setText(f"Curve Removal: {self.curve_slider.value()}%")

        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Binarize image
        _, binary = cv2.threshold(gray, 255-self.threshold_slider.value(), 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Create mask for curve removal (initialize as black)
        mask = np.zeros_like(binary)
        
        # Calculate number of contours to remove
        num_contours_to_remove = int(len(contours) * (self.curve_slider.value() / 100.0))
        
        # Sort contours by area and remove the smallest ones (likely to be curves)
        contours = sorted(contours, key=cv2.contourArea)
        
        for contour in contours[:num_contours_to_remove]:
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Draw in white

        # Apply mask to binary image using bitwise_or
        result = cv2.bitwise_or(binary, mask)

        # Convert to RGB for display
        rgb_image = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

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
