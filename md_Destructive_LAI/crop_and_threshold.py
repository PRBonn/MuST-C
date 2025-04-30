import cv2
import os
import numpy as np
import yaml
import click
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QSlider,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QAction,
    QMessageBox,
    QSizePolicy,
    QScrollArea,
    QGroupBox,
    QLineEdit,
    # QGuiApplication,
    QGraphicsView,
    QGraphicsScene,
)

from functools import partial

class CropAndThresholdTool(QMainWindow):
    def __init__(self, image_list, save_dest):
        super().__init__()

        self.image_list = image_list
        self.images = []
        self.current_image_index = 0
        self.overlay = None
        self.crop_rect = None
        self.config = None  # Store the configuration
        self.overlay_enabled = True  # Flag to enable/disable overlay
        self.save_dest = save_dest

        self.setWindowTitle("CropAndThresholdTool")

        self.initUI()

    def initUI(self):
        # Create the main widget and layout
        main_widget = QWidget(self)
        layout = QVBoxLayout(main_widget)

        # Create a QGraphicsView to display the image
        self.image_view = QGraphicsView(self)
        self.image_view.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_view)

        # Create a QGraphicsScene to hold the image
        self.image_scene = QGraphicsScene()
        self.image_view.setScene(self.image_scene)

        # Connect the mouse wheel event to the zoom_image method
        self.image_view.wheelEvent = self.zoom_image

        # Create a group box for sliders
        slider_group = QGroupBox("HSV Sliders")
        slider_layout = QVBoxLayout(slider_group)

        self.slider_dict = {}
        self.textbox_dict = {}

        # HSV sliders with labels
        self.slider_dict["lower_hue"] = self.create_slider("Lower Hue", 0, 179, 0)
        self.slider_dict["upper_hue"] = self.create_slider("Upper Hue", 0, 179, 179)
        self.slider_dict["lower_saturation"] = self.create_slider("Lower Saturation", 0, 255, 0)
        self.slider_dict["upper_saturation"] = self.create_slider("Upper Saturation", 0, 255, 255)
        self.slider_dict["lower_value"] = self.create_slider("Lower Value", 0, 255, 0)
        self.slider_dict["upper_value"] = self.create_slider("Upper Value", 0, 255, 255)

        for name_tag in self.slider_dict:
            slider_layout.addWidget(self.setup_slider_textbox_combo(name_tag))

        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)

        # Create a group box for cropping sliders
        crop_group = QGroupBox("Crop Sliders")
        crop_layout = QVBoxLayout(crop_group)

        # Crop sliders with labels
        self.crop_x_slider = self.create_slider("Crop X", 0, 100, 0)
        self.crop_y_slider = self.create_slider("Crop Y", 0, 100, 0)
        self.crop_width_slider = self.create_slider("Crop Width", 1, 100, 100)
        self.crop_height_slider = self.create_slider("Crop Height", 1, 100, 100)

        crop_layout.addWidget(self.crop_x_slider)
        crop_layout.addWidget(self.crop_y_slider)
        crop_layout.addWidget(self.crop_width_slider)
        crop_layout.addWidget(self.crop_height_slider)

        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)

        # Create a button to toggle overlay
        self.overlay_button = QPushButton("Toggle Overlay", self)
        self.overlay_button.setCheckable(True)
        self.overlay_button.setChecked(True)  # Overlay is initially enabled
        self.overlay_button.clicked.connect(self.toggle_overlay)
        layout.addWidget(self.overlay_button)

        # Load, Save, and Save Configuration buttons
        load_button = QPushButton("Load Random Image", self)
        load_button.clicked.connect(self.load_random_image)
        save_button = QPushButton("Save Config", self)
        save_button.clicked.connect(self.save_config)
        save_config_button = QPushButton("Save Configuration to 'calibration.yaml'", self)
        save_config_button.clicked.connect(self.save_to_calibration_yaml)

        layout.addWidget(load_button)
        layout.addWidget(save_button)
        layout.addWidget(save_config_button)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Create a menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Add actions to the File menu
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)

        self.load_config()  # Load configuration if it exists
        self.update_image()

    def create_textbox(self, name_str):
        corresp_slider = self.slider_dict[name_str]
        textbox = QLineEdit()
        textbox.setAlignment(Qt.AlignCenter)

        update_textbox_to_corresp_slider = partial(self.update_textbox, name_str)
        corresp_slider.valueChanged.connect(update_textbox_to_corresp_slider)

        textbox.setText(str(corresp_slider.value()))

        update_corresp_slider = partial(self.update_slider, corresp_slider)
        textbox.textChanged.connect(update_corresp_slider)

        return textbox

    def setup_slider_textbox_combo(self,name_tag):
        slider_group = QGroupBox(f"{name_tag}Slider")
        slider_layout = QHBoxLayout(slider_group)
        self.textbox_dict[name_tag] = self.create_textbox(name_tag)
        slider_layout.addWidget(self.slider_dict[name_tag], 5)
        slider_layout.addWidget(self.textbox_dict[name_tag], 1)
        slider_group.setLayout(slider_layout)
        return slider_group


    def zoom_image(self, event):
        # Zoom in or out based on mouse wheel event
        zoom_factor = 1.2  # You can adjust the zoom factor as needed

        if event.angleDelta().y() > 0:
            # Zoom in
            self.image_view.scale(zoom_factor, zoom_factor)
        else:
            # Zoom out
            self.image_view.scale(1 / zoom_factor, 1 / zoom_factor)

        event.accept()  # Accept the event

    def create_slider(self, label, min_val, max_val, init_val):
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(self.update_image)
        
        return slider

    def update_slider(self, slider, value):
        try:
            int_val = int(value)
            slider.setValue(int_val)
        except ValueError:
            print(f"slider cannot update to non int val: {value}")

    def update_textbox(self, namestr, value):
        # Update the text box with the slider's value
        self.textbox_dict[namestr].setText(str(value))

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        self.update_image()
        
    def keyPressEvent(self, event):
        # Toggle overlay when the "h" key is pressed
        if event.key() == Qt.Key_H:
            self.overlay_enabled = not self.overlay_enabled
            self.overlay_button.setChecked(self.overlay_enabled)
            self.update_image()

    def load_random_image(self):
        if len(self.image_list) > 0:
            self.current_image_index = np.random.randint(0, len(self.image_list))
            self.update_image()
            self.reset_view()
        else:
            QMessageBox.warning(self, "No Images", "No images found in the folder.")
            
    def reset_view(self):
        if self.image_scene.sceneRect().isValid():
            self.image_view.fitInView(self.image_scene.sceneRect(), Qt.KeepAspectRatio)


    def update_image(self):
        if not self.image_list:
            return

        img_filename = os.path.join(self.image_list[self.current_image_index])
        img = cv2.imread(img_filename)
        while img is None:
            print(f"Error reading {img_filename}")
            self.current_image_index = np.random.randint(0, len(self.image_list))
            img_filename = os.path.join(self.image_list[self.current_image_index])
            img = cv2.imread(img_filename)


        # Get HSV slider values
        lower_hue = self.slider_dict["lower_hue"].value()
        upper_hue = self.slider_dict["upper_hue"].value()
        lower_saturation = self.slider_dict["lower_saturation"].value()
        upper_saturation = self.slider_dict["upper_saturation"].value()
        lower_value = self.slider_dict["lower_value"].value()
        upper_value = self.slider_dict["upper_value"].value()

        # Create a mask based on HSV thresholds
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([lower_hue, lower_saturation, lower_value])
        upper_bound = np.array([upper_hue, upper_saturation, upper_value])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        
        # Update crop_rect based on slider values
        x = int(self.crop_x_slider.value() / 100 * img.shape[1])
        y = int(self.crop_y_slider.value() / 100 * img.shape[0])
        w = int(self.crop_width_slider.value() / 100 * img.shape[1])
        h = int(self.crop_height_slider.value() / 100 * img.shape[0])
        self.crop_rect = (x, y, w, h)

        # Crop image based on user-defined rectangle
        if self.crop_rect is not None:
            x, y, w, h = self.crop_rect
            img = img[y: y + h, x: x + w]
            mask = mask[y: y + h, x: x + w]

        # Overlay the mask onto the image if overlay is enabled
        if self.overlay_enabled:
            mask_rgb = np.repeat(np.expand_dims(mask, -1), 3, -1)
            mask_rgb *= np.array(([0, 0, 1]), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, mask_rgb, 0.5, 0.0)

        # Convert the OpenCV image to a QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Display the image in QGraphicsView
        pixmap = QPixmap.fromImage(q_img)
        self.image_scene.clear()
        self.image_scene.addPixmap(pixmap)
        self.image_view.setScene(self.image_scene)
        self.image_view.setSceneRect(0, 0, width, height)  # Set the scene rect



    def load_config(self):
        # Load configuration from 'calibration.yaml' if it exists
        if os.path.isfile("calibration.yaml"):
            with open("calibration.yaml", "r") as config_file:
                self.config = yaml.load(config_file, Loader=yaml.FullLoader)

                # Update slider values and crop rect based on loaded config
                if self.config:
                    self.slider_dict["lower_hue"].setValue(self.config.get("lower_hue", 0))
                    self.slider_dict["upper_hue"].setValue(self.config.get("upper_hue", 179))
                    self.slider_dict["lower_saturation"].setValue(self.config.get("lower_saturation", 0))
                    self.slider_dict["upper_saturation"].setValue(self.config.get("upper_saturation", 255))
                    self.slider_dict["lower_value"].setValue(self.config.get("lower_value", 0))
                    self.slider_dict["upper_value"].setValue(self.config.get("upper_value", 255))
                    self.crop_rect = self.config.get("crop_rect", None)

    def save_to_calibration_yaml(self):
        # Save configuration to 'calibration.yaml'
        config = {
            "lower_hue": self.slider_dict["lower_hue"].value(),
            "upper_hue": self.slider_dict["upper_hue"].value(),
            "lower_saturation": self.slider_dict["lower_saturation"].value(),
            "upper_saturation": self.slider_dict["upper_saturation"].value(),
            "lower_value": self.slider_dict["lower_value"].value(),
            "upper_value": self.slider_dict["upper_value"].value(),
            "crop_rect": self.crop_rect,
        }

        try:
            with open(os.path.join(self.save_dest, "calibration.yaml"), "w") as config_file:
                yaml.dump(config, config_file, default_flow_style=False)
                QMessageBox.information(self, "Saved", "Configuration saved to 'calibration.yaml'.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred while saving the configuration: {str(e)}")

    def save_config(self):
        self.config = {
            "lower_hue": self.slider_dict["lower_hue"].value(),
            "upper_hue": self.slider_dict["upper_hue"].value(),
            "lower_saturation": self.slider_dict["lower_saturation"].value(),
            "upper_saturation": self.slider_dict["upper_saturation"].value(),
            "lower_value": self.slider_dict["lower_value"].value(),
            "upper_value": self.slider_dict["upper_value"].value(),
            "crop_rect": self.crop_rect,
        }

        # Open a file dialog to choose the save location
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "YAML Files (*.yaml);;All Files (*)", options=options
        )

        if file_path:
            try:
                with open(file_path, "w") as config_file:
                    yaml.dump(self.config, config_file, default_flow_style=False)
                    QMessageBox.information(self, "Saved", "Configuration saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"An error occurred while saving the configuration: {str(e)}")

@click.command()
@click.option(
    "--data_folder",
    "-d",
    type=str,
    help="path to the data folder",
)
@click.option(
    "--output_folder",
    "-o",
    type=str,
    help="path to the output folder",
)
def main(data_folder, output_folder):
    app = QApplication(sys.argv)
    os.makedirs(os.path.join(output_folder), exist_ok=True)
        
    image_list = []
    for img in os.listdir(data_folder):
        img_fp = os.path.join(data_folder, img)
        img_np = cv2.imread(img_fp)
        if img_np is  None:
            print(f"skipping {img_fp} because it probably is not an image")
        else:
            image_list.append(img_fp)

    ex = CropAndThresholdTool(image_list, save_dest=output_folder)
    ex.show()
    app.exec_()

if __name__ == "__main__":
    main()
