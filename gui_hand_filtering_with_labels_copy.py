import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QGridLayout, QFileDialog, QVBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QPalette, QPainter, QBrush, QColor
from PyQt5.QtCore import Qt, QSize


def get_image_files(input_dir, extensions={'.png', '.jpg', '.jpeg', '.bmp', '.gif'}):
    """
    Recursively collect image file paths from the given directory.
    """
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def corresponding_annotation(img_path, ann_input_dir, ann_ext='.txt'):
    """
    Get annotation file path for given image path, based on base name.
    """
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_file = os.path.join(ann_input_dir, base + ann_ext)
    return ann_file if os.path.isfile(ann_file) else None


class ClickableLabel(QLabel):
    """A QLabel that emits clicked signal and toggles selection."""
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.selected = False
        self.setFrameStyle(QLabel.Box | QLabel.Plain)
        self.setLineWidth(2)
        self.update_border()

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.update_border()

    def update_border(self):
        color = QColor('green') if self.selected else QColor('lightgray')
        self.setStyleSheet(f"border: 3px solid {color.name()};")


class ImageSelector(QWidget):
    def __init__(self, input_dir, output_dir, ann_input_dir, ann_output_dir, batch_size=9):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ann_input_dir = ann_input_dir
        self.ann_output_dir = ann_output_dir
        self.batch_size = batch_size

        self.images = get_image_files(self.input_dir)
        self.index = 0

        self.init_ui()
        self.load_batch()

    def init_ui(self):
        self.setWindowTitle("Image Selector with Annotations")
        self.layout = QVBoxLayout(self)

        # Grid for thumbnails
        self.grid = QGridLayout()
        self.layout.addLayout(self.grid)

        # Instructions
        inst = QLabel("Click to select images. Press SPACE to confirm and load next batch.")
        self.layout.addWidget(inst)

        self.setLayout(self.layout)
        self.resize(800, 600)

    def load_batch(self):
        # Clear existing widgets
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Load next batch of images
        batch = self.images[self.index:self.index + self.batch_size]
        if not batch:
            QMessageBox.information(self, "Done", "No more images to display.")
            self.close()
            return

        # Populate grid
        cols = int(self.batch_size**0.5)
        for i, img_path in enumerate(batch):
            row, col = divmod(i, cols)
            label = ClickableLabel(img_path)
            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                continue
            pixmap = pixmap.scaled(QSize(200, 200), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            label.setFixedSize(210, 210)
            self.grid.addWidget(label, row, col)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.save_selected()
            self.index += self.batch_size
            self.load_batch()
        else:
            super().keyPressEvent(event)

    def save_selected(self):
        """
        Copy selected images and their annotations (if exist) to output directories.
        """
        # Ensure output dirs exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ann_output_dir, exist_ok=True)

        # Iterate through grid widgets
        for i in range(self.grid.count()):
            widget = self.grid.itemAt(i).widget()
            if isinstance(widget, ClickableLabel) and widget.selected:
                # Copy image
                src = widget.path
                dst = os.path.join(self.output_dir, os.path.basename(src))
                shutil.copy2(src, dst)

                # Copy annotation if exists
                ann_src = corresponding_annotation(src, self.ann_input_dir)
                if ann_src:
                    ann_dst = os.path.join(self.ann_output_dir, os.path.basename(ann_src))
                    shutil.copy2(ann_src, ann_dst)


def main():
    app = QApplication(sys.argv)

    # Prompt for directories
    input_dir = QFileDialog.getExistingDirectory(None, "Select Input Image Directory")
    if not input_dir:
        return
    ann_input_dir = QFileDialog.getExistingDirectory(None, "Select Annotation Directory (.txt)")
    if not ann_input_dir:
        return
    output_dir = QFileDialog.getExistingDirectory(None, "Select Output Image Directory")
    if not output_dir:
        return
    ann_output_dir = QFileDialog.getExistingDirectory(None, "Select Output Annotation Directory")
    if not ann_output_dir:
        return

    selector = ImageSelector(
        input_dir, output_dir,
        ann_input_dir, ann_output_dir,
        batch_size=20
    )
    selector.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
