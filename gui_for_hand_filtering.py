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
        palette = self.palette()
        palette.setColor(QPalette.WindowText, color)
        self.setPalette(palette)
        self.setStyleSheet(f"border: 3px solid {color.name()};")


class ImageSelector(QWidget):
    def __init__(self, input_dir, output_dir, batch_size=9):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        self.images = get_image_files(self.input_dir)
        self.index = 0

        self.init_ui()
        self.load_batch()

    def init_ui(self):
        self.setWindowTitle("Image Selector")
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
        Copy selected images from current batch to output directory.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Iterate through grid widgets
        for i in range(self.grid.count()):
            widget = self.grid.itemAt(i).widget()
            if isinstance(widget, ClickableLabel) and widget.selected:
                src = widget.path
                dst = os.path.join(self.output_dir, os.path.basename(src))
                shutil.copy2(src, dst)


def main():
    app = QApplication(sys.argv)

    # Prompt for directories
    input_dir = QFileDialog.getExistingDirectory(None, "Select Input Directory")
    if not input_dir:
        return
    output_dir = QFileDialog.getExistingDirectory(None, "Select Output Directory")
    if not output_dir:
        return

    selector = ImageSelector(input_dir, output_dir, batch_size=20)
    selector.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
