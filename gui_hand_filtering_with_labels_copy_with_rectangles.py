import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QFileDialog, QVBoxLayout, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QSize


def get_image_files(input_dir, extensions={'.png', '.jpg', '.jpeg', '.bmp', '.gif'}):
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def load_annotations(ann_path):
    """
    Load YOLO-format annotations: class x_center y_center width height (all relative).
    Returns list of tuples (cls, x_center, y_center, w, h).
    """
    ann = []
    try:
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_c, y_c, w, h = parts
                    ann.append((int(cls), float(x_c), float(y_c), float(w), float(h)))
    except FileNotFoundError:
        pass
    return ann


def draw_annotations(pixmap, annotations):
    """
    Draw semi-transparent bounding boxes on pixmap based on YOLO annotations.
    """
    painter = QPainter(pixmap)
    # Semi-transparent red pen for border
    pen = QPen(QColor(255, 0, 0, 50))  # RGBA, alpha=180
    pen.setWidth(2)
    painter.setPen(pen)
    # Semi-transparent red brush for fill
    brush = QBrush(QColor(255, 0, 0, 0))  # RGBA, alpha=80
    painter.setBrush(brush)

    img_w = pixmap.width()
    img_h = pixmap.height()

    for cls, x_c, y_c, w, h in annotations:
        # convert relative to absolute
        bw = w * img_w
        bh = h * img_h
        bx = x_c * img_w - bw / 2
        by = y_c * img_h - bh / 2
        painter.drawRect(int(bx), int(by), int(bw), int(bh))
    painter.end()
    return pixmap


class ClickableLabel(QLabel):
    def __init__(self, path, ann_path, parent=None):
        super().__init__(parent)
        self.path = path
        self.ann_path = ann_path
        self.selected = False
        self.setFrameStyle(QLabel.Box | QLabel.Plain)
        self.setLineWidth(2)
        self.update_border()

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.update_border()

    def update_border(self):
        color = 'green' if self.selected else 'lightgray'
        self.setStyleSheet(f"border: 3px solid {color};")

    def setPixmap(self, pixmap):
        # Draw annotations on pixmap before setting
        annotations = load_annotations(self.ann_path)
        if annotations:
            pixmap = pixmap.copy()
            pixmap = draw_annotations(pixmap, annotations)
        super().setPixmap(pixmap)


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
        self.setWindowTitle("Annotated Image Selector")
        self.layout = QVBoxLayout(self)
        self.grid = QGridLayout()
        self.layout.addLayout(self.grid)
        inst = QLabel("Click images to select. Press SPACE to save and go to next batch.")
        self.layout.addWidget(inst)
        self.resize(900, 700)

    def load_batch(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        batch = self.images[self.index:self.index + self.batch_size]
        if not batch:
            QMessageBox.information(self, "Done", "No more images.")
            self.close()
            return

        cols = int(self.batch_size**0.5)
        for i, img_path in enumerate(batch):
            row, col = divmod(i, cols)
            ann_path = os.path.join(
                self.ann_input_dir,
                os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            )
            label = ClickableLabel(img_path, ann_path)
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
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ann_output_dir, exist_ok=True)
        for i in range(self.grid.count()):
            widget = self.grid.itemAt(i).widget()
            if isinstance(widget, ClickableLabel) and widget.selected:
                # copy image
                dst_img = os.path.join(self.output_dir, os.path.basename(widget.path))
                shutil.copy2(widget.path, dst_img)
                # copy annotation if exists
                if os.path.isfile(widget.ann_path):
                    dst_ann = os.path.join(self.ann_output_dir, os.path.basename(widget.ann_path))
                    shutil.copy2(widget.ann_path, dst_ann)


def main():
    app = QApplication(sys.argv)
    inp = QFileDialog.getExistingDirectory(None, "Select Input Image Directory")
    if not inp: return
    ann_in = QFileDialog.getExistingDirectory(None, "Select Input Annotation Directory")
    if not ann_in: return
    out = QFileDialog.getExistingDirectory(None, "Select Output Image Directory")
    if not out: return
    ann_out = QFileDialog.getExistingDirectory(None, "Select Output Annotation Directory")
    if not ann_out: return

    selector = ImageSelector(inp, out, ann_in, ann_out, batch_size=25)
    selector.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
