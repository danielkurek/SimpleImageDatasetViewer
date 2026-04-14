import sys
import argparse
from io import BytesIO

from datasets import load_dataset
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QScrollArea, QFormLayout, 
    QSizePolicy, QSplitter, QTextEdit, QFrame
)

from PySide6.QtGui import (
    QPixmap, QImage, QIntValidator, QShortcut, QKeySequence, QTextOption
)
from PySide6.QtCore import Qt, QTimer


def pil_to_pixmap(pil_img):
    """Converts a PIL Image to a PySide6 QPixmap."""
    try:
        # Convert to RGBA to ensure consistent saving format and no color space crashes
        pil_img = pil_img.convert("RGBA")
        bio = BytesIO()
        pil_img.save(bio, format="PNG")
        img = QImage.fromData(bio.getvalue())
        return QPixmap.fromImage(img)
    except Exception as e:
        print(f"Error converting image: {e}")
        return QPixmap()


class ResizingImageLabel(QLabel):
    """A QLabel that scales its pixmap to fit its bounding box, preserving aspect ratio."""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)  # Constraint: minimum height of 200px
        self.setAlignment(Qt.AlignCenter)
        self.original_pixmap = None
        
        # Ignored size policy is critical so it doesn't force the layout
        # to expand to the original, unscaled image's size.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def update_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            # Constraint: fill the space and keep original aspect ratio
            scaled = self.original_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)


class WrappingTextEdit(QTextEdit):
    """A text editor acting as a Label that breaks long text anywhere if it lacks spaces, 
       while auto-resizing vertically to fit its wrapped contents."""
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Blend flawlessly with window layout background
        self.setStyleSheet("background: transparent;")
        
        # The key to breaking words anywhere or at boundaries
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Discard the document's inherent padding to align exactly like a standard QLabel
        self.document().setDocumentMargin(0)
        
        # Whenever width dynamically changes text wrapping, we recalculate and fix height
        self.document().documentLayout().documentSizeChanged.connect(self.adjust_height)

    def adjust_height(self, size):
        new_height = int(size.height()) + 2  # Added 2 pixels buffer margin
        # Ensure it doesn't recurse infinitely due to resize events 
        if self.maximumHeight() != new_height:
            self.setFixedHeight(new_height)

    def setText(self, text):
        self.setPlainText(str(text))


class DatasetViewer(QMainWindow):
    def __init__(self, dataset, image_col, text_cols):
        super().__init__()
        self.dataset = dataset
        self.image_col = image_col
        self.text_cols = text_cols
        
        self.current_split = list(dataset.keys())[0]
        self.current_index = 0

        self.setWindowTitle("Hugging Face Dataset Viewer")
        self.resize(800, 700)

        # Timer setup for Leading-Edge Debouncing
        self.load_timer = QTimer(self)
        self.load_timer.setSingleShot(True)
        self.load_timer.setInterval(350) # debounce interval in ms
        self.load_timer.timeout.connect(self.on_load_timeout)
        self.pending_load = False

        self.init_ui()
        self.schedule_load() # Initial load

        self.setFocusPolicy(Qt.StrongFocus)

        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev.activated.connect(self.on_prev)

        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next.activated.connect(self.on_next)

    # Force the main window to take focus so QComboBox doesn't swallow arrow keys on startup
    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()

    def init_ui(self):
        # Central widget and main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # -- Top Control Bar --
        controls_layout = QHBoxLayout()

        # Split Selection Dropdown
        self.split_combo = QComboBox()
        self.split_combo.addItems(list(self.dataset.keys()))
        self.split_combo.currentTextChanged.connect(self.on_split_changed)
        controls_layout.addWidget(QLabel("Split:"))
        controls_layout.addWidget(self.split_combo)

        controls_layout.addStretch()

        # Navigation: Prev
        self.btn_prev = QPushButton("◄ Previous")
        self.btn_prev.clicked.connect(self.on_prev)
        # Optional: prevent the button from taking focus when clicked
        self.btn_prev.setFocusPolicy(Qt.NoFocus) 
        controls_layout.addWidget(self.btn_prev)

        # Navigation: Index Input / Max
        self.index_input = QLineEdit()
        self.index_input.setFixedWidth(60)
        self.index_input.setAlignment(Qt.AlignRight)
        self.index_input.returnPressed.connect(self.on_index_jump)
        controls_layout.addWidget(self.index_input)

        self.max_index_label = QLabel()
        controls_layout.addWidget(self.max_index_label)

        # Navigation: Next
        self.btn_next = QPushButton("Next ►")
        self.btn_next.clicked.connect(self.on_next)
        # Optional: prevent the button from taking focus when clicked
        self.btn_next.setFocusPolicy(Qt.NoFocus)
        controls_layout.addWidget(self.btn_next)

        main_layout.addLayout(controls_layout)

        # -- Splitter for Image & Text Data --
        self.splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.splitter)

        # Top half: Image Viewer
        self.image_label = ResizingImageLabel()
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        self.splitter.addWidget(self.image_label)

        # Bottom half: Scroll Area for Transcriptions/Texts
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.splitter.addWidget(self.scroll_area)

        # Initial splitter ratio (e.g., Image gets roughly 400px, scroll area gets 300px)
        self.splitter.setSizes([400, 300])

        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(15)

        # Dynamic Form Layout for Transcriptions/Texts
        self.text_layout = QFormLayout()
        self.text_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        self.content_layout.addLayout(self.text_layout)
        
        self.content_layout.addStretch()
        self.text_labels = {}

    def update_index_ui(self):
        """Updates purely the UI navigational states to react instantly"""
        split_data = self.dataset[self.current_split]
        max_idx = len(split_data) - 1

        if max_idx < 0:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.index_input.setText("0")
            self.max_index_label.setText("/ 0")
            self.index_input.setValidator(QIntValidator(0, 0, self))
            return

        # UI State updates
        self.btn_prev.setEnabled(self.current_index > 0)
        self.btn_next.setEnabled(self.current_index < max_idx)
        
        # Display Current / Max Index (0-based)
        self.index_input.setText(str(self.current_index))
        self.max_index_label.setText(f"/ {max_idx}")
        self.index_input.setValidator(QIntValidator(0, max_idx, self))

    def schedule_load(self):
        """Smart debouncing: Execs immediately on normal clicks, throttles rapid holds."""
        self.update_index_ui()
        
        if not self.load_timer.isActive():
            # Leading Edge: The app has been idle. 
            # We defer the actual load via a 0-ms timer so the UI event loop 
            # has a chance to physically draw the index update on screen first.
            QTimer.singleShot(0, self.load_item_data)
            self.load_timer.start() # Start a cooldown window
            self.pending_load = False
        else:
            # Trailing Edge: User is holding the key. Delay loading until they stop.
            self.pending_load = True
            self.load_timer.start() # Resets the cooldown timer

    def on_load_timeout(self):
        """Called when the timer naturally finishes (user stopped spamming)"""
        if self.pending_load:
            self.load_item_data()
            self.pending_load = False

    def load_item_data(self):
        """Performs actual dataset reading and heavy image operations."""
        split_data = self.dataset[self.current_split]
        max_idx = len(split_data) - 1

        if max_idx < 0:
            self.image_label.clear()
            self.image_label.setText("Dataset split is empty.")
            return

        # Fetch row dictionary
        row = split_data[self.current_index]

        # 1. Update Image
        if self.image_col in row:
            pil_img = row[self.image_col]
            if hasattr(pil_img, "convert"):  # Check if it's actually a PIL Image
                pixmap = pil_to_pixmap(pil_img)
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.clear()
                self.image_label.setText(f"Value in '{self.image_col}' is not an image.")
        else:
            self.image_label.clear()
            self.image_label.setText(f"Image column '{self.image_col}' not found.")

        # 2. Update Text Columns
        cols_to_show = self.text_cols
        if not cols_to_show:
            # Auto-detect all non-image columns if not specified
            cols_to_show = [col for col in row.keys() if col != self.image_col]

        # Check if we need to rebuild the FormLayout (e.g. initial load or differing schemas)
        if list(self.text_labels.keys()) != cols_to_show:
            while self.text_layout.count():
                child = self.text_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.text_labels.clear()

            for col in cols_to_show:
                key_label = QLabel(f"<b>{col}:</b>")
                
                # Custom Auto-Resizing QTextEdit Widget 
                val_label = WrappingTextEdit()
                
                self.text_layout.addRow(key_label, val_label)
                self.text_labels[col] = val_label

        # Populate text values
        for col, label in self.text_labels.items():
            val = row.get(col, "N/A")
            label.setText(str(val))

    def on_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.schedule_load()

    def on_next(self):
        max_idx = len(self.dataset[self.current_split]) - 1
        if self.current_index < max_idx:
            self.current_index += 1
            self.schedule_load()

    def on_split_changed(self, split_name):
        self.current_split = split_name
        self.current_index = 0
        self.schedule_load()
        
        # Return focus to window so arrows work right after changing splits
        self.setFocus() 

    def on_index_jump(self):
        try:
            idx = int(self.index_input.text())
            max_idx = len(self.dataset[self.current_split]) - 1
            idx = max(0, min(idx, max_idx)) # Clamp to valid range
            self.current_index = idx
            self.schedule_load()
            
            # Return focus to window
            self.setFocus()
        except ValueError:
            self.index_input.setText(str(self.current_index))


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Dataset Image & Transcription Viewer")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path or name of the Hugging Face dataset (e.g., 'mnist' or 'user/my-dataset')")
    parser.add_argument("--config", type=str, default=None, 
                        help="Dataset configuration/subset name (if applicable)")
    parser.add_argument("--image-col", type=str, default="image", 
                        help="Name of the column containing the images (default: 'image')")
    parser.add_argument("--text-cols", type=str, nargs="+", default=None, 
                        help="Names of the text/int columns to display. If empty, all non-image columns are shown automatically.")
    
    args = parser.parse_args()

    print(f"Loading dataset '{args.dataset}'...")
    try:
        if args.config:
            ds = load_dataset(args.dataset, args.config)
        else:
            ds = load_dataset(args.dataset)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    
    viewer = DatasetViewer(ds, args.image_col, args.text_cols)
    viewer.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()