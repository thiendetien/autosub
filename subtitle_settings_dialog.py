#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Subtitle Settings Dialog with Video Preview
- Load actual video frames
- Timeline slider to scrub
- Vertical/Horizontal orientation support
- Live preview of subtitle styling
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import json
import os
import numpy as np

class SubtitleSettingsDialog(QtWidgets.QDialog):
    """
    Advanced subtitle customization dialog with video preview
    """
    def __init__(self, video_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Subtitle Settings - Live Preview")
        self.setMinimumSize(1000, 700)
        
        self.video_path = video_path
        self.cap = None
        self.current_frame = None
        self.total_frames = 0
        self.fps = 25
        self.video_width = 540
        self.video_height = 960
        self.orientation = 'vertical'  # vertical or horizontal
        
        # Default settings
        self.settings = {
            'font_name': 'UTM-IMPACT',
            'font_size': 24,
            'font_color': '#FFFFFF',
            'border_width': 2,
            'border_color': '#000000',
            'bg_enabled': True,
            'bg_color': '#000000',
            'bg_opacity': 0.7,
            'bg_padding': 10,
            'position_v': 60,
            'position_h': 'center',
            'bold': True,
            'italic': False,
            'shadow_enabled': True,
            'shadow_offset': 2,
            'shadow_color': '#000000',
            'shadow_opacity': 0.8,
        }
        
        self.load_settings()
        self.init_ui()
        self.load_video()
        
    def init_ui(self):
        main_layout = QtWidgets.QHBoxLayout()
        
        # Left panel: Controls
        left_panel = self.create_controls_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Right panel: Video Preview
        right_panel = self.create_preview_panel()
        main_layout.addWidget(right_panel, 3)
        
        self.setLayout(main_layout)
        
    def create_controls_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        title = QtWidgets.QLabel("🎨 Subtitle Customization")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Scroll area for controls
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout()
        
        # === FONT SETTINGS ===
        font_group = QtWidgets.QGroupBox("📝 Font")
        font_layout = QtWidgets.QFormLayout()
        
        self.font_combo = QtWidgets.QFontComboBox()
        self.font_combo.setCurrentFont(QtGui.QFont(self.settings['font_name']))
        self.font_combo.currentFontChanged.connect(self.update_preview)
        font_layout.addRow("Font:", self.font_combo)
        
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(10, 100)
        self.size_spin.setValue(self.settings['font_size'])
        self.size_spin.valueChanged.connect(self.update_preview)
        font_layout.addRow("Size:", self.size_spin)
        
        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setStyleSheet(f"background: {self.settings['font_color']}; min-height: 25px;")
        self.color_btn.clicked.connect(lambda: self.choose_color('font_color', self.color_btn))
        font_layout.addRow("Color:", self.color_btn)
        
        style_layout = QtWidgets.QHBoxLayout()
        self.bold_check = QtWidgets.QCheckBox("Bold")
        self.bold_check.setChecked(self.settings['bold'])
        self.bold_check.stateChanged.connect(self.update_preview)
        style_layout.addWidget(self.bold_check)
        
        self.italic_check = QtWidgets.QCheckBox("Italic")
        self.italic_check.setChecked(self.settings['italic'])
        self.italic_check.stateChanged.connect(self.update_preview)
        style_layout.addWidget(self.italic_check)
        font_layout.addRow("Style:", style_layout)
        
        font_group.setLayout(font_layout)
        scroll_layout.addWidget(font_group)
        
        # === BORDER SETTINGS ===
        border_group = QtWidgets.QGroupBox("🔲 Border/Outline")
        border_layout = QtWidgets.QFormLayout()
        
        self.border_spin = QtWidgets.QSpinBox()
        self.border_spin.setRange(0, 10)
        self.border_spin.setValue(self.settings['border_width'])
        self.border_spin.valueChanged.connect(self.update_preview)
        border_layout.addRow("Width:", self.border_spin)
        
        self.border_color_btn = QtWidgets.QPushButton()
        self.border_color_btn.setStyleSheet(f"background: {self.settings['border_color']}; min-height: 25px;")
        self.border_color_btn.clicked.connect(lambda: self.choose_color('border_color', self.border_color_btn))
        border_layout.addRow("Color:", self.border_color_btn)
        
        border_group.setLayout(border_layout)
        scroll_layout.addWidget(border_group)
        
        # === BACKGROUND SETTINGS ===
        bg_group = QtWidgets.QGroupBox("🎭 Background Box")
        bg_layout = QtWidgets.QFormLayout()
        
        self.bg_check = QtWidgets.QCheckBox("Enable Background")
        self.bg_check.setChecked(self.settings['bg_enabled'])
        self.bg_check.stateChanged.connect(self.update_preview)
        bg_layout.addRow("", self.bg_check)
        
        self.bg_color_btn = QtWidgets.QPushButton()
        self.bg_color_btn.setStyleSheet(f"background: {self.settings['bg_color']}; min-height: 25px;")
        self.bg_color_btn.clicked.connect(lambda: self.choose_color('bg_color', self.bg_color_btn))
        bg_layout.addRow("Color:", self.bg_color_btn)
        
        self.bg_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_opacity_slider.setRange(0, 100)
        self.bg_opacity_slider.setValue(int(self.settings['bg_opacity'] * 100))
        self.bg_opacity_slider.valueChanged.connect(self.update_preview)
        self.bg_opacity_label = QtWidgets.QLabel(f"{int(self.settings['bg_opacity'] * 100)}%")
        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.addWidget(self.bg_opacity_slider)
        opacity_layout.addWidget(self.bg_opacity_label)
        bg_layout.addRow("Opacity:", opacity_layout)
        
        self.bg_padding_spin = QtWidgets.QSpinBox()
        self.bg_padding_spin.setRange(0, 150)
        self.bg_padding_spin.setValue(self.settings['bg_padding'])
        self.bg_padding_spin.valueChanged.connect(self.update_preview)
        bg_layout.addRow("Padding:", self.bg_padding_spin)
        
        bg_group.setLayout(bg_layout)
        scroll_layout.addWidget(bg_group)
        
        # === POSITION SETTINGS ===
        pos_group = QtWidgets.QGroupBox("📍 Position")
        pos_layout = QtWidgets.QFormLayout()
        
        self.pos_v_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pos_v_slider.setRange(0, 900)
        self.pos_v_slider.setValue(self.settings['position_v'])
        self.pos_v_slider.valueChanged.connect(self.update_preview)
        self.pos_v_label = QtWidgets.QLabel(f"{self.settings['position_v']}px")
        pos_v_layout = QtWidgets.QHBoxLayout()
        pos_v_layout.addWidget(self.pos_v_slider)
        pos_v_layout.addWidget(self.pos_v_label)
        pos_layout.addRow("Bottom:", pos_v_layout)
        
        self.pos_h_combo = QtWidgets.QComboBox()
        self.pos_h_combo.addItems(['left', 'center', 'right'])
        self.pos_h_combo.setCurrentText(self.settings['position_h'])
        self.pos_h_combo.currentTextChanged.connect(self.update_preview)
        pos_layout.addRow("Align:", self.pos_h_combo)
        
        pos_group.setLayout(pos_layout)
        scroll_layout.addWidget(pos_group)
        
        # === SHADOW SETTINGS ===
        shadow_group = QtWidgets.QGroupBox("🌑 Shadow")
        shadow_layout = QtWidgets.QFormLayout()
        
        self.shadow_check = QtWidgets.QCheckBox("Enable Shadow")
        self.shadow_check.setChecked(self.settings['shadow_enabled'])
        self.shadow_check.stateChanged.connect(self.update_preview)
        shadow_layout.addRow("", self.shadow_check)
        
        self.shadow_offset_spin = QtWidgets.QSpinBox()
        self.shadow_offset_spin.setRange(0, 10)
        self.shadow_offset_spin.setValue(self.settings['shadow_offset'])
        self.shadow_offset_spin.valueChanged.connect(self.update_preview)
        shadow_layout.addRow("Offset:", self.shadow_offset_spin)
        
        shadow_group.setLayout(shadow_layout)
        scroll_layout.addWidget(shadow_group)
        
        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.reset_btn = QtWidgets.QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_to_default)
        btn_layout.addWidget(self.reset_btn)
        
        self.save_btn = QtWidgets.QPushButton("💾 Save")
        self.save_btn.setStyleSheet("background: #4CAF50; color: white; font-weight: bold;")
        self.save_btn.clicked.connect(self.save_and_close)
        btn_layout.addWidget(self.save_btn)
        
        layout.addLayout(btn_layout)
        
        panel.setLayout(layout)
        panel.setMaximumWidth(350)
        return panel
        
    def create_preview_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Header with video info
        header_layout = QtWidgets.QHBoxLayout()
        
        title = QtWidgets.QLabel("👁️ Video Preview")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Orientation toggle
        header_layout.addWidget(QtWidgets.QLabel("Orientation:"))
        self.orient_combo = QtWidgets.QComboBox()
        self.orient_combo.addItems(['Vertical (9:16)', 'Horizontal (16:9)'])
        self.orient_combo.currentIndexChanged.connect(self.on_orientation_changed)
        header_layout.addWidget(self.orient_combo)
        
        layout.addLayout(header_layout)
        
        # Video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #1a1a1a; border: 2px solid #333;")
        self.video_label.setMinimumSize(400, 400)
        layout.addWidget(self.video_label, 1)
        
        # Timeline slider
        timeline_layout = QtWidgets.QHBoxLayout()
        
        self.time_label = QtWidgets.QLabel("00:00")
        timeline_layout.addWidget(self.time_label)
        
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setRange(0, 100)
        self.timeline_slider.valueChanged.connect(self.on_timeline_changed)
        timeline_layout.addWidget(self.timeline_slider, 1)
        
        self.duration_label = QtWidgets.QLabel("00:00")
        timeline_layout.addWidget(self.duration_label)
        
        layout.addLayout(timeline_layout)
        
        # Sample text
        text_layout = QtWidgets.QHBoxLayout()
        text_layout.addWidget(QtWidgets.QLabel("Sample Text:"))
        self.sample_text = QtWidgets.QLineEdit("Đây là phụ đề mẫu")
        self.sample_text.textChanged.connect(self.update_preview)
        text_layout.addWidget(self.sample_text)
        layout.addLayout(text_layout)
        
        panel.setLayout(layout)
        return panel
        
    def load_video(self):
        """Load video file"""
        if not self.video_path or not os.path.exists(self.video_path):
            # Create a placeholder frame
            self.current_frame = np.zeros((960, 540, 3), dtype=np.uint8)
            self.current_frame[:] = (40, 40, 40)
            self.video_width = 540
            self.video_height = 960
            self.update_preview()
            return
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Detect orientation
        if self.video_width > self.video_height:
            self.orientation = 'horizontal'
            self.orient_combo.setCurrentIndex(1)
        else:
            self.orientation = 'vertical'
            self.orient_combo.setCurrentIndex(0)
            
        # Update timeline
        self.timeline_slider.setRange(0, self.total_frames - 1)
        duration_sec = self.total_frames / self.fps
        self.duration_label.setText(self.format_time(duration_sec))
        
        # Read first frame
        self.seek_frame(0)
        
    def seek_frame(self, frame_num):
        """Seek to specific frame"""
        if self.cap is None:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_preview()
            
    def on_timeline_changed(self, value):
        """Handle timeline slider change"""
        self.seek_frame(value)
        current_sec = value / self.fps
        self.time_label.setText(self.format_time(current_sec))
        
    def on_orientation_changed(self, index):
        """Handle orientation change"""
        self.orientation = 'vertical' if index == 0 else 'horizontal'
        self.update_preview()
        
    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
        
    def update_preview(self):
        """Update video preview with subtitle overlay"""
        # Update settings from controls
        self.settings['font_name'] = self.font_combo.currentFont().family()
        self.settings['font_size'] = self.size_spin.value()
        self.settings['bold'] = self.bold_check.isChecked()
        self.settings['italic'] = self.italic_check.isChecked()
        self.settings['border_width'] = self.border_spin.value()
        self.settings['bg_enabled'] = self.bg_check.isChecked()
        self.settings['bg_opacity'] = self.bg_opacity_slider.value() / 100.0
        self.bg_opacity_label.setText(f"{self.bg_opacity_slider.value()}%")
        self.settings['bg_padding'] = self.bg_padding_spin.value()
        self.settings['position_v'] = self.pos_v_slider.value()
        self.pos_v_label.setText(f"{self.pos_v_slider.value()}px")
        self.settings['position_h'] = self.pos_h_combo.currentText()
        self.settings['shadow_enabled'] = self.shadow_check.isChecked()
        self.settings['shadow_offset'] = self.shadow_offset_spin.value()
        
        if self.current_frame is None:
            return
            
        # Create display image
        frame = self.current_frame.copy()
        h, w = frame.shape[:2]
        
        # Scale for display
        max_display = 500
        if self.orientation == 'vertical':
            scale = min(max_display / h, max_display / w)
        else:
            scale = min(max_display / w, max_display / h)
            
        display_w = int(w * scale)
        display_h = int(h * scale)
        
        # Create QImage from frame
        frame_resized = cv2.resize(frame, (display_w, display_h))
        
        # Convert to QPixmap
        qimg = QtGui.QImage(frame_resized.data, display_w, display_h, 
                           display_w * 3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        
        # Draw subtitle on pixmap
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        
        text = self.sample_text.text()
        if text:
            self.draw_subtitle(painter, text, display_w, display_h, scale)
        
        painter.end()
        
        self.video_label.setPixmap(pixmap)
        
    def draw_subtitle(self, painter, text, w, h, scale):
        """Draw subtitle with all styling options"""
        # Setup font
        font = QtGui.QFont(self.settings['font_name'])
        font.setPixelSize(int(self.settings['font_size'] * scale))
        font.setBold(self.settings['bold'])
        font.setItalic(self.settings['italic'])
        painter.setFont(font)
        
        fm = QtGui.QFontMetrics(font)
        text_width = fm.horizontalAdvance(text)
        text_height = fm.height()
        
        # Calculate position
        margin_v = int(self.settings['position_v'] * scale)
        y = h - margin_v
        
        if self.settings['position_h'] == 'center':
            x = (w - text_width) // 2
        elif self.settings['position_h'] == 'left':
            x = 20
        else:
            x = w - text_width - 20
            
        # Draw background
        if self.settings['bg_enabled']:
            padding = int(self.settings['bg_padding'] * scale)
            bg_color = QtGui.QColor(self.settings['bg_color'])
            bg_color.setAlphaF(self.settings['bg_opacity'])
            painter.fillRect(
                x - padding, 
                y - fm.ascent() - padding,
                text_width + padding * 2, 
                text_height + padding * 2,
                bg_color
            )
            
        # Draw shadow
        if self.settings['shadow_enabled']:
            offset = int(self.settings['shadow_offset'] * scale)
            shadow_color = QtGui.QColor(self.settings['shadow_color'])
            shadow_color.setAlphaF(self.settings['shadow_opacity'])
            
            path = QtGui.QPainterPath()
            path.addText(x + offset, y + offset, font, text)
            painter.fillPath(path, shadow_color)
            
        # Draw border/outline
        if self.settings['border_width'] > 0:
            path = QtGui.QPainterPath()
            path.addText(x, y, font, text)
            
            pen = QtGui.QPen(QtGui.QColor(self.settings['border_color']))
            pen.setWidth(int(self.settings['border_width'] * scale * 2))
            painter.strokePath(path, pen)
            
        # Draw main text
        path = QtGui.QPainterPath()
        path.addText(x, y, font, text)
        painter.fillPath(path, QtGui.QColor(self.settings['font_color']))
        
    def choose_color(self, key, btn):
        """Open color dialog and update setting"""
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.settings[key]))
        if color.isValid():
            self.settings[key] = color.name()
            btn.setStyleSheet(f"background: {color.name()}; min-height: 25px;")
            self.update_preview()
            
    def reset_to_default(self):
        """Reset all settings to default"""
        self.settings = {
            'font_name': 'UTM-IMPACT',
            'font_size': 24,
            'font_color': '#FFFFFF',
            'border_width': 2,
            'border_color': '#000000',
            'bg_enabled': True,
            'bg_color': '#000000',
            'bg_opacity': 0.7,
            'bg_padding': 10,
            'position_v': 60,
            'position_h': 'center',
            'bold': True,
            'italic': False,
            'shadow_enabled': True,
            'shadow_offset': 2,
            'shadow_color': '#000000',
            'shadow_opacity': 0.8,
        }
        self.refresh_controls()
        self.update_preview()
        
    def refresh_controls(self):
        """Refresh all controls to match current settings"""
        self.font_combo.setCurrentFont(QtGui.QFont(self.settings['font_name']))
        self.size_spin.setValue(self.settings['font_size'])
        self.bold_check.setChecked(self.settings['bold'])
        self.italic_check.setChecked(self.settings['italic'])
        self.border_spin.setValue(self.settings['border_width'])
        self.bg_check.setChecked(self.settings['bg_enabled'])
        self.bg_opacity_slider.setValue(int(self.settings['bg_opacity'] * 100))
        self.bg_padding_spin.setValue(self.settings['bg_padding'])
        self.pos_v_slider.setValue(self.settings['position_v'])
        self.pos_h_combo.setCurrentText(self.settings['position_h'])
        self.shadow_check.setChecked(self.settings['shadow_enabled'])
        self.shadow_offset_spin.setValue(self.settings['shadow_offset'])
        
        self.color_btn.setStyleSheet(f"background: {self.settings['font_color']}; min-height: 25px;")
        self.border_color_btn.setStyleSheet(f"background: {self.settings['border_color']}; min-height: 25px;")
        self.bg_color_btn.setStyleSheet(f"background: {self.settings['bg_color']}; min-height: 25px;")
        
    def load_settings(self):
        """Load settings from file"""
        settings_file = os.path.expanduser('~/.subtitle_settings.json')
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    saved = json.load(f)
                    self.settings.update(saved)
            except:
                pass
                
    def save_settings(self):
        """Save settings to file"""
        settings_file = os.path.expanduser('~/.subtitle_settings.json')
        with open(settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
            
    def save_and_close(self):
        """Save settings and close dialog"""
        self.save_settings()
        self.accept()
        
    def get_settings(self):
        """Return current settings"""
        return self.settings.copy()
        
    def closeEvent(self, event):
        """Clean up on close"""
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    """Test the dialog with a sample video"""
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # Try to find a sample video
    test_video = None
    test_paths = [
        "/home/thien/chum/autosub/8movie_10999/episode_001.mp4",
        "/home/thien/chum/autosub/8movie_10993/episode_101.mp4",
    ]
    for p in test_paths:
        if os.path.exists(p):
            test_video = p
            break
            
    dialog = SubtitleSettingsDialog(video_path=test_video)
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        print("Settings saved!")
        print(json.dumps(dialog.get_settings(), indent=2))
    sys.exit()


if __name__ == '__main__':
    main()
