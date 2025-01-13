import sys
import os
import numpy as np
import glob
import cv2
import logging
import argparse
import scipy.io as sio
from yacs.config import CfgNode
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import camera as cam


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()

        # Variables
        logging.basicConfig(level=logging.DEBUG)
        self.image_folder = cfg.image_folder
        self.label_folder = cfg.label_folder
        self.coeff_folder = cfg.coeff_folder
        self.file_list = []
        self.file_index = 0
        self.num_file = 0
        self.image = None
        self.lines = np.zeros((0, 2, 2), np.float32)
        self.line_index = len(self.lines) - 1
        self.label_endpoint = None
        self.capture_endpoint = None

        self.save_flag = True
        self.label_flag = False

        # Delete mode
        self.delete_mode = False
        self.selected_lines = set()  # set of indices of lines selected for deletion

        # New: for endpoint dragging
        self.is_dragging = False
        self.dragging_line_index = -1
        self.dragging_endpoint_index = -1

        self.type = cfg.type
        self.coeff_file = cfg.coeff_file
        self.camera = None
        self.decimal_precision = cfg.decimal_precision
        self.default_image_size = cfg.default_image_size

        # Parameters
        self.scale = cfg.init_scale
        self.scale_limit = cfg.scale_limit
        self.line_width = cfg.line_width
        self.point_radius = cfg.point_radius

        # If endpoint_select_thresh is not defined, default to 3x the radius
        if hasattr(cfg, "endpoint_select_thresh"):
            self.endpoint_select_thresh = cfg.endpoint_select_thresh
        else:
            self.endpoint_select_thresh = 3 * self.point_radius

        # For backward-compat, we still read point_select_thresh for snapping.
        # But if you want them the same, just unify them.
        self.point_select_thresh = (
            2 * self.point_radius if cfg.point_select_thresh is None else cfg.point_select_thresh
        )
        self.line_select_thresh = cfg.line_select_thresh
        self.point_align_thresh = cfg.point_align_thresh
        self.patterns = cfg.patterns

        # UI
        self.menuBar = QMenuBar()
        self.menu_File = self.menuBar.addMenu('File')
        self.menu_Edit = self.menuBar.addMenu('Edit')
        self.menu_Help = self.menuBar.addMenu('Help')
        self.menu_OpenDir = self.menu_File.addAction(QIcon('icon/open.png'), 'OpenDir')
        self.menu_File.addSeparator()
        self.menu_Save = self.menu_File.addAction(QIcon('icon/save.png'), 'Save')
        self.menu_Next = self.menu_Edit.addAction(QIcon('icon/next.png'), 'Next')
        self.menu_Edit.addSeparator()
        self.menu_Prev = self.menu_Edit.addAction(QIcon('icon/prev.png'), 'Prev')
        self.menu_Edit.addSeparator()
        self.menu_Create = self.menu_Edit.addAction(QIcon('icon/create.png'), 'Create')
        self.menu_Edit.addSeparator()
        self.menu_Delete = self.menu_Edit.addAction(QIcon('icon/delete.png'), 'Delete')

        # Toggle delete mode
        self.menu_DeleteMode = self.menu_Edit.addAction(QIcon('icon/delete.png'), 'Delete Mode')
        self.menu_Tutorial = self.menu_Help.addAction(QIcon('icon/tutorial.png'), 'Tutorial')

        self.button_ExportMat = QPushButton(text='Export .mat', icon=QIcon('icon/export.png'))
        
        self.button_OpenDir = QPushButton(text='Open Dir', icon=QIcon('icon/open.png'))
        self.button_Save = QPushButton(text='Save', icon=QIcon('icon/save.png'))
        self.button_Next = QPushButton(text='Next', icon=QIcon('icon/next.png'))
        self.button_Prev = QPushButton(text='Prev', icon=QIcon('icon/prev.png'))
        self.button_Create = QPushButton(text='Create', icon=QIcon('icon/create.png'))
        self.button_Delete = QPushButton(text='Delete', icon=QIcon('icon/delete.png'))
        self.button_DeleteMode = QPushButton(text='Delete Mode', icon=QIcon('icon/delete.png'))

        self.button_ZoomIn = QPushButton(text='Zoom In', icon=QIcon('icon/zoom-in.png'))
        self.button_ZoomOut = QPushButton(text='Zoom Out', icon=QIcon('icon/zoom-out.png'))
        self.text_zoom = QLineEdit(text='100%', alignment=Qt.AlignCenter)
        self.text_zoom.setValidator(QRegExpValidator(QRegExp(r'\d+%?')))
        self.text_zoom.setFixedSize(self.button_OpenDir.sizeHint())
        self.buttonLayout = QVBoxLayout()
        self.label_Image = QLabel()
        self.imageLayout = QVBoxLayout()

        self.text_Line = QLabel('Line list')
        self.text_File = QLabel('File list')
        self.list_Line = QListWidget()
        self.list_File = QListWidget()
        self.listLayout = QVBoxLayout()

        # Allow multi-selection in the line list
        self.list_Line.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.layout = QHBoxLayout()
        self.centralWidget = QWidget()

        self.InitUI(cfg)

    def InitUI(self, cfg):
        # Set UI
        self.buttonLayout.addWidget(self.button_ExportMat)
        self.buttonLayout.addWidget(self.button_OpenDir)
        self.buttonLayout.addWidget(self.button_Save)
        self.buttonLayout.addWidget(QFrame(frameShape=QFrame.HLine))
        self.buttonLayout.addWidget(self.button_Next)
        self.buttonLayout.addWidget(self.button_Prev)
        self.buttonLayout.addWidget(self.button_Create)
        self.buttonLayout.addWidget(self.button_Delete)
        self.buttonLayout.addWidget(self.button_DeleteMode)
        self.buttonLayout.addWidget(QFrame(frameShape=QFrame.HLine))
        self.buttonLayout.addWidget(self.text_zoom)
        self.buttonLayout.addWidget(self.button_ZoomIn)
        self.buttonLayout.addWidget(self.button_ZoomOut)
        self.imageLayout.addWidget(self.label_Image, Qt.AlignCenter)
        self.listLayout.addWidget(self.text_Line)
        self.listLayout.addWidget(self.list_Line)
        self.listLayout.addWidget(self.text_File)
        self.listLayout.addWidget(self.list_File)
        self.layout.addLayout(self.buttonLayout)
        self.layout.addStretch(1)
        self.layout.addLayout(self.imageLayout)
        self.layout.addStretch(1)
        self.layout.addLayout(self.listLayout)
        self.centralWidget.setLayout(self.layout)
        self.setMenuBar(self.menuBar)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('Labelline')
        self.resize(cfg.window_size[0], cfg.window_size[1])

        # Register callbacks
        self.button_ExportMat.clicked.connect(self.ExportMat_Callback)
        self.button_OpenDir.clicked.connect(self.OpenDir_Callback)
        self.button_Save.clicked.connect(self.Save_Callback)
        self.button_Next.clicked.connect(self.Next_Callback)
        self.button_Prev.clicked.connect(self.Prev_Callback)
        self.button_Create.clicked.connect(self.Create_Callback)
        self.button_Delete.clicked.connect(self.Delete_Callback)
        self.button_DeleteMode.clicked.connect(self.ToggleDeleteMode_Callback)

        self.button_ZoomIn.clicked.connect(self.ZoomIn_Callback)
        self.button_ZoomOut.clicked.connect(self.ZoomOut_Callback)

        self.menu_OpenDir.triggered.connect(self.OpenDir_Callback)
        self.menu_Save.triggered.connect(self.Save_Callback)
        self.menu_Next.triggered.connect(self.Next_Callback)
        self.menu_Prev.triggered.connect(self.Prev_Callback)
        self.menu_Create.triggered.connect(self.Create_Callback)
        self.menu_Delete.triggered.connect(self.Delete_Callback)
        self.menu_DeleteMode.triggered.connect(self.ToggleDeleteMode_Callback)
        self.menu_Tutorial.triggered.connect(self.Tutorial_Callback)

        self.menu_OpenDir.setShortcut(cfg.menu_OpenDir_shortcut)
        self.menu_Save.setShortcut(cfg.menu_Save_shortcut)
        self.menu_Next.setShortcut(cfg.menu_Next_shortcut)
        self.menu_Prev.setShortcut(cfg.menu_Prev_shortcut)
        self.menu_Create.setShortcut(cfg.menu_Create_shortcut)
        self.menu_Delete.setShortcut(cfg.menu_Delete_shortcut)
        self.menu_DeleteMode.setShortcut("Ctrl+Shift+D")
        self.menu_Tutorial.setShortcut(cfg.menu_Tutorial_shortcut)

        self.list_Line.clicked.connect(self.ListLine_Callback)
        self.list_File.clicked.connect(self.ListFile_Callback)
        self.text_zoom.editingFinished.connect(self.TextZoom_Callback)

        # Override mouse events on the label
        self.label_Image.mousePressEvent = self.mousePress_Callback
        self.label_Image.mouseReleaseEvent = self.mouseRelease_Callback
        self.label_Image.mouseMoveEvent = self.mouseMove_Callback
        self.label_Image.setMouseTracking(True)

        self.reset()
        self.button_OpenDir.setEnabled(True)
        self.menu_OpenDir.setEnabled(True)

    def reset(self):
        self.list_Line.setEnabled(False)
        self.list_File.setEnabled(False)
        self.text_zoom.setEnabled(False)

        self.button_OpenDir.setEnabled(False)
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(False)
        self.button_Prev.setEnabled(False)
        self.button_Create.setEnabled(False)
        self.button_Delete.setEnabled(False)
        self.button_DeleteMode.setEnabled(False)
        self.button_ZoomIn.setEnabled(False)
        self.button_ZoomOut.setEnabled(False)

        self.menu_OpenDir.setEnabled(False)
        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(False)
        self.menu_Prev.setEnabled(False)
        self.menu_Create.setEnabled(False)
        self.menu_Delete.setEnabled(False)
        self.menu_DeleteMode.setEnabled(False)

        self.label_Image.setEnabled(False)

    def image_update(self):
        image = self.image.copy()
        if len(self.lines) > 0:
            try:
                lines = self.camera.truncate_line(self.lines)
                lines = self.camera.remove_line(lines, 1.0)
            except Exception:
                lines = self.lines

            # Draw all lines in green
            self.camera.insert_line(image, lines, color=[0, 255, 0], thickness=self.line_width)

            # Draw all endpoints in red
            pts = self.lines.reshape(-1, 2)
            for pt in pts:
                pt = np.int32(np.round(pt))
                cv2.circle(image, tuple(pt), radius=self.point_radius, color=[0, 0, 255], thickness=-1)

            # The currently "active" line is drawn in blue
            if self.line_index >= 0 and self.label_endpoint is None:
                try:
                    single_line = self.camera.truncate_line(
                        self.lines[self.line_index : self.line_index + 1]
                    )
                except Exception:
                    single_line = self.lines[self.line_index : self.line_index + 1]
                self.camera.insert_line(image, single_line, color=[255, 0, 0], thickness=self.line_width)

            # Highlight lines that are selected for deletion (in magenta)
            for idx in self.selected_lines:
                try:
                    selected_line = self.camera.truncate_line(self.lines[idx : idx + 1])
                except Exception:
                    selected_line = self.lines[idx : idx + 1]
                self.camera.insert_line(image, selected_line, color=[255, 0, 255], thickness=self.line_width)

        # If we're in the middle of creating a line, draw its endpoint in blue
        if self.label_endpoint is not None:
            pt = np.int32(np.round(self.label_endpoint))
            cv2.circle(image, tuple(pt), radius=self.point_radius, color=[255, 0, 0], thickness=-1)

        # If there's a "capture endpoint" for snapping, draw it in pink
        if self.capture_endpoint is not None:
            pt = np.int32(np.round(self.capture_endpoint))
            cv2.circle(image, tuple(pt), radius=self.point_select_thresh, color=[255, 0, 255], thickness=-1)

        new_size = (
            int(round(image.shape[1] * self.scale)),
            int(round(image.shape[0] * self.scale)),
        )
        image = cv2.resize(image, new_size, cv2.INTER_CUBIC)
        image = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.shape[1] * 3,
            QImage.Format.Format_BGR888,
        )
        pixmap = QPixmap(image).scaled(new_size[0], new_size[1])
        self.label_Image.setPixmap(pixmap)

    def set_camera(self):
        if self.type == 0:
            self.camera = cam.Pinhole()
        elif self.type == 1:
            self.camera = cam.Fisheye()
        else:
            self.camera = cam.Spherical((self.image.shape[1], self.image.shape[0]))

        if os.path.isfile(self.coeff_file):
            self.camera.load_coeff(self.coeff_file)
        else:
            image_file = self.file_list[self.file_index]
            filename = os.path.basename(image_file)
            filename = os.path.splitext(filename)[0] + '.yaml'
            coeff_file = os.path.join(self.data_path, self.coeff_folder, filename)
            if os.path.isfile(coeff_file):
                self.camera.load_coeff(coeff_file)
            elif self.type == 1:
                logging.error(f'{coeff_file} does not exist!')
                exit()

    def data_update(self):
        self.label_flag = False
        self.label_endpoint = None
        self.capture_endpoint = None
        self.selected_lines.clear()

        image_file = self.file_list[self.file_index]
        filename = os.path.basename(image_file)
        filename = os.path.splitext(filename)[0] + '.mat'
        line_file = os.path.join(self.data_path, self.label_folder, filename)

        self.image = cv2.imread(image_file)
        if self.image is None:
            logging.error(f'{image_file} does not exist!')
            exit()

        self.lines = np.zeros((0, 2, 2), np.float32)
        if os.path.isfile(line_file):
            lines = sio.loadmat(line_file)['lines']
            if len(lines):
                self.lines = lines
        self.line_index = len(self.lines) - 1
        self.set_camera()

    def widget_update(self):
        self.text_File.setText(f'File list: {self.file_index + 1} / {self.num_file}')
        self.list_File.clear()
        self.list_File.addItems(self.file_list)
        self.list_File.setCurrentRow(self.file_index)
        self.list_File.setEnabled(self.save_flag)

        self.button_OpenDir.setEnabled(True)
        self.button_Save.setEnabled(not self.save_flag)
        self.button_Next.setEnabled(self.save_flag and self.file_index < self.num_file - 1)
        self.button_Prev.setEnabled(self.save_flag and self.file_index > 0)
        self.button_Create.setEnabled(True)
        self.button_Delete.setEnabled(len(self.lines) > 0)
        self.button_DeleteMode.setEnabled(True)

        self.menu_OpenDir.setEnabled(True)
        self.menu_Save.setEnabled(not self.save_flag)
        self.menu_Next.setEnabled(self.save_flag and self.file_index < self.num_file - 1)
        self.menu_Prev.setEnabled(self.save_flag and self.file_index > 0)
        self.menu_Create.setEnabled(True)
        self.menu_Delete.setEnabled(len(self.lines) > 0)
        self.menu_DeleteMode.setEnabled(True)

        self.label_Image.setEnabled(True)

    def line_update(self):
        self.label_endpoint = None
        self.text_Line.setText(f'Line list: {self.line_index + 1} / {len(self.lines)}')
        self.list_Line.clear()

        # Rebuild the list items
        for i, line in enumerate(self.lines.reshape(-1, 4)):
            item_text = f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]'
            item = QListWidgetItem(item_text)
            self.list_Line.addItem(item)
            # Optionally store the line index in item data
            item.setData(Qt.UserRole, i)

        if self.line_index >= 0:
            self.list_Line.setCurrentRow(self.line_index)

        self.list_Line.setEnabled(len(self.lines) > 0)
        self.button_Delete.setEnabled(len(self.lines) > 0)

    def zoom_update(self):
        self.text_zoom.setText(f'{int(round(self.scale * 100))}%')
        self.text_zoom.setEnabled(True)
        self.button_ZoomIn.setEnabled(self.scale < self.scale_limit[1])
        self.button_ZoomOut.setEnabled(self.scale > self.scale_limit[0])

    def OpenDir_Callback(self, data_path=None):
        self.Save_Callback()
        if not data_path:
            data_path = QFileDialog.getExistingDirectory()
        image_path = os.path.join(data_path, self.image_folder)
        if data_path == '' or not os.path.isdir(image_path):
            return

        file_list = []
        for pattern in self.patterns:
            file_list += sorted(glob.glob(os.path.join(image_path, pattern)))
        num_file = len(file_list)
        if num_file == 0:
            return
        label_path = os.path.join(data_path, self.label_folder)
        os.makedirs(label_path, exist_ok=True)

        self.data_path = data_path
        self.file_list = file_list
        self.file_index = 0
        self.num_file = num_file

        self.data_update()
        if self.default_image_size and self.scale == 0:
            scale = min(
                self.default_image_size[0] / self.image.shape[1],
                self.default_image_size[1] / self.image.shape[0],
            )
            self.scale = float(np.clip(scale, self.scale_limit[0], self.scale_limit[1]))
            self.text_zoom.setText(f'{int(round(self.scale * 100))}%')

        self.widget_update()
        self.line_update()
        self.zoom_update()
        self.image_update()

    def Save_Callback(self):
        if self.save_flag:
            return
        self.save_flag = True
        image_file = self.file_list[self.file_index]
        filename = os.path.basename(image_file)
        filename = os.path.splitext(filename)[0] + '.mat'
        line_file = os.path.join(self.data_path, self.label_folder, filename)
        if self.camera.coeff:
            K, D = self.camera.coeff['K'], self.camera.coeff['D']
            sio.savemat(line_file, {'lines': self.lines, 'K': K, 'D': D})
        else:
            sio.savemat(line_file, {'lines': self.lines})

        self.widget_update()

    def Next_Callback(self):
        self.file_index += 1
        self.data_update()
        self.widget_update()
        self.line_update()
        self.image_update()

    def Prev_Callback(self):
        self.file_index -= 1
        self.data_update()
        self.widget_update()
        self.line_update()
        self.image_update()

    def ListFile_Callback(self):
        self.file_index = self.list_File.currentRow()
        self.data_update()
        self.widget_update()
        self.line_update()
        self.image_update()

    def ListLine_Callback(self):
        # If multiple lines are selected, line_index will refer to the last clicked
        selected_items = self.list_Line.selectedItems()
        if not selected_items:
            return
        last_item = selected_items[-1]
        idx = last_item.data(Qt.UserRole)
        if idx is not None:
            self.line_index = idx
        else:
            self.line_index = self.list_Line.currentRow()
        self.line_update()
        self.image_update()

    def Create_Callback(self):
        self.label_flag = True
        self.setCursor(Qt.CrossCursor)
        self.reset()
        self.label_Image.setEnabled(True)

    def Delete_Callback(self):
        """
        If in delete_mode, remove all lines that are selected in the UI or 
        selected_lines in the image. Otherwise, remove the single line at line_index.
        """
        # Collect all selected lines from the list
        selected_items = self.list_Line.selectedItems()
        selected_from_list = set()
        for it in selected_items:
            idx = it.data(Qt.UserRole)
            if idx is not None:
                selected_from_list.add(idx)

        # Combine with selected lines from the image
        lines_to_delete = selected_from_list.union(self.selected_lines)

        if len(lines_to_delete) == 0 and not self.delete_mode:
            # Fallback: old behavior (delete the line_index)
            if self.line_index < 0 or self.line_index >= len(self.lines):
                return
            lines_to_delete = {self.line_index}

        for idx in sorted(lines_to_delete, reverse=True):
            if 0 <= idx < len(self.lines):
                self.lines = np.delete(self.lines, idx, axis=0)

        self.line_index = len(self.lines) - 1
        self.selected_lines.clear()

        self.save_flag = False
        self.widget_update()
        self.line_update()
        self.image_update()

    def ToggleDeleteMode_Callback(self):
        """Toggle the delete mode on/off."""
        self.delete_mode = not self.delete_mode
        if self.delete_mode:
            self.statusBar().showMessage("Delete Mode ON - click lines to select/unselect them")
            self.button_DeleteMode.setText("Delete Mode: ON")
        else:
            self.statusBar().clearMessage()
            self.button_DeleteMode.setText("Delete Mode")

        self.selected_lines.clear()
        self.image_update()

    def Transform_Callback(self):
        self.Save_Callback()
        self.transform_flag = not self.transform_flag
        self.data_update()
        self.image_update()

    def ZoomIn_Callback(self):
        self.scale = float(np.round(self.scale + 0.1, decimals=1))
        self.zoom_update()
        self.image_update()

    def ZoomOut_Callback(self):
        self.scale = float(np.round(self.scale - 0.1, decimals=1))
        self.zoom_update()
        self.image_update()

    def TextZoom_Callback(self):
        text = self.text_zoom.text().strip('%')
        self.scale = np.round(float(text) / 100.0, decimals=1)
        self.scale = float(np.clip(self.scale, self.scale_limit[0], self.scale_limit[1]))
        self.zoom_update()
        self.image_update()

    ###
    # Updated mouse event overrides for endpoint dragging:
    # We attempt to find the nearest line endpoint among ALL lines, then we
    # automatically select that line (line_index) if found.
    ###
    def mousePress_Callback(self, event):
        if event.button() == Qt.LeftButton:
            x, y = self._mapToImageCoords(event)
            if x < 0 or y < 0:
                return

            # If user is NOT creating or deleting lines, let's see if they're near any endpoint
            if not self.label_flag and not self.delete_mode and len(self.lines) > 0:
                best_dist = float('inf')
                best_line = -1
                best_endpoint = -1
                # Check all lines for the closest endpoint
                for i, line in enumerate(self.lines):
                    for ep_idx in range(2):
                        dist = np.linalg.norm(line[ep_idx] - [x, y])
                        if dist < best_dist:
                            best_dist = dist
                            best_line = i
                            best_endpoint = ep_idx
                # If the best line is within self.endpoint_select_thresh, drag it
                if best_dist <= self.endpoint_select_thresh:
                    # Automatically select that line
                    self.line_index = best_line
                    self.line_update()
                    self.image_update()

                    self.is_dragging = True
                    self.dragging_line_index = best_line
                    self.dragging_endpoint_index = best_endpoint
                    return

        # If not handled above, pass on to the default logic
        super().mousePressEvent(event)

    def mouseRelease_Callback(self, event):
        # If we were dragging an endpoint, finalize that
        if self.is_dragging and event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.dragging_line_index = -1
            self.dragging_endpoint_index = -1
            self.save_flag = False
            self.image_update()
            return

        # Otherwise, do the old logic
        if event.button() == Qt.RightButton:
            if self.button_Create.isEnabled():
                self.Create_Callback()
            else:
                self.label_flag = False
                self.label_endpoint = None
                self.capture_endpoint = None
                self.setCursor(Qt.ArrowCursor)
                self.widget_update()
                self.line_update()
                self.zoom_update()
                self.image_update()
        elif event.button() == Qt.LeftButton:
            self._handleLeftReleaseForCreatingOrSelecting(event)
        # Could handle middle button, etc.

    def mouseMove_Callback(self, event):
        if self.is_dragging:
            # If dragging an endpoint, continuously update that endpoint
            x, y = self._mapToImageCoords(event)
            if x < 0 or y < 0:
                return
            if 0 <= self.dragging_line_index < len(self.lines):
                self.lines[self.dragging_line_index][self.dragging_endpoint_index] = [x, y]
                self.image_update()
            return
        else:
            # Not dragging; do old logic for capture_endpoint if in create mode
            last_capture_endpoint = self.capture_endpoint
            self.capture_endpoint = None
            if self.label_flag and len(self.lines) > 0:
                width, height = self.image.shape[1], self.image.shape[0]
                x, y = self._mapToImageCoords(event)
                if 0 <= x < width and 0 <= y < height:
                    pt = np.array([x, y], np.float32)
                    pts = self.lines.reshape(-1, 2)
                    dists = np.linalg.norm(pts - pt[None], axis=-1)
                    dist = dists.min()
                    if dist <= self.point_select_thresh:
                        index = dists.argmin()
                        self.capture_endpoint = pts[index]

            # Update UI if capture endpoint changed
            if last_capture_endpoint is not None or self.capture_endpoint is not None:
                self.image_update()

    def _handleLeftReleaseForCreatingOrSelecting(self, event):
        width, height = self.image.shape[1], self.image.shape[0]
        x, y = self._mapToImageCoords(event)
        if x < 0 or x >= width or y < 0 or y >= height:
            return

        pt = np.array([x, y], np.float32)

        if self.delete_mode:
            # In delete mode, left-click toggles selection of the nearest line
            if len(self.lines) == 0:
                return
            dists = []
            pts_list = self.camera.interp_line(self.lines)
            for i, pts_ in enumerate(pts_list):
                dist = np.linalg.norm(pts_ - pt[None], axis=-1).min()
                dists.append(dist)
            dists = np.asarray(dists)
            min_dist = dists.min()
            if min_dist <= self.line_select_thresh:
                clicked_line = dists.argmin()
                if clicked_line in self.selected_lines:
                    self.selected_lines.remove(clicked_line)
                else:
                    self.selected_lines.add(clicked_line)
            self.image_update()
            return

        if self.label_flag:
            # Creating a new line
            if self.label_endpoint is None:
                if self.capture_endpoint is not None:
                    self.label_endpoint = self.capture_endpoint
                    self.capture_endpoint = None
                else:
                    self.label_endpoint = pt
                self.image_update()
            else:
                if abs(self.label_endpoint[0] - pt[0]) <= self.point_align_thresh:
                    pt[0] = self.label_endpoint[0]
                if abs(self.label_endpoint[1] - pt[1]) <= self.point_align_thresh:
                    pt[1] = self.label_endpoint[1]
                self.label_flag = False
                if self.capture_endpoint is not None:
                    pt = self.capture_endpoint
                    self.capture_endpoint = None

                line = np.stack((self.label_endpoint, pt))
                self.lines = np.concatenate((self.lines, line[None]))
                self.line_index = len(self.lines) - 1

                self.save_flag = False
                self.setCursor(Qt.ArrowCursor)
                self.widget_update()
                self.line_update()
                self.zoom_update()
                self.image_update()
        else:
            # Not in create mode or delete mode => pick line to highlight
            if len(self.lines) == 0:
                return
            dists = []
            pts_list = self.camera.interp_line(self.lines)
            for pts_ in pts_list:
                dist = np.linalg.norm(pts_ - pt[None], axis=-1).min()
                dists.append(dist)
            dists = np.asarray(dists)
            min_dist = dists.min()
            if min_dist > self.line_select_thresh:
                return

            self.line_index = dists.argmin()
            self.line_update()
            self.image_update()

    def _mapToImageCoords(self, event):
        """
        Utility: given a mouse event, return (x, y) in the image's coordinate space.
        Returns (-1, -1) if out of bounds.
        """
        width, height = self.image.shape[1], self.image.shape[0]
        image_width = int(round(width * self.scale))
        image_height = int(round(height * self.scale))
        widget_width = self.label_Image.width()
        widget_height = self.label_Image.height()
        dx = (widget_width - image_width) / 2.0
        dy = (widget_height - image_height) / 2.0

        mx = event.x() - dx
        my = event.y() - dy
        if mx < 0 or my < 0:
            return -1, -1
        x_img = np.round(mx / self.scale, decimals=self.decimal_precision)
        y_img = np.round(my / self.scale, decimals=self.decimal_precision)

        if x_img < 0 or x_img >= width or y_img < 0 or y_img >= height:
            return -1, -1
        return float(x_img), float(y_img)

    def Tutorial_Callback(self):
        QMessageBox.information(
            self,
            'Tutorial',
            'Version: 3.1\n'
            'Author: lh9171338 + ChatGPT\n'
            'Date: 2022-02-20\n'
            'Features:\n'
            '- Create lines by left-clicking twice in the image (Ctrl + C)\n'
            '- Delete multiple lines (Delete Mode, Ctrl+Shift+D)\n'
            '- Drag endpoints of any line to move them (no need to select first)\n'
            '- Larger endpoint selection threshold (configurable)\n'
            '\n'
            'Shortcut (default):\n'
            '\tCtrl + O: Select an image folder\n'
            '\tCtrl + S: Save the annotations\n'
            '\tCtrl + V: Go to the next image\n'
            '\tCtrl + B: Go to the previous image\n'
            '\tCtrl + C: Create a new annotation\n'
            '\tCtrl + D: Delete the selected annotation\n'
            '\tCtrl + Shift + D: Toggle delete mode\n'
            '\tCtrl + U: View the tutorial\n'
            '\n'
            'Mouse usage:\n'
            '\tLeft-click: Create endpoints if in create mode, toggle line selection in delete mode,\n'
            '\t            or drag endpoints if near them.\n'
            '\tRight-click: Quickly start a new line (if create is enabled).',
            QMessageBox.Close,
        )

    def ExportMat_Callback(self):
        """Export all .mat files' content in the label folder to a result.txt file."""
        if not hasattr(self, 'data_path') or not self.data_path:
            QMessageBox.warning(self, 'Warning', 'No folder opened. Please open a folder first.')
            return

        label_path = os.path.join(self.data_path, self.label_folder)
        output_file = os.path.join(self.data_path, 'result.txt')

        if not os.path.isdir(label_path):
            QMessageBox.warning(self, 'Warning', 'Label folder does not exist.')
            return

        mat_files = glob.glob(os.path.join(label_path, '*.mat'))
        if not mat_files:
            QMessageBox.warning(self, 'Warning', 'No .mat files found in the label folder.')
            return

        with open(output_file, 'w') as f:
            for mat_file in mat_files:
                try:
                    data = sio.loadmat(mat_file)
                    f.write(f'File: {os.path.basename(mat_file)}\n')
                    for key, value in data.items():
                        if not key.startswith('__'):
                            f.write(f'{key}: {value}\n')
                    f.write('\n')
                except Exception as e:
                    logging.error(f"Error reading {mat_file}: {e}")
                    f.write(f"Error reading {mat_file}: {e}\n")

        QMessageBox.information(self, 'Success', f'Exported .mat contents to {output_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=int, choices=[0, 1, 2],
                        help='0: pinhole image, 1: fisheye image, 2: spherical image', required=True)
    parser.add_argument('-c', '--coeff_file', type=str, help='camera distortion coefficients file')
    opts = parser.parse_args()
    opts_dict = vars(opts)
    opts_list = []
    for key, value in zip(opts_dict.keys(), opts_dict.values()):
        if value is not None:
            opts_list.append(key)
            opts_list.append(value)

    cfg = CfgNode.load_cfg(open('default.yaml'))
    cfg.merge_from_list(opts_list)
    cfg.freeze()
    print(cfg)

    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    window.OpenDir_Callback()
    sys.exit(app.exec_())
