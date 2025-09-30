#
# util.py
# arma-rs-utils
#
# Created by Junggyun Oh on 04/19/2023.
# Copyright (c) 2023 Junggyun Oh All rights reserved.
#
import os
import os.path as pth
import subprocess
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget


def create_extra_dialog():
    message_box = QMessageBox()
    message_box.setWindowTitle("Hello Out There")
    message_box.setTextFormat(Qt.RichText)
    msg = "¯\\_(ツ)_/¯ \
        <br> Copyright &copy; 2023 Junggyun Oh and Yechan Kim. All rights reserved. \
        <br> Please report bugs and submit additional feature requests here. \
        <br> We appreciate your support! \
        <br> => <a href='https://github.com/dodant/AMOD-viewer'>dodant/AMOD-viewer</a>"
    message_box.setText(msg)
    message_box.exec()


def move_to_center(window):
    qr = window.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    window.move(qr.topLeft())


def open_file(file_name):
    if file_name:
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', file_name))
        elif os.name == 'nt':  # For Windows
            os.startfile(file_name)
        elif os.name == 'posix':  # For Linux, Mac, etc.
            subprocess.call(('xdg-open', file_name))


def count_legit_folder(set_path):
    return len([i for i in os.listdir(set_path) if pth.isdir(pth.join(set_path, i))])


def get_color_info(row, mono, label_color):
    main_class, middle_class = row['main_class'], row['middle_class']
    color_key = main_class if main_class in mono else f'{main_class}/{middle_class}'
    return label_color[color_key]


def usable_check(t_checkbtn, f_checkbtn, usable):
    draw_true_circle = t_checkbtn.isChecked() and usable == 'T'
    draw_false_circle = f_checkbtn.isChecked() and usable == 'F'
    return draw_true_circle or draw_false_circle


def scene_navigation(func):
    def wrapper(self):
        if not self.ds.get_scene_path_list():
            return
        idx = self.ds.get_scene_index()
        length = len(self.ds.get_scene_path_list())
        self.ds.set_scene_index(func(self, idx, length))
        if self.auto_radio.isChecked():
            self.auto_plot_index = 0
            self.ds.current_view_idx = str(self.angle[self.auto_plot_index])
            self.ds.set_view_name(self.ds.current_view_idx)
        else:
            self.ds.set_view_name(str(self.ds.base_view_idx))
        self.change_image_at_scene()
    return wrapper


def view_navigation(func):
    def wrapper(self):
        view_names, _ = self.ds.get_view_name_path_list()
        idx = view_names.index(int(self.ds.get_view_name()))
        new_view = func(self, view_names, idx)
        if new_view is not None:
            self.ds.set_view_name(new_view)
            self.change_image_at_view()
    return wrapper

# util.py 파일의 맨 마지막에 추가 (또는 적절한 위치에)

from functools import wraps
from PyQt5.QtWidgets import QMessageBox

# scene_navigation_modified 데코레이터 추가
def scene_navigation_modified(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'ds') or self.ds is None:
            QMessageBox.warning(self, "경고", "데이터셋이 로드되지 않았습니다.")
            return
        func(self)
        if hasattr(self, 'ds') and self.ds is not None:
            self.ds.set_view_name(str(self.ds.base_view_idx)) # 씬 변경 시 기본 뷰로 리셋
    return wrapper

# view_navigation_modified 데코레이터 추가
def view_navigation_modified(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'ds') or self.ds is None:
            QMessageBox.warning(self, "경고", "데이터셋이 로드되지 않았습니다.")
            return

        view_names = self.ds.get_view_name_list()
        if not view_names:
            QMessageBox.warning(self, "경고", "현재 씬에 뷰가 없습니다.")
            return
            
        current_view_name = self.ds.get_view_name()
        try:
            idx = view_names.index(current_view_name)
        except ValueError:
            QMessageBox.warning(self, "경고", f"현재 뷰 '{current_view_name}'를 찾을 수 없습니다.")
            return
            
        func(self, view_names, idx, *args, **kwargs)
    return wrapper
