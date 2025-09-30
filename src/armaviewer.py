#
# armaviewer.py (Old Data BBOX 기능 추가 및 명칭 정리 최종 버전)
#
import os
import os.path as pth
import random as rd
import sys
from datetime import datetime
import warnings
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import qimage2ndarray as q2n
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QApplication, QWidget, QRadioButton, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton,
                             QLabel, QCheckBox, QButtonGroup, QInputDialog, QSizePolicy, QListWidgetItem, QLineEdit, QFileDialog, QListWidget, QMessageBox)
from PyQt5.QtGui import QPixmap, QColor, QPainter, QPen, QFont, QIntValidator, QIcon
import multiviewset as MVS
import util
from annotation_object import AnnotationObject # annotation_object.py가 필요합니다.

warnings.filterwarnings("ignore", category=DeprecationWarning)

class ArmaViewer(QWidget):
    def __init__(self):
        super().__init__(flags=Qt.Window)
        self.supported_classes = (
            'Armored', 'Artillery', 'Boat', 'Helicopter', 'LCU', 'MLRS', 'Plane', 'RADAR', 'SAM',
            'Self-propelled Artillery', 'Support', 'TEL', 'Tank'
        )
        self.initialize_var_let_widget()
        self.init_levels()
        self.make_levels()
        self.init_ui()

    def initialize_var_let_widget(self):

        def init_ui_settings_constants():
            self.remark_limit = 30
            self.side_space, self.updown_space = 15, 10

        def init_global_constants():
            self.key_map = {
                Qt.Key_A: self.goto_prev_scene, Qt.Key_D: self.goto_next_scene,
                Qt.Key_W: self.goto_prev_view, Qt.Key_S: self.goto_next_view,
            }
            self.FIXED_COLOR_STYLE = {
                'Armored': [(244,67,54), '-'], 'Armored:APC': [(244,67,54), '-'], 'Armored:ASV': [(239,62,49), '--'],
                'Armored:IFV': [(234,57,44), '-.'], 'Armored:MRAP': [(229,52,39), ':'], 'Artillery': [(255,51,204), '-'],
                'Boat': [(156,39,176), '-'], 'Boat:Boat': [(156,39,176), '-'], 'Boat:RHIB': [(151,34,171), '--'],
                'Helicopter': [(103,58,183), '-'], 'Helicopter:AH': [(103,58,183), '-'], 'Helicopter:CH': [(98,53,178), '--'],
                'Helicopter:OH': [(93,48,173), '-.'], 'Helicopter:UH': [(88,43,168), ':'], 'LCU': [(63,81,181), '-'],
                'MLRS': [(33,150,243), '-'], 'Plane': [(0,188,212), '-'], 'Plane:Attacker': [(0,188,212), '-'],
                'Plane:Bomber': [(0,183,207), '--'], 'Plane:Cargo': [(0,178,202), '-.'], 'Plane:Fighter': [(0,173,197), ':'],
                'RADAR': [(0,150,136), '-'], 'SAM': [(76,175,80), '-'], 'Self-propelled Artillery': [(139,195,74), '-'],
                'Support': [(205,220,57), '-'], 'Support:Mil_car': [(205,220,57), '-'],
                'Support:Mil_truck': [(200,215,52), '--'], 'Support:ASV':[(195,210,47), '-.'],
                'Tank': [(255,122,0), '-'], 'TEL': [(121,85,72), '-'],
            }
            self.dsize = (1280, 720)

        def init_global_variables():
            self.ds: Optional[MVS.MultiViewSet] = None
            self.anno_file = pd.DataFrame()
            self.checked_list = []
            self.auto_plot_timer = QTimer()
            self.auto_plot_index = 0
            self.angle = []
            self.annotation_objects = []
            self.selected_object: Optional[AnnotationObject] = None
            self.transform_step = 1.0
            self.scale_step = 0.01
            self.angle_step = 1.0
            self.edit_mode = False
            self.old_data_objects = [] # ★★★ Old Data BBOX를 위한 객체 리스트 ★★★
            self.current_display_objects = [] # AnnotationObject 인스턴스들을 저장할 리스트
            self.legend_widget = QListWidget(self) # 레전드 위젯 초기화

        def init_lvl0_panel_widget():
            self.image_widget = QLabel(self)
            self.image_pixmap = QPixmap('figs/AMOD-Viewer-Logo.png')
            target_height = 130
            target_width = int(self.image_pixmap.width() * (target_height / self.image_pixmap.height()))
            self.image_pixmap = self.image_pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio)
            self.image_widget.setPixmap(self.image_pixmap)
            self.image_widget.setFixedSize(self.image_pixmap.size())

        def init_lvl1_panel_widget():
            self.set_input = QLineEdit(self, placeholderText='Set Path - ex) /home/user/Downloads/altis_sunny_6_20')
            self.set_select_btn = QPushButton('Open', self)

        def lvl1_panel_widget_setting():
            self.set_input.setFixedWidth(600)
            self.set_select_btn.setFixedWidth(100)
            self.set_select_btn.clicked.connect(self.db_parse)

        def init_lvl2_panel_widget():
            self.num_of_scene_lbl = QLabel(f'{"|" * 7} W {"|" * 8}')
            self.goto_input = QLineEdit(self, placeholderText='#')
            self.goto_scene_btn = QPushButton('Go', self)
            self.prev_scn, self.first_scn, self.next_scn, self.prev_vue, self.base_vue, self.next_vue = \
                (QPushButton(i, self) for i in ('<<<', '#1', '>>>', '-', '0', '+'))

        def lvl2_panel_widget_setting():
            btns = (self.goto_input, self.goto_scene_btn, self.prev_scn, self.first_scn, self.next_scn)
            sizes = (80, 30, 40, 29, 40)
            [btn.setFixedWidth(width) for btn, width in zip(btns, sizes)]
            [btn.setFixedWidth(30) for btn in (self.prev_vue, self.base_vue, self.next_vue)]

            [btn.clicked.connect(func) for func, btn in
             zip((self.goto_prev_scene, self.goto_first_scene, self.goto_next_scene),
                 (self.prev_scn, self.first_scn, self.next_scn))]
            [btn.clicked.connect(func) for func, btn in
             zip((self.goto_prev_view, self.goto_base_view, self.goto_next_view),
                 (self.prev_vue, self.base_vue, self.next_vue))]
            self.goto_scene_btn.clicked.connect(self.goto_scene)

        def init_lvl3_panel_widget():
            title = ('Random Shuffle', 'Sort', 'Remark Sort')
            self.file_num_name = QLabel(f'{"|" * 5} A  S  D {"|" * 5}')
            self.pix_w_input, self.pix_h_input = QLineEdit(self, placeholderText='1280'), QLineEdit(self, placeholderText='720')
            self.pix_x = QLabel(' ✕ ')
            self.set_pix_size = QPushButton('Set', self)
            self.img_mix, self.img_sort, self.rmk_sort = (QPushButton(i, self) for i in title)

        def lvl3_panel_widget_setting():
            [btn.setFixedWidth(width) for btn, width in
             zip((self.img_mix, self.img_sort, self.rmk_sort), (125, 43, 95))]
            self.pix_w_input.setFixedWidth(60)
            self.pix_h_input.setFixedWidth(60)
            self.set_pix_size.setFixedWidth(40)
            self.set_pix_size.clicked.connect(self.change_res)
            self.img_mix.clicked.connect(lambda: rd.shuffle(self.ds.get_scene_path_list()) if self.ds else None)
            self.img_sort.clicked.connect(lambda: self.ds.get_scene_path_list().sort() if self.ds else None)
            self.rmk_sort.clicked.connect(self.remark_sort)

        def init_lvl4_panel_widget():
            self.mode_group = QButtonGroup()
            self.mode_box_group = QGroupBox('Mode')
            self.mode_layout = QHBoxLayout()
            self.view_mode_radio = QRadioButton('View Mode', self)
            self.edit_mode_radio = QRadioButton('Edit Mode', self)
            self.mode_layout.addWidget(self.view_mode_radio)
            self.mode_layout.addWidget(self.edit_mode_radio)
            self.mode_box_group.setLayout(self.mode_layout)
            self.view_mode_radio.setChecked(True)

            self.use_group = QButtonGroup()
            self.usable_btngroup, self.label_btngroup, self.animate_btngroup = (QGroupBox(i) for i in ('Usable', 'Label', 'Animate'))
            self.usable_box, self.label_box, self.animate_box = (QHBoxLayout() for _ in range(3))
            self.t_check, self.f_check = (QCheckBox(i) for i in ('True', 'False'))
            
            # ★★★ 체크박스 UI 재정의 ★★★
            label_titles = ('Center Point', 'Oriented BBOX', 'BBox', 'Main', 'Middle', 'ID', 'Show Original Box', 'Old Oriented BBOX')
            self.center_check, self.obox_check, self.bbox_check, self.main_check, self.mid_check, self.id_check, self.show_original_box_check, self.old_obox_check = \
                (QCheckBox(i, self) for i in label_titles) # <--- 여기 'self.old_obox_check'로 정확히 선언됩니다.
            self.static_radio, self.auto_radio = (QRadioButton(i) for i in ('Static', 'Dynamic'))

        def lvl4_panel_widget_setting():
            self.mode_group.addButton(self.view_mode_radio, 0)
            self.mode_group.addButton(self.edit_mode_radio, 1)
            self.mode_group.buttonClicked[int].connect(self.set_mode)

            self.use_group.setExclusive(False)
            self.usable_btngroup.setLayout(self.usable_box)
            self.t_check.setChecked(True)
            for btn in [self.t_check, self.f_check]: self.usable_box.addWidget(btn, alignment=Qt.AlignLeading)
            self.use_group.buttonClicked.connect(self.render_refined_scene)

            self.label_btngroup.setLayout(self.label_box)
            # ★★★ 모든 체크박스 연결 및 Old Data BBOX 초기화 ★★★
            for i, btn in enumerate([self.center_check, self.obox_check, self.bbox_check, self.main_check, self.mid_check, self.id_check, self.show_original_box_check, self.old_obox_check]): # <--- 이 부분!
                self.label_box.addWidget(btn, alignment=Qt.AlignCenter)
                btn.toggled.connect(self.checkbox_toggle)
                btn.toggled.connect(self.render_refined_scene)
                btn.toggled.connect(self.create_legend)
                
                # Old Data BBOX 체크박스만 특별히
                
            self.obox_check.setChecked(True) # OBOX는 기본으로 체크

            self.old_obox_check.toggled.connect(self.load_old_data_bbox_if_needed)

            self.obox_check.setChecked(True) # OBOX는 기본으로 체크

            self.animate_btngroup.setLayout(self.animate_box)
            self.animate_box.addWidget(self.static_radio, alignment=Qt.AlignCenter)
            self.animate_box.addWidget(self.auto_radio, alignment=Qt.AlignCenter)
            self.static_radio.setChecked(True)
            self.static_radio.clicked.connect(lambda: self.auto_plot_timer.stop())
            self.auto_plot_timer.timeout.connect(self.auto_plot_step)
            self.auto_radio.clicked.connect(self.auto_plot)
        
        def init_lvl5_panel_widget():
            self.pixmap, self.lbl_img = QPixmap(), QLabel()
            self.view_group = QButtonGroup()
            self.lgd_layout, self.view_layout = QVBoxLayout(), QVBoxLayout()
            self.lgd_box, self.view_box = QGroupBox('Legend'), QGroupBox('View')
            self.lgd_box.setLayout(self.lgd_layout) # lgd_box의 레이아웃 설정
            self.lgd_layout.addWidget(self.legend_widget) # ★★★ legend_widget을 lgd_layout에 추가 ★★★
            self.lgd_layout.addStretch(1) # 레전드 아이템들이 상단에 모이도록

        def lvl5_panel_widget_setting():
            self.lgd_box.setFixedWidth(167)
            self.lgd_box.setFixedHeight(200)
            self.view_box.setFixedWidth(167)
            self.view_box.setFixedHeight(250)
            self.view_group.setExclusive(True)

        def init_lvl6_panel_widget():
            title = ('PNG', 'Save Annotations', 'Report Issue', 'Hello Out There')
            self.png_save, self.save_anno_btn, self.report, self.extra = (QPushButton(i, self) for i in title)
            self.svg_save, self.pdf_save, self.gif_save = (QPushButton(i, self) for i in ('SVG', 'PDF', 'GIF'))
            self.indicator = QLabel(f'{"|" * 45}')

        def lvl6_panel_widget_setting():
            self.png_save.setFixedWidth(50)
            self.save_anno_btn.setFixedWidth(120)
            for btn in (self.svg_save, self.pdf_save, self.gif_save):
                btn.setFixedWidth(50)
                btn.setEnabled(False)

            self.png_save.clicked.connect(self.save_png)
            self.save_anno_btn.clicked.connect(self.save_modified_annotations)
            self.report.clicked.connect(self.create_report_dialog)
            self.extra.clicked.connect(util.create_extra_dialog)

        def init_transform_panel_widget():
            self.transform_box = QGroupBox('Transformation Controls', self)
            main_layout = QVBoxLayout()
            self.selected_id_label = QLabel("Selected ID: N/A", self)
            font = self.selected_id_label.font()
            font.setBold(True)
            font.setPointSize(font.pointSize() + 1) # 조금 더 크게, 볼드체로
            self.selected_id_label.setFont(font)
            self.selected_id_label.setAlignment(Qt.AlignCenter) # 중앙 정렬
            main_layout.addWidget(self.selected_id_label) # 가장 위에 추가
        
            self.tx_layout, self.ty_layout = QHBoxLayout(), QHBoxLayout()
            self.sw_layout, self.sh_layout = QHBoxLayout(), QHBoxLayout()
            self.rot_layout = QHBoxLayout()

            self.tx_label = QLabel("Translate X:")
            self.tx_edit = QLineEdit("0.00", self)
            self.tx_up_btn, self.tx_down_btn = QPushButton("▲", self), QPushButton("▼", self)
            for w in [self.tx_label, self.tx_edit, self.tx_up_btn, self.tx_down_btn]: self.tx_layout.addWidget(w)

            self.ty_label = QLabel("Translate Y:")
            self.ty_edit = QLineEdit("0.00", self)
            self.ty_up_btn, self.ty_down_btn = QPushButton("▲", self), QPushButton("▼", self)
            for w in [self.ty_label, self.ty_edit, self.ty_up_btn, self.ty_down_btn]: self.ty_layout.addWidget(w)

            self.sw_label = QLabel("Scale W:")
            self.sw_edit = QLineEdit("1.00", self)
            self.sw_up_btn, self.sw_down_btn = QPushButton("▲", self), QPushButton("▼", self)
            for w in [self.sw_label, self.sw_edit, self.sw_up_btn, self.sw_down_btn]: self.sw_layout.addWidget(w)

            self.sh_label = QLabel("Scale H:")
            self.sh_edit = QLineEdit("1.00", self)
            self.sh_up_btn, self.sh_down_btn = QPushButton("▲", self), QPushButton("▼", self)
            for w in [self.sh_label, self.sh_edit, self.sh_up_btn, self.sh_down_btn]: self.sh_layout.addWidget(w)
            
            self.rot_label = QLabel("Rotation:")
            self.rot_edit = QLineEdit("0.00", self)
            self.rot_cw_btn, self.rot_ccw_btn = QPushButton("⟳", self), QPushButton("⟲", self)
            for w in [self.rot_label, self.rot_edit, self.rot_cw_btn, self.rot_ccw_btn]: self.rot_layout.addWidget(w)

            for layout in [self.tx_layout, self.ty_layout, self.sw_layout, self.sh_layout, self.rot_layout]:
                main_layout.addLayout(layout)
            
            self.transform_box.setLayout(main_layout)
            self.transform_box.setFixedWidth(250)
            self.set_transform_controls_enabled(False)

        def transform_panel_widget_setting():
            self.tx_up_btn.clicked.connect(lambda: self.adjust_transform('tx', self.transform_step))
            self.tx_down_btn.clicked.connect(lambda: self.adjust_transform('tx', -self.transform_step))
            self.ty_up_btn.clicked.connect(lambda: self.adjust_transform('ty', self.transform_step))
            self.ty_down_btn.clicked.connect(lambda: self.adjust_transform('ty', -self.transform_step))
            self.sw_up_btn.clicked.connect(lambda: self.adjust_transform('sw', self.scale_step))
            self.sw_down_btn.clicked.connect(lambda: self.adjust_transform('sw', -self.scale_step))
            self.sh_up_btn.clicked.connect(lambda: self.adjust_transform('sh', self.scale_step))
            self.sh_down_btn.clicked.connect(lambda: self.adjust_transform('sh', -self.scale_step))
            self.rot_cw_btn.clicked.connect(lambda: self.adjust_transform('angle', -self.angle_step))
            self.rot_ccw_btn.clicked.connect(lambda: self.adjust_transform('angle', self.angle_step))
            
            self.tx_edit.editingFinished.connect(lambda: self.apply_transform_from_text_edit('tx', self.tx_edit.text()))
            self.ty_edit.editingFinished.connect(lambda: self.apply_transform_from_text_edit('ty', self.ty_edit.text()))
            self.sw_edit.editingFinished.connect(lambda: self.apply_transform_from_text_edit('sw', self.sw_edit.text()))
            self.sh_edit.editingFinished.connect(lambda: self.apply_transform_from_text_edit('sh', self.sh_edit.text()))
            self.rot_edit.editingFinished.connect(lambda: self.apply_transform_from_text_edit('angle', self.rot_edit.text()))

        init_ui_settings_constants()
        init_global_constants()
        init_global_variables()
        init_lvl0_panel_widget()
        init_lvl1_panel_widget()
        init_lvl2_panel_widget()
        init_lvl3_panel_widget()
        init_lvl4_panel_widget()
        init_lvl5_panel_widget()
        init_lvl6_panel_widget()
        init_transform_panel_widget()
        lvl1_panel_widget_setting()
        lvl2_panel_widget_setting()
        lvl3_panel_widget_setting()
        lvl4_panel_widget_setting()
        lvl5_panel_widget_setting()
        lvl6_panel_widget_setting()
        transform_panel_widget_setting()

    def init_levels(self):
        self.lvl0, self.lvl1, self.lvl2, self.lvl3, self.lvl4, self.lvl5, self.lvl6 = (QHBoxLayout() for _ in range(7))

    def make_levels(self):
        self.lvl0.addSpacing(self.side_space)
        self.lvl0.addWidget(self.image_widget, alignment=Qt.AlignCenter)
        self.lvl0.addSpacing(self.side_space)

        self.lvl1.addSpacing(self.side_space)
        self.lvl1.addWidget(self.set_input, alignment=Qt.AlignCenter)
        self.lvl1.addWidget(self.set_select_btn, alignment=Qt.AlignCenter)
        self.lvl1.addStretch(1)
        self.lvl1.addSpacing(self.side_space)

        self.lvl2.addSpacing(self.side_space)
        self.lvl2.addWidget(self.num_of_scene_lbl, alignment=Qt.AlignCenter)
        self.lvl2.addStretch(1)
        self.lvl2.addWidget(self.goto_input, alignment=Qt.AlignCenter)
        self.lvl2.addWidget(self.goto_scene_btn, alignment=Qt.AlignCenter)
        self.lvl2.addSpacing(5)
        self.lvl2.addWidget(self.prev_scn, alignment=Qt.AlignCenter)
        self.lvl2.addWidget(self.first_scn, alignment=Qt.AlignCenter)
        self.lvl2.addWidget(self.next_scn, alignment=Qt.AlignCenter)
        self.lvl2.addSpacing(5)
        self.lvl2.addWidget(self.prev_vue, alignment=Qt.AlignCenter)
        self.lvl2.addWidget(self.base_vue, alignment=Qt.AlignCenter)
        self.lvl2.addWidget(self.next_vue, alignment=Qt.AlignCenter)
        self.lvl2.addSpacing(self.side_space)

        self.lvl3.addSpacing(self.side_space)
        self.lvl3.addWidget(self.file_num_name, alignment=Qt.AlignCenter)
        self.lvl3.addStretch(1)
        self.lvl3.addWidget(self.pix_w_input, alignment=Qt.AlignCenter)
        self.lvl3.addWidget(self.pix_x, alignment=Qt.AlignCenter)
        self.lvl3.addWidget(self.pix_h_input, alignment=Qt.AlignCenter)
        self.lvl3.addWidget(self.set_pix_size, alignment=Qt.AlignCenter)
        self.lvl3.addWidget(self.img_mix, alignment=Qt.AlignCenter)
        self.lvl3.addWidget(self.img_sort, alignment=Qt.AlignCenter)
        self.lvl3.addWidget(self.rmk_sort, alignment=Qt.AlignCenter)
        self.lvl3.addSpacing(self.side_space)

        self.lvl4.addSpacing(self.side_space)
        self.lvl4.addWidget(self.mode_box_group)
        self.lvl4.addWidget(self.usable_btngroup, alignment=Qt.AlignCenter)
        self.lvl4.addStretch(1)
        self.lvl4.addWidget(self.label_btngroup, alignment=Qt.AlignCenter)
        self.lvl4.addStretch(1)
        self.lvl4.addWidget(self.animate_btngroup, alignment=Qt.AlignCenter)
        self.lvl4.addSpacing(self.side_space)

        extra_box = QVBoxLayout()
        extra_box.addWidget(self.lgd_box, alignment=Qt.AlignTop)
        extra_box.addWidget(self.view_box, alignment=Qt.AlignTop)
        extra_box.addStretch(1)
        extra_box.addWidget(self.transform_box, alignment=Qt.AlignBottom)
        
        self.lvl5.addStretch(1)
        self.lbl_img = QLabel(self)
        self.lbl_img.setScaledContents(True)
        self.lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lvl5.addWidget(self.lbl_img, alignment=Qt.AlignCenter)
        self.lvl5.addStretch(1)
        self.lvl5.addLayout(extra_box)
        self.lvl5.addSpacing(self.side_space)

        self.lvl6.addSpacing(self.side_space)
        self.lvl6.addWidget(self.png_save, alignment=Qt.AlignCenter)
        self.lvl6.addWidget(self.save_anno_btn)
        self.lvl6.addStretch(1)
        self.lvl6.addWidget(self.indicator, alignment=Qt.AlignCenter)
        self.lvl6.addStretch(1)
        self.lvl6.addWidget(self.report, alignment=Qt.AlignCenter)
        self.lvl6.addWidget(self.extra, alignment=Qt.AlignCenter)
        self.lvl6.addSpacing(self.side_space)

    def init_ui(self):
        vbox = QVBoxLayout()
        vbox.addSpacing(self.updown_space)
        [vbox.addLayout(i) for i in (self.lvl0, self.lvl1, self.lvl2, self.lvl3, self.lvl4, self.lvl5, self.lvl6)]
        vbox.addSpacing(self.updown_space)
        self.setLayout(vbox)
        self.setWindowTitle('AMOD-Viewer (Refined Mode)')
        self.resize(1600, 900)
        util.move_to_center(self)
        self.show()

    def load_old_data_bbox_if_needed(self, checked):
        """'Old Data BBOX' 체크박스가 켜질 때, 필요한 old_data를 로드합니다."""
        # ★★★ 항상 old_data_objects를 초기화하여 이전 씬 데이터 잔류 방지 ★★★
        self.old_data_objects = [] 
        
        if checked: # 'checked'가 True일 때만 데이터 로드를 시도합니다.
            if not hasattr(self, 'ds') or self.ds is None:
                QMessageBox.warning(self, "경고", "데이터셋이 먼저 로드되어야 합니다.")
                self.old_obox_check.setChecked(False)
                # 이 경우 데이터 로드 실패 후 화면 갱신 및 레전드 업데이트
                self.render_refined_scene() 
                self.create_legend()
                return

            preloaded_data = self.ds.get_preloaded_data_for_current_view()
            if not preloaded_data or preloaded_data['csv'] is None:
                QMessageBox.warning(self, "경고", "현재 뷰의 CSV 데이터를 찾을 수 없습니다.")
                self.old_obox_check.setChecked(False)
                # 이 경우 데이터 로드 실패 후 화면 갱신 및 레전드 업데이트
                self.render_refined_scene() 
                self.create_legend()
                return

            csv_data = preloaded_data['csv']
            if 'x1_old' not in csv_data.columns:
                QMessageBox.information(self, "정보", "현재 CSV 파일에 'old' 데이터가 없습니다.")
                self.old_obox_check.setChecked(False)
                # 이 경우 old 데이터가 없으므로 체크박스 해제 후 화면 갱신 및 레전드 업데이트
                self.render_refined_scene() 
                self.create_legend()
                return

            for idx, row in csv_data.iterrows():
                if pd.notna(row.get('x1_old')):
                    old_obj = {
                        'points': np.array([
                            [row['x1_old'], row['y1_old']],
                            [row['x2_old'], row['y2_old']],
                            [row['x3_old'], row['y3_old']],
                            [row['x4_old'], row['y4_old']]
                        ], dtype=np.int32),
                        'id': row.get('id', f'obj_{idx}')
                    }
                    self.old_data_objects.append(old_obj)

            
        # 'checked'가 False이면, 위에서 이미 self.old_data_objects가 비워졌으므로 추가 작업 불필요.

        self.render_refined_scene() # 화면 다시 그리기
        self.create_legend() # 레전드도 업데이트 (Old Data BBOX 항목 추가/제거)
    
    
     
        self.render_refined_scene() # 화면 다시 그리기
        self.create_legend()
    def db_parse(self):
        input_path = self.set_input.text().strip()
        if not os.path.isdir(input_path):
            QMessageBox.warning(self, "경고", "유효한 폴더 경로가 아닙니다.")
            return

        try:
            self.ds = MVS.MultiViewSet()
            self.ds.set_path_and_name(input_path)
            self.ds.update_best_view_idx()
            self.num_of_scene_lbl.setText(f'# of Scenes : {util.count_legit_folder(self.ds.get_set_path())}')
            self.ds.preload_scene_data()
            self.change_image_at_scene()
            self.set_mode(0)
        except Exception as e:
            print(f"데이터셋 파싱 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "오류", f"데이터셋 파싱 중 오류가 발생했습니다:\n{e}")

    def set_mode(self, mode_id):
        self.edit_mode = (mode_id == 1)

        if self.selected_object:
            self.selected_object.is_selected = False
            self.selected_object = None
        
        self.set_transform_controls_enabled(self.edit_mode)
        self.update_transform_display()
        self.render_refined_scene()
    
    def render_refined_scene(self):
        if not hasattr(self, 'ds') or self.ds is None:
            # 데이터셋이 로드되지 않았을 경우, 빈 화면을 표시하고 종료
            canvas = np.full((self.dsize[1], self.dsize[0], 3), 240, dtype=np.uint8) # 회색 배경 BGR
            # set_scale_and_policy에서 BGR->RGB 변환을 처리하므로 여기에선 필요 없음
            self.set_scale_and_policy(canvas) 
            self.change_image_info()
            return

        preloaded = self.ds.get_preloaded_data_for_current_view()
        if not preloaded:
            # 사전 로드된 데이터가 없을 경우, 빈 화면 표시하고 종료
            canvas = np.full((self.dsize[1], self.dsize[0], 3), 240, dtype=np.uint8) # 회색 배경 BGR
            self.set_scale_and_policy(canvas)
            self.change_image_info()
            return

        image = preloaded['image'] # 이 image가 RGB인지 BGR인지 확인 필요
        csv_data = preloaded['csv']

        # ★★★ 이미지 컬러 스페이스 일관성 확보: BGR로 통일 ★★★
        # 만약 preloaded['image']가 RGB로 로드된다면, 여기서 BGR로 변환하여 OpenCV 그리기와 일치시킵니다.
        # 대부분의 cv2.imread는 BGR로 로드하므로, 이 변환이 필요 없을 수도 있습니다.
        # 하지만 배경색이 이상하다면 이 부분이 원인일 가능성이 높습니다.
        if image.ndim == 3 and image.shape[2] == 3:
            # 이 조건을 통해 image가 RGB인지 BGR인지 알 수 없으므로,
            # 안전하게 is_rgb 플래그를 통해 한 번 더 확인하는 것이 좋습니다.
            # (MultiViewSet의 이미지 로드 로직을 아는 것이 가장 정확하지만, 현재로서는 추정)
            
            # 여기서 RGB2BGR 변환을 시도해보고, 만약 이미 BGR이었다면 색이 뒤집힐 수 있습니다.
            # 가정: `image`가 RGB로 로드되어 들어온다.
            # 만약 이 가정이 틀렸다면, 아래 if 문을 주석 처리하고 `canvas_base = image.copy()`만 남겨두세요.
            try:
                # 간단한 테스트로 image[0,0,0]과 image[0,0,2]를 비교하여 BGR/RGB 추정 가능
                # 하지만 가장 확실한 것은 로드하는 소스(MultiViewSet)의 코드를 직접 확인하는 것입니다.
                # 임시로 RGB로 가정하고 변환을 시도합니다.
                # 이 부분이 문제를 일으키면 제거해야 합니다.
                canvas_base = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB -> BGR 변환 시도
            except cv2.error:
                # 이미 BGR이거나 다른 문제로 변환 실패 시 원본 사용
                canvas_base = image.copy()
        else:
            # 그레이스케일 또는 다른 채널 수의 이미지
            canvas_base = image.copy()

        canvas = canvas_base.copy() # 모든 그리기 작업은 이 BGR canvas 위에서 이루어집니다.


        # CSV 데이터가 변경되었는지 확인하고 AnnotationObject를 새로 로드
        
        if self.anno_file is not csv_data:
       
            self.load_annotations_from_csv(csv_data)
            self.selected_object = None
            self.set_transform_controls_enabled(False)
            self.update_transform_display()


        if self.annotation_objects:
            filtered_objects = [obj for obj in self.annotation_objects if (self.t_check.isChecked() and obj.row_data.get('usable', 'T') == 'T') or (self.f_check.isChecked() and obj.row_data.get('usable', 'T') == 'F')]

            for obj in filtered_objects:
                label_text, class_color_rgb = self.get_label_and_color(obj.row_data['main_class'], obj.row_data['middle_class'])

                # RGB -> BGR 변환 (OpenCV 그리기 함수는 BGR을 사용)
                base_color_bgr = (class_color_rgb[2], class_color_rgb[1], class_color_rgb[0])

                current_draw_color = base_color_bgr
                if self.edit_mode:
                    if obj.is_selected:
                        current_draw_color = (0, 255, 0)  # 선택된 객체는 초록색 (BGR)
                    elif obj.is_modified:
                        current_draw_color = (0, 255, 255)  # 수정된 객체는 노란색 (BGR)

                thickness = max(1, round(canvas.shape[1] / 640))
                selected_thickness = max(2, round(canvas.shape[1] / 500))
                draw_thickness = selected_thickness if (self.edit_mode and obj.is_selected) else thickness

                # Edit 모드에 따라 그리기 기준점을 분기
                if self.edit_mode:
                    oriented_bbox_points = obj.get_transformed_points()

                    original_center = np.mean(obj.original_points, axis=0).astype(int)
                    min_x_orig, min_y_orig = np.min(obj.original_points, axis=0).astype(int)
                    max_x_orig, max_y_orig = np.max(obj.original_points, axis=0).astype(int)

                    render_center = original_center
                    render_bbox_rect = (min_x_orig, min_y_orig, max_x_orig, max_y_orig)
                    render_id_pos = (original_center[0] + 8, original_center[1] - 8)
                    render_label_pos = (original_center[0] + 8, original_center[1] + 20)
                else:
                    oriented_bbox_points = obj.get_transformed_points()

                    min_x_transformed, min_y_transformed = np.min(oriented_bbox_points, axis=0).astype(int)
                    max_x_transformed, max_y_transformed = np.max(oriented_bbox_points, axis=0).astype(int)
                    render_center = np.mean(oriented_bbox_points, axis=0).astype(int)

                    render_bbox_rect = (min_x_transformed, min_y_transformed, max_x_transformed, max_y_transformed)
                    render_id_pos = (render_center[0] + 8, render_center[1] - 8)
                    render_label_pos = (render_center[0] + 8, render_center[1] + 20)

                # Oriented BBOX 그리기
                if 1 in self.checked_list:
                    cv2.polylines(canvas, [oriented_bbox_points.astype(np.int32)], True, current_draw_color, draw_thickness)

                # AABB (Axis-Aligned BBox) 그리기
                if 2 in self.checked_list:
                    cv2.rectangle(canvas, (render_bbox_rect[0], render_bbox_rect[1]),
                                  (render_bbox_rect[2], render_bbox_rect[3]), current_draw_color, thickness)

                # Center Point 그리기
                if 0 in self.checked_list:
                    cv2.circle(canvas, tuple(render_center), 3, current_draw_color, -1)

                font_scale = 0.5
                font_thickness = max(1, round(canvas.shape[1] / 1000))
                font = cv2.FONT_HERSHEY_SIMPLEX

                def draw_text_with_background(text, pos, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    bg_rect_pt1 = (pos[0], pos[1] - text_height - 4)
                    bg_rect_pt2 = (pos[0] + text_width, pos[1])
                    cv2.rectangle(canvas, bg_rect_pt1, bg_rect_pt2, bg_color, -1)
                    cv2.putText(canvas, text, (pos[0], pos[1] - 3), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                if 5 in self.checked_list:
                    obj_id_text = str(obj.id).replace("id_", "")
                    draw_text_with_background(obj_id_text, render_id_pos)

                if 3 in self.checked_list or 4 in self.checked_list:
                    draw_text_with_background(label_text, render_label_pos)

                if 6 in self.checked_list and obj.is_modified:  # Show Original Box
                    original_points_for_display = obj.original_points.astype(np.int32)
                    cv2.polylines(canvas, [original_points_for_display], True, (255, 0, 255), 1)  # 원본은 보라색 (BGR)

            # Old Data BBOX 그리기 로직 (고정된 회색 (128,128,128) BGR)
            # 여기서 old_obj는 딕셔너리 {'points': ...} 형태입니다.
            if 7 in self.checked_list and self.old_data_objects:
                for old_obj in self.old_data_objects:
                
                    cv2.polylines(canvas, [old_obj['points']], True, (128, 128, 128), 1)

        canvas_resized = cv2.resize(canvas, dsize=self.dsize, interpolation=cv2.INTER_AREA)

        # 최종적으로 QPixmap으로 변환하기 위해 set_scale_and_policy에 전달합니다.
        # set_scale_and_policy 내부에서 BGR -> RGB 변환이 이루어집니다.
        self.set_scale_and_policy(canvas_resized)
        preloaded = self.ds.get_preloaded_data_for_current_view()
        if not preloaded:
            canvas = np.full((self.dsize[1], self.dsize[0], 3), 240, dtype=np.uint8)
            self.lbl_img.setPixmap(QPixmap(q2n.array2qimage(canvas, normalize=False)))
            self.change_image_info()
            return


        
        
         


        canvas_resized = cv2.resize(canvas, dsize=self.dsize, interpolation=cv2.INTER_AREA)
        self.set_scale_and_policy(canvas_resized)

    def load_annotations_from_csv(self, csv_data):
        self.anno_file = csv_data 
        self.annotation_objects = []
        self.selected_object = None
        self.set_transform_controls_enabled(False)
        self.update_transform_display()

        if csv_data is None or csv_data.empty: return

        for idx, row in csv_data.iterrows():
            # AnnotationObject 생성자에 'row' (pandas Series)와 'self' (ArmaViewer 인스턴스)를 전달합니다.
            # AnnotationObject 내부에서 row_data를 파싱하고 변환 상태를 초기화합니다.
            self.annotation_objects.append(AnnotationObject(row, self))

    def set_transform_controls_enabled(self, enabled):
        self.transform_box.setEnabled(enabled)
        if not enabled and self.selected_object:
            self.selected_object.is_selected = False
            self.selected_object = None

    def adjust_transform(self, param_type, value):
        if self.selected_object is None or not self.edit_mode: return
        
        if param_type == 'tx': self.selected_object.translation[0] += value
        elif param_type == 'ty': self.selected_object.translation[1] += value
        elif param_type == 'sw': self.selected_object.scale[0] += value
        elif param_type == 'sh': self.selected_object.scale[1] += value
        elif param_type == 'angle': self.selected_object.rotation_angle += value
        
        self.selected_object.mark_as_modified()
        self.update_transform_display()
        self.render_refined_scene()

    def apply_transform_from_text_edit(self, param_type, text):
        if self.selected_object is None or not self.edit_mode: return
        
        try:
            value = float(text)
        except ValueError:
            self.update_transform_display()
            return
        
        if param_type == 'tx': self.selected_object.translation[0] = value
        elif param_type == 'ty': self.selected_object.translation[1] = value
        elif param_type == 'sw': self.selected_object.scale[0] = max(0.01, value)
        elif param_type == 'sh': self.selected_object.scale[1] = max(0.01, value)
        elif param_type == 'angle': self.selected_object.rotation_angle = value
        
        self.selected_object.mark_as_modified()
        self.update_transform_display()
        self.render_refined_scene()  

    def update_transform_display(self):
        if self.selected_object and self.edit_mode:
            self.tx_edit.setText(f"{self.selected_object.translation[0]:.2f}")
            self.ty_edit.setText(f"{self.selected_object.translation[1]:.2f}")
            self.sw_edit.setText(f"{self.selected_object.scale[0]:.2f}")
            self.sh_edit.setText(f"{self.selected_object.scale[1]:.2f}")
            self.rot_edit.setText(f"{self.selected_object.rotation_angle:.2f}")
            self.selected_id_label.setText(f"Selected ID: {self.selected_object.id}")
        else:
            self.tx_edit.setText("0.00"); self.ty_edit.setText("0.00")
            self.sw_edit.setText("1.00"); self.sh_edit.setText("1.00")
            self.rot_edit.setText("0.00")

    def mousePressEvent(self, event):
        if not (hasattr(self, 'ds') and self.ds) or not self.edit_mode: return

        if event.button() == Qt.LeftButton:
            pos_in_widget = self.lbl_img.mapFromGlobal(self.mapToGlobal(event.pos()))
            widget_rect = self.lbl_img.rect()

            if widget_rect.contains(pos_in_widget):
                preloaded = self.ds.get_preloaded_data_for_current_view()
                if not preloaded or 'image' not in preloaded: return

                orig_h, orig_w = preloaded['image'].shape[:2]
                click_x = pos_in_widget.x() * (orig_w / widget_rect.width())
                click_y = pos_in_widget.y() * (orig_h / widget_rect.height())
                click_point = (click_x, click_y)

                found_object = None
                for obj in reversed(self.annotation_objects):
                    if obj.check_selection(click_point):
                        found_object = obj
                        break
                
                if self.selected_object and self.selected_object is not found_object:
                    self.selected_object.is_selected = False
                
                self.selected_object = found_object
                
                if self.selected_object:
                    self.selected_object.is_selected = True
                    self.set_transform_controls_enabled(True)
                    self.update_transform_display()
                else:
                    self.set_transform_controls_enabled(False)
                    self.update_transform_display()

                self.render_refined_scene()
    
    def get_label_and_color(self, main_class, middle_class):
        # 'Middle' 체크박스의 상태를 가져옵니다.
        middle_checkbox_is_checked = self.mid_check.isChecked() if hasattr(self, 'mid_check') else False

        if middle_checkbox_is_checked:
            # Middle Class가 Main Class와 같으면 Main Class만 표시
            # 그렇지 않으면 "Main:Middle" 형식으로 표시
            label = main_class if main_class == middle_class else f'{main_class}:{middle_class}'
        else:
            # Middle Class 체크박스가 비활성화되어 있으면 Main Class만 표시
            label = main_class

        # ★★★ 색상 검색 로직 개선 ★★★
        # 1. 먼저 정확한 label (main:middle 또는 main)로 색상 검색 시도
        color_entry = self.FIXED_COLOR_STYLE.get(label)
        
        # 2. 정확한 label로 못 찾았으면, main_class만으로 다시 검색 시도 (폴백)
        if color_entry is None:
            color_entry = self.FIXED_COLOR_STYLE.get(main_class)
        
        # 3. 그래도 못 찾았으면, 기본 회색 (혹은 다른 적절한 색상)으로 설정
        if color_entry is None:
            color_rgb = (100, 100, 100) # 기본 회색
        else:
            color_rgb = color_entry[0] # 색상 정보만 가져옴

        return label, color_rgb

    def change_image_at_scene(self):
        if not hasattr(self, 'ds') or self.ds is None: return
        self.checkbox_toggle()
        was_edit_mode = self.edit_mode 
        self.selected_object = None
        self.create_multiview()
        self.change_image_at_view()
        self.change_indicator()
        if self.old_obox_check.isChecked():
           self.load_old_data_bbox_if_needed(True) # 체크되어 있으면 다시 로드
        else:
           self.load_old_data_bbox_if_needed(False) # 체크 안 되어 있으면 초기화
        
        if was_edit_mode:
            self.set_mode(1) # 이전에 Edit 모드였다면 다시 Edit 모드로 설정
        else:
            self.set_mode(0) # 이전에 View 모드였다면 View 모드로 설정

    def change_image_at_view(self):
        if not hasattr(self, 'ds') or self.ds is None: return
        
        was_edit_mode = self.edit_mode

        selected_object_id = None
        if self.selected_object:
            selected_object_id = self.selected_object.id
            self.selected_object.is_selected = False
        self.selected_object = None
        
        
        current_view_name = self.ds.get_view_name()
        try:
            if current_view_name.isdigit():
                current_view_int = int(current_view_name)
                if self.angle and current_view_int in self.angle:
                    clicked_angle_id = self.angle.index(current_view_int)
                    self.view_group.button(clicked_angle_id).setChecked(True)
        except (ValueError, IndexError):
            pass
            
        self.view_box.setTitle(f'View : {current_view_name}')
        self.change_image_info()
        self.create_legend()
        if self.old_obox_check.isChecked():
            self.load_old_data_bbox_if_needed(True) # 체크되어 있으면 다시 로드
        else:
            self.load_old_data_bbox_if_needed(False) # 체크 안 되어 있으면 초기화 (old_data_objects를 비움)
        
        
        if was_edit_mode:
            self.set_mode(1) # 이전에 Edit 모드였다면 다시 Edit 모드로 설정
        else:
            self.set_mode(0) # 이전에 View 모드였다면 View 모드로 설정
        
        if was_edit_mode and selected_object_id: # Edit 모드였고 이전에 선택된 ID가 있다면
            found_object = None
            for obj in self.annotation_objects:
                if obj.id == selected_object_id:
                    found_object = obj
                    break
            
            if found_object:
                self.selected_object = found_object
                self.selected_object.is_selected = True
                self.set_transform_controls_enabled(True) # 다시 활성화
                self.update_transform_display() # UI 업데이트 (ID 라벨, 변환 값 등)
            else:
                # 같은 ID의 객체를 새 View에서 찾지 못하면 선택 해제
                self.selected_object = None
                self.set_transform_controls_enabled(False) # 비활성화
                self.update_transform_display() # UI 초기화
        
        self.render_refined_scene()

    

    def change_res(self):
        try:
            w = int(self.pix_w_input.text())
            h = int(self.pix_h_input.text())
            if w <= 0 or h <= 0: raise ValueError("Width/height must be positive.")
            self.dsize = (w, h)
            self.render_refined_scene()
        except ValueError as e:
            QMessageBox.warning(self, "입력 오류", f"유효한 양수 값을 입력해주세요: {e}")

    def checkbox_toggle(self):
        # ★★★ 체크박스 리스트 업데이트 ★★★
        self.checked_list = [i for i, btn in enumerate(
            (self.center_check, self.obox_check, self.bbox_check, self.main_check, self.mid_check, self.id_check, self.show_original_box_check, self.old_obox_check)
        ) if btn.isChecked()]
        self.create_legend() # 체크박스 토글 시 레전드를 다시 그립니다.
        self.render_refined_scene() 

    def change_image_info(self):
        if not hasattr(self, 'ds') or self.ds is None:
            self.file_num_name.setText(f'# N/A | Scene: N/A | View: N/A | File: N/A')
            return

        idx = self.ds.get_scene_index() + 1
        name = self.ds.get_scene_name()
        current_view_name = self.ds.get_view_name()
        csv_path = self.ds.get_current_refined_csv_path()
        csv_name = csv_path.name if csv_path else "N/A"
        self.file_num_name.setText(f'#{idx} | Scene: {name} | View: {current_view_name} | File: {csv_name}')
       
    def create_legend(self):
        self.legend_widget.clear() # QListWidget의 내용을 지웁니다.

        if not self.annotation_objects: return
            
        # 현재 화면에 표시될 객체 (usable 필터링 적용)
        filtered_objects_for_legend = [obj for obj in self.annotation_objects if (self.t_check.isChecked() and obj.row_data.get('usable', 'T') == 'T') or (self.f_check.isChecked() and obj.row_data.get('usable', 'T') == 'F')]

        # 고유한 (label_text, color_rgb) 조합을 저장할 set
        unique_legend_entries = set()

        for obj in filtered_objects_for_legend:
            main_c = obj.row_data['main_class']
            middle_c = obj.row_data['middle_class']
            
            # get_label_and_color 함수를 호출하여 그림과 동일한 레이블 및 색상 가져오기
            legend_label, legend_color_rgb = self.get_label_and_color(main_c, middle_c)
            unique_legend_entries.add((legend_label, legend_color_rgb))

        if self.old_obox_check.isChecked() and self.old_data_objects:
        # Old Data BBOX는 고정된 회색 (128,128,128)으로 표시
            unique_legend_entries.add(('Old Data BBOX', (128, 128, 128))) # RGB 순서로 (R,G,B)

        # 레전드 항목 정렬 (레이블 텍스트 기준)
        sorted_legend_entries = sorted(list(unique_legend_entries), key=lambda x: x[0])

        for label_text, color_rgb in sorted_legend_entries:
            # 색상 샘플을 위한 QPixmap 생성
            color_pixmap = QPixmap(20, 10)
            color_pixmap.fill(QColor(*color_rgb))

            # 아이콘 레이블과 텍스트 레이블 생성
            color_label = QLabel()
            color_label.setPixmap(color_pixmap)
            text_label = QLabel(label_text)

            # 수평 레이아웃에 아이콘과 텍스트 배치
            hbox = QHBoxLayout()
            hbox.addWidget(color_label)
            hbox.addWidget(text_label)
            hbox.addStretch(1) # 텍스트를 왼쪽에 정렬

            # QListWidgetItem에 들어갈 위젯 생성 및 레이아웃 설정
            legend_item_widget = QWidget()
            legend_item_widget.setLayout(hbox)

            # QListWidget에 아이템 추가
            item = QListWidgetItem(self.legend_widget)
            item.setSizeHint(legend_item_widget.sizeHint()) # 위젯 크기에 맞춰 아이템 높이 설정
            self.legend_widget.addItem(item)
            self.legend_widget.setItemWidget(item, legend_item_widget)

    def create_multiview(self):
        for i in reversed(range(self.view_layout.count())):
            self.view_layout.itemAt(i).widget().deleteLater()
        self.view_box.setLayout(self.view_layout)
        
        if not hasattr(self, 'ds') or self.ds is None: return

        try:
            current_scene_path = self.ds.get_scene_path()
            if not current_scene_path or not os.path.isdir(current_scene_path): self.angle = []
            else: self.angle = sorted([int(d) for d in os.listdir(current_scene_path) if d.isdigit()])
        except Exception as e:
            print(f"Error listing view angles: {e}"); self.angle = []

        if not self.angle: return

        for idx, i in enumerate(self.angle):
            lbl = QRadioButton(str(i)); self.view_layout.addWidget(lbl); self.view_group.addButton(lbl, idx)
        
        current_view_name = self.ds.get_view_name()
        if current_view_name.isdigit():
            try:
                current_view_int = int(current_view_name)
                if current_view_int in self.angle:
                    clicked_angle_id = self.angle.index(current_view_int)
                    self.view_group.button(clicked_angle_id).setChecked(True)
            except ValueError: pass
        self.view_group.buttonClicked[int].connect(self.goto_view)

    def change_indicator(self):
        if not hasattr(self, 'ds') or self.ds is None: self.indicator.setText(f'{"|" * 45}'); return
        idx = self.ds.get_scene_index() + 1
        length = len(self.ds.get_scene_name_list())
        if length > 0: self.indicator.setText(f'{"|" * int(((idx / length) * 45))}')
        else: self.indicator.setText(f'{"|" * 45}')
    
    def set_scale_and_policy(self, canvas_):
        # OpenCV 결과는 보통 BGR. qimage2ndarray.array2qimage는 RGB 기대.
        # 따라서 여기서 최종적으로 QImage로 변환하기 전에 채널 순서를 확인하고 조정합니다.
        if canvas_.ndim == 3:
            if canvas_.shape[2] == 3:                # BGR 이미지인 경우 → RGB로 변환
                canvas_show = cv2.cvtColor(canvas_, cv2.COLOR_BGR2RGB)
            elif canvas_.shape[2] == 4:              # BGRA 이미지인 경우 → RGBA로 변환 (알파 채널이 있을 때)
                canvas_show = cv2.cvtColor(canvas_, cv2.COLOR_BGRA2RGBA)
            else: # 그 외 (알 수 없는 채널 수)
                canvas_show = canvas_
        else: # 그레이스케일 또는 단일 채널 이미지는 그대로 사용
            canvas_show = canvas_

        self.pixmap = QPixmap(q2n.array2qimage(canvas_show, normalize=False))
        self.lbl_img.setPixmap(self.pixmap)

    

    def goto_scene(self):
        if not hasattr(self, 'ds') or self.ds is None: return
        max_length = self.ds.get_max_name_length()
        scene_name = self.goto_input.text().strip().zfill(max_length)
        if scene_name not in self.ds.get_scene_name_list(): self.goto_input.setText('Not Found'); return
        
        self.ds.set_scene_name(scene_name)
        self.ds.preload_scene_data()
        self.ds.set_view_name(str(self.ds.base_view_idx))
        self.change_image_at_scene()

    def filter_non_supported_classes(self, anno_file):
        if anno_file is None or anno_file.empty: return pd.DataFrame()
        return anno_file[anno_file['main_class'].isin(self.supported_classes)].dropna(subset=['main_class', 'middle_class']).reset_index(drop=True)

    @util.scene_navigation_modified
    def goto_prev_scene(self):
        new_idx = self.ds.get_scene_index() - 1
        if new_idx < 0: new_idx = len(self.ds.get_scene_name_list()) - 1
        self.ds.set_scene_index(new_idx); self.ds.preload_scene_data(); self.change_image_at_scene()

    @util.scene_navigation_modified
    def goto_next_scene(self):
        new_idx = self.ds.get_scene_index() + 1
        if new_idx >= len(self.ds.get_scene_name_list()): new_idx = 0
        self.ds.set_scene_index(new_idx); self.ds.preload_scene_data(); self.change_image_at_scene()

    @util.scene_navigation_modified
    def goto_first_scene(self):
        self.ds.set_scene_index(0); self.ds.preload_scene_data(); self.change_image_at_scene()

    def goto_view(self, selected_id):
        if not hasattr(self, 'ds') or self.ds is None or not self.angle: return
        if 0 <= selected_id < len(self.angle): self.ds.set_view_name(str(self.angle[selected_id])); self.change_image_at_view()

    @util.view_navigation_modified
    def goto_prev_view(self, view_names, idx):
        if idx > 0: self.ds.set_view_name(str(view_names[idx - 1])); self.change_image_at_view()

    @util.view_navigation_modified
    def goto_next_view(self, view_names, idx):
        if idx < len(view_names) - 1: self.ds.set_view_name(str(view_names[idx + 1])); self.change_image_at_view()

    @util.view_navigation_modified
    def goto_base_view(self, view_names, idx):
        new_view_name = str(self.ds.base_view_idx)
        if self.ds.get_view_name() != new_view_name: self.ds.set_view_name(new_view_name); self.change_image_at_view()

    def save_png(self):
        if not hasattr(self, 'ds') or self.pixmap.isNull(): QMessageBox.warning(self, "경고", "표시된 이미지가 없습니다."); return
        scene_name = self.ds.get_scene_name(); view_name = self.ds.get_view_name()
        default_filename = f"AMOD_Viewer_{scene_name}_{view_name}.png"
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save File', default_filename, 'PNG(*.png)')
        if file_name: self.pixmap.save(file_name, 'png')

    def save_modified_annotations(self):
        modified_objects = [obj for obj in self.annotation_objects if obj.is_modified]
        if not modified_objects: 
            QMessageBox.information(self, "정보", "수정된 어노테이션이 없습니다."); 
            return

        original_csv_path = self.ds.get_current_refined_csv_path()
        if not original_csv_path: 
            QMessageBox.warning(self, "경고", "원본 어노테이션 파일 경로를 찾을 수 없습니다."); 
            return

        dir_name, file_stem = original_csv_path.parent, original_csv_path.stem
        new_filename = f"{file_stem}_modified.csv"
        save_path = dir_name / new_filename

        try:
            preloaded_data = self.ds.get_preloaded_data_for_current_view()
            if not preloaded_data or preloaded_data['csv'] is None: 
                QMessageBox.warning(self, "경고", "저장할 원본 데이터가 없습니다."); 
                return
            
            df_to_save = preloaded_data['csv'].copy()
            
            if 'id' in df_to_save.columns: 
                df_to_save.set_index('id', inplace=True)
            else: 
                QMessageBox.warning(self, "경고", "CSV 파일에 'id' 컬럼이 없어 저장이 불가합니다."); 
                return

            for obj in modified_objects:
                points = obj.get_transformed_points() # 이 좌표는 float 형태일 수 있습니다.
                obj_id = obj.id
                if obj_id in df_to_save.index:
                    points_flat = points.flatten()
                    
                    # ★★★ 이 부분을 수정하여 정수 형태로 저장합니다. ★★★
                    # 각 좌표를 int로 캐스팅하고, 반올림하여 가장 가까운 정수로 만듭니다.
                    df_to_save.loc[obj_id, ['x1','y1','x2','y2','x3','y3','x4','y4']] = [int(round(p)) for p in points_flat]
                    
                    new_center = np.mean(points, axis=0)
                    # ★★★ 중심 좌표도 정수 형태로 저장합니다. ★★★
                    df_to_save.loc[obj_id, ['cx', 'cy']] = [int(round(c)) for c in new_center]
                else: 
                    print(f"경고: DataFrame에서 ID '{obj_id}'를 찾을 수 없어 업데이트를 건너뜁니다.")
            
            df_to_save.reset_index(inplace=True)
            df_to_save.to_csv(save_path, index=False)
            QMessageBox.information(self, "성공", f"수정된 어노테이션 {len(modified_objects)}개가 다음 파일로 저장되었습니다:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"어노테이션 저장 중 오류가 발생했습니다:\n{e}")
    
    def create_report_dialog(self):
        if not hasattr(self, 'ds') or not self.ds.get_set_path(): QMessageBox.warning(self, "경고", "데이터셋이 로드되지 않았습니다."); return
        text, ok = QInputDialog.getMultiLineText(self, 'Report', "What's the issue?")
        if ok and text:
            report_dir = pth.join(self.ds.get_set_path(), 'reports')
            os.makedirs(report_dir, exist_ok=True)
            report_file_path = pth.join(report_dir, 'report_issues.csv')
            file_exists = os.path.exists(report_file_path)
            with open(report_file_path, 'a', encoding='utf-8') as f:
                if not file_exists or os.stat(report_file_path).st_size == 0: f.write("ScenePath,Timestamp,Issue\n")
                f.write(f'{self.ds.get_scene_path()},{datetime.now().strftime("%Y%m%d%H%M%S")},{text.replace(",", ";")}\n')
            QMessageBox.information(self, "성공", "이슈 리포트가 성공적으로 저장되었습니다.")

    def remark_sort(self):
        QMessageBox.information(self, "알림", "Remark Sort는 Meta CSV 기반이므로 현재 비활성화되었습니다.")

    def auto_plot(self):
        if not hasattr(self, 'ds') or self.ds is None or not self.angle:
            QMessageBox.warning(self, "경고", "자동 재생할 뷰 목록이 없습니다.")
            self.static_radio.setChecked(True)
            return

        self.auto_plot_index = 0
        self.auto_plot_timer.start(500)

    def auto_plot_step(self):
        if not hasattr(self, 'ds') or self.ds is None:
            self.auto_plot_timer.stop()
            self.static_radio.setChecked(True)
            return
            
        if self.auto_plot_index < len(self.angle):
            self.ds.set_view_name(str(self.angle[self.auto_plot_index]))
            self.change_image_at_view()
            self.auto_plot_index += 1
        else:
            self.auto_plot_index = 0
            self.ds.set_view_name(str(self.angle[self.auto_plot_index]))
            self.change_image_at_view()

    def keyPressEvent(self, e):
        if e.key() in self.key_map:
            self.key_map[e.key()]()

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) == 'src': os.chdir('..')
    app = QApplication(sys.argv)
    
    icon_path = 'figs/AMOD-Viewer-Icon.svg'
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    ex = ArmaViewer()
    sys.exit(app.exec_())
