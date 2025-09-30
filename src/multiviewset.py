#
# multiviewset.py
# arma-rs-utils
#
# Created by Junggyun Oh on 04/04/2023.
# Copyright (c) 2023 Junggyun Oh All rights reserved.
#
import os
import os.path as pth
import pandas as pd
import numpy as np
import re
import glob
from pathlib import Path
from typing import List, Optional
import cv2


class MultiViewSet:
    """
    A class representing a dataset.
    """
    def __init__(self, base_scene_idx=0, base_view_idx='0'):
        self.__set_path = ''
        self.__set_name = ''
        self.__scene_name = ''
        self.__current_scene_idx = base_scene_idx
        self.__current_view_idx = base_view_idx
        self.__scene_path_list = []
        self.__scene_name_list = []
        self.__view_path_list = []
        self.base_view_idx = base_view_idx
        self.__refined_label_root = ''
        self.preloaded_scene_data = {} # ★★★ 씬 데이터를 캐시할 딕셔너리 추가 ★★★

    def get_set_path(self): return self.__set_path
    def set_set_path(self, path): self.__set_path = path
    def get_set_name(self): return self.__set_name
    def set_set_name(self, name): self.__set_name = name
    def get_scene_index(self): return self.__current_scene_idx
    def set_scene_index(self, idx): self.__current_scene_idx = idx
    def get_view_name(self): return self.__current_view_idx
    def set_view_name(self, name): self.__current_view_idx = name
    def get_max_name_length(self): return len(self.__scene_name_list[0])

    def get_scene_name(self):
        return self.__scene_name_list[self.__current_scene_idx]

    def set_scene_name(self, name):
        self.__current_scene_idx = self.__scene_name_list.index(name)

    def get_scene_name_list(self):
        return self.__scene_name_list

    def get_scene_path(self):
        return self.__scene_path_list[self.__current_scene_idx]

    def get_view_path(self):
        return pth.join(self.__scene_path_list[self.__current_scene_idx], self.__current_view_idx)

    def get_ir_path(self):
        view_path = self.get_view_path()
        return next((pth.join(view_path, i) for i in os.listdir(view_path) if 'IR' in i and i.endswith('png')), None)

    def get_scene_path_list(self):
        return self.__scene_path_list

    def set_scene_path_list(self, path_list):
        self.__scene_path_list = path_list

    # ★★★ 올바른 위치로 이동 및 수정된 함수 ★★★
    def set_path_and_name(self, path):
        """데이터셋의 루트 경로를 설정하고 씬 목록을 파싱합니다."""
        self.__set_path = path
        self.__set_name = pth.basename(path)

        base_list = []
        for f in os.listdir(path):
            if f.isdigit():
                base_list.append(f)
        
        base_list.sort()
        self.__scene_name_list = sorted(base_list)
        self.__scene_path_list = [os.path.join(path, f) for f in self.__scene_name_list]

        # Refined 라벨 폴더 (train/test) 자동 감지
        train_label_path = pth.join(path, "train_label_v1.5")
        test_label_path = pth.join(path, "test_label_v1.5")

        if pth.isdir(train_label_path):
            self.set_refined_label_root(train_label_path)
            print(f"감지된 라벨 폴더: {train_label_path}")
        elif pth.isdir(test_label_path):
            self.set_refined_label_root(test_label_path)
            print(f"감지된 라벨 폴더: {test_label_path}")
        else:
            self.set_refined_label_root('')
            print("경고: 'train_label_v1.5' 또는 'test_label_v1.5' 폴더를 찾을 수 없습니다.")

    def update_best_view_idx(self):
        try:
            look_angles = sorted([d for d in os.listdir(self.__scene_path_list[0]) if d.isdigit()])
            if not look_angles: return '0'
            
            median_idx = len(look_angles) // 2
            best_view = look_angles[median_idx]

            if self.__current_view_idx != best_view:
                print(f'current_view_idx (base_look_angle) {self.__current_view_idx} has been replaced with {best_view}')
                self.__current_view_idx = best_view
                self.base_view_idx = best_view
            return self.__current_view_idx
        except Exception as e:
            print(f'error at __update_best_view_idx() in multiviewset.py: {e}')
            return '0'

    # ★★★ 올바른 위치로 이동된 Refined CSV 및 Pre-loading 관련 함수들 ★★★
    def get_refined_label_root(self) -> str:
        return self.__refined_label_root

    def set_refined_label_root(self, path: str):
        if not pth.exists(path):
            print(f"경고: 설정하려는 Refined 라벨 루트 폴더가 존재하지 않습니다: {path}")
        self.__refined_label_root = path

    def get_current_refined_csv_path(self) -> Optional[Path]:
        scene_name = self.get_scene_name()
        view_name = self.get_view_name()
        target_csv_name = f"Refined-EO_{scene_name}_{view_name}.csv"
        refined_root_path = self.get_refined_label_root()
        if not refined_root_path: return None
        
        target_path = Path(refined_root_path) / target_csv_name
        
        if target_path.exists():
            return target_path
        return None

    def get_refined_csv(self) -> Optional[pd.DataFrame]:
        csv_path = self.get_current_refined_csv_path()
        if csv_path:
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Refined CSV 파일 읽기 오류: {csv_path} - {e}")
        return None

    def get_refined_eo_path(self) -> Optional[str]:
        scene_name = self.get_scene_name()
        view_name = self.get_view_name()
        target_img_name = f"EO_{scene_name}_{view_name}.png"
        view_path = self.get_view_path()
        target_path = pth.join(view_path, target_img_name)
        
        if pth.exists(target_path):
            return target_path
        return None

    def get_refined_data_for_view(self, scene_index: int, view_name: str) -> Optional[dict]:
        original_scene_idx = self.__current_scene_idx
        original_view_idx = self.__current_view_idx
        
        self.__current_scene_idx = scene_index
        self.__current_view_idx = view_name

        img_path = self.get_refined_eo_path()
        csv_data = self.get_refined_csv()
        
        self.__current_scene_idx = original_scene_idx
        self.__current_view_idx = original_view_idx

        if img_path and os.path.exists(img_path):
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return {'image': image, 'csv': csv_data}
        return None

    def preload_scene_data(self):
        scene_index = self.get_scene_index()
        scene_path = self.get_scene_path()
        
        self.preloaded_scene_data = {} 
        print(f"\nPre-loading data for Scene #{scene_index}...")

        view_names = sorted([d for d in os.listdir(scene_path) if d.isdigit()])
        for view_name in view_names:
            data = self.get_refined_data_for_view(scene_index, view_name)
            if data:
                self.preloaded_scene_data[view_name] = data
        print("Pre-loading complete.")

    def get_preloaded_data_for_current_view(self) -> Optional[dict]:
        return self.preloaded_scene_data.get(self.get_view_name())
    
    def get_view_name_path_list(self):
        current_scene_path = self.get_scene_path()
        if not current_scene_path: return [], []
        
        # 현재 씬 경로에서 숫자로 된 서브디렉토리를 찾아 뷰 이름으로 사용
        # (예: '0', '30', '330' 등)
        view_dirs = sorted([
            d for d in os.listdir(current_scene_path) 
            if pth.isdir(pth.join(current_scene_path, d)) and d.isdigit()
        ])
        view_paths = [pth.join(current_scene_path, d) for d in view_dirs]
        
        # 뷰 이름을 int로 변환하여 반환 (armaviewer.py의 self.angle과 호환)
        return [int(d) for d in view_dirs], view_paths
