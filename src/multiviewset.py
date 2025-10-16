import os
import os.path as pth
import pandas as pd
import numpy as np
import re # <<< re 모듈을 임포트합니다.
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
        # <<< self.__view_path_list 는 더 이상 필요 없으므로 삭제해도 됩니다.
        self.base_view_idx = base_view_idx
        self.__refined_label_root = ''
        self.preloaded_scene_data = {}

    def get_set_path(self): return self.__set_path
    def set_set_path(self, path): self.__set_path = path
    def get_set_name(self): return self.__set_name
    def set_set_name(self, name): self.__set_name = name
    def get_scene_index(self): return self.__current_scene_idx
    def set_scene_index(self, idx): self.__current_scene_idx = idx
    def get_view_name(self): return self.__current_view_idx
    def set_view_name(self, name): self.__current_view_idx = name
    def get_max_name_length(self):
        # <<< 씬 이름이 없을 경우를 대비한 방어 코드
        return len(self.__scene_name_list[0]) if self.__scene_name_list else 0

    def get_scene_name(self):
        # <<< 씬 목록이 비어있을 경우를 대비한 방어 코드
        if not self.__scene_name_list or self.__current_scene_idx >= len(self.__scene_name_list):
            return None
        return self.__scene_name_list[self.__current_scene_idx]

    def set_scene_name(self, name):
        if name in self.__scene_name_list:
            self.__current_scene_idx = self.__scene_name_list.index(name)

    def get_scene_name_list(self):
        return self.__scene_name_list

    def get_scene_path(self):
        # <<< 씬 경로가 없을 경우를 대비한 방어 코드
        if not self.__scene_path_list or self.__current_scene_idx >= len(self.__scene_path_list):
            return None
        return self.__scene_path_list[self.__current_scene_idx]

    def get_view_path(self):
        # <<< 뷰 폴더가 없어졌으므로, 이미지가 있는 씬 폴더 경로를 대신 반환합니다.
        return self.get_scene_path()

    def get_ir_path(self):
        # <<< IR 이미지 로직은 새 구조에 맞게 수정 필요 (일단 None 반환)
        # view_path = self.get_view_path()
        # return next((pth.join(view_path, i) for i in os.listdir(view_path) if 'IR' in i and i.endswith('png')), None)
        return None # 새 구조에서는 IR 파일 규칙을 확인해야 함

    def get_scene_path_list(self):
        return self.__scene_path_list

    def set_scene_path_list(self, path_list):
        self.__scene_path_list = path_list

    # <<< 핵심 수정: 새로운 폴더 구조를 파싱하도록 완전히 변경
    
    def set_path_and_name(self, path):
        """데이터셋의 루트 경로를 설정하고 씬 목록을 파싱합니다."""
        self.__set_path = path
        self.__set_name = pth.basename(path)

        train_path = pth.join(path, "train")
        imgs_path = pth.join(train_path, "train_imgs")
        
        # <<< --- 핵심 수정: 가장 높은 버전의 라벨 폴더를 선택합니다. ---
        labels_path = None
        if pth.isdir(train_path):
            # 'train_labels_v'로 시작하는 모든 폴더를 찾음
            candidate_folders = [
                d for d in os.listdir(train_path)
                if d.startswith("train_labels_v") and pth.isdir(pth.join(train_path, d))
            ]
            
            if candidate_folders:
                # 폴더 목록을 이름순으로 정렬 (예: v1.0, v1.5, v2.0 순)
                candidate_folders.sort()
                # 정렬된 목록의 가장 마지막 항목(가장 높은 버전)을 선택
                highest_version_folder = candidate_folders[-1]
                labels_path = pth.join(train_path, highest_version_folder)
        # --- 여기까지 ---

        # 필수 폴더들이 존재하는지 확인 (labels_path가 찾아졌는지 포함)
        if not all([pth.isdir(train_path), pth.isdir(imgs_path), labels_path]):
            print(f"오류: {path} 안에 train/train_imgs/train_labels_v... 폴더 구조가 없습니다.")
            self.__scene_name_list = []
            self.__scene_path_list = []
            self.set_refined_label_root('')
            return

        # train_imgs 폴더에서 씬 목록(숫자 이름 폴더)을 가져옴
        scene_names = sorted([d for d in os.listdir(imgs_path) if pth.isdir(pth.join(imgs_path, d)) and d.isdigit()])

        self.__scene_name_list = scene_names
        self.__scene_path_list = [pth.join(imgs_path, name) for name in scene_names]
        self.set_refined_label_root(labels_path)
        print(f"감지된 라벨 폴더: {labels_path}")
        print(f"{len(self.__scene_name_list)}개의 씬을 찾았습니다.")

    # <<< 핵심 수정: 파일명에서 뷰 각도를 파싱하도록 변경
    def get_view_name_list(self) -> List[str]:
        """현재 씬 폴더의 이미지 파일명에서 뷰 이름(각도) 목록을 추출합니다."""
        current_scene_path = self.get_scene_path()
        if not current_scene_path or not os.path.isdir(current_scene_path):
            return []

        view_angles = []
        # 정규표현식을 사용하여 파일명에서 씬 번호와 뷰 각도를 추출
        # 예: EO_0000_0.png -> 0, EO_0001_330.png -> 330
        pattern = re.compile(r"EO_" + self.get_scene_name() + r"_(\d+)\.png$")
        for filename in os.listdir(current_scene_path):
            match = pattern.match(filename)
            if match:
                view_angles.append(int(match.group(1))) # 각도를 정수로 변환하여 추가

        # 숫자 순서대로 정렬 후 다시 문자열 리스트로 변환
        return [str(angle) for angle in sorted(view_angles)]
    
    def view_exists(self, view_name: str) -> bool:
        """현재 씬의 뷰 목록에 특정 뷰가 존재하는지 확인합니다."""
        return view_name in self.get_view_name_list()

    # <<< 핵심 수정: 뷰 목록을 가져오는 방식 변경
    def update_best_view_idx(self):
        try:
            # 뷰 목록을 파일명에서 파싱해 옴
            look_angles = self.get_view_name_list()
            if not look_angles:
                return '0'

            # 중앙값을 기본 뷰로 설정
            median_idx = len(look_angles) // 2
            best_view = look_angles[median_idx]

            if self.__current_view_idx != best_view:
                print(f'current_view_idx (base_look_angle) {self.__current_view_idx} has been replaced with {best_view}')
                self.__current_view_idx = best_view
                self.base_view_idx = best_view
            return self.__current_view_idx
        except Exception as e:
            print(f'error at update_best_view_idx() in multiviewset.py: {e}')
            return '0'

    def get_refined_label_root(self) -> str:
        return self.__refined_label_root

    def set_refined_label_root(self, path: str):
        if path and not pth.exists(path):
            print(f"경고: 설정하려는 Refined 라벨 루트 폴더가 존재하지 않습니다: {path}")
        self.__refined_label_root = path

    
    def get_current_refined_csv_path(self) -> Optional[Path]:
        scene_name = self.get_scene_name()
        view_name = self.get_view_name()
        if scene_name is None or view_name is None: return None

        refined_root_path = self.get_refined_label_root()
        if not refined_root_path: return None

        # 원본 파일과 수정된 파일의 기본 이름을 생성
        base_filename = f"ANNOTATION-EO_{scene_name}_{view_name}"
        modified_csv_name = f"{base_filename}_modified.csv"
        original_csv_name = f"{base_filename}.csv"

        modified_path = Path(refined_root_path) / modified_csv_name
        original_path = Path(refined_root_path) / original_csv_name

        if modified_path.exists():
            return modified_path  # 수정된 파일이 있으면 그 경로를 반환
        
        # 수정된 파일이 없으면 원본 파일을 확인
        if original_path.exists():
            return original_path  # 원본 파일 경로를 반환

        return None # 둘 다 없으면 None을 반환

    def get_refined_csv(self) -> Optional[pd.DataFrame]:
        csv_path = self.get_current_refined_csv_path()
        if csv_path:
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Refined CSV 파일 읽기 오류: {csv_path} - {e}")
        return None

    # <<< 핵심 수정: 이미지 파일 경로 및 이름 규칙 변경
    def get_refined_eo_path(self) -> Optional[str]:
        scene_name = self.get_scene_name()
        view_name = self.get_view_name()
        scene_path = self.get_scene_path()
        if scene_name is None or view_name is None or scene_path is None: return None

        # 새 파일명 형식: EO_0000_0.png
        target_img_name = f"EO_{scene_name}_{view_name}.png"
        target_path = pth.join(scene_path, target_img_name)

        if pth.exists(target_path):
            return target_path
        return None

    # <<< 이 아래 함수들은 기존과 거의 동일합니다.
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
        if scene_index is None: return

        self.preloaded_scene_data = {}
        print(f"\nPre-loading data for Scene #{self.get_scene_name()}...")

        view_names = self.get_view_name_list()
        for view_name in view_names:
            data = self.get_refined_data_for_view(scene_index, view_name)
            if data:
                self.preloaded_scene_data[view_name] = data
        print("Pre-loading complete.")

    def get_preloaded_data_for_current_view(self) -> Optional[dict]:
        return self.preloaded_scene_data.get(self.get_view_name())

    def get_view_name_path_list(self):
        # <<< 이 함수는 현재 로직에서 직접 사용되지 않지만, 호환성을 위해 남겨둡니다.
        view_names = self.get_view_name_list()
        # 뷰 '경로'는 없으므로 None으로 채운 리스트를 반환합니다.
        return [int(v) for v in view_names], [None] * len(view_names)