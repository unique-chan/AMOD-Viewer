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
        self.__set_path = Path('') # Path 객체로 변경
        self.__set_name = ''
        self.__scene_name = '' # 이 변수는 get_scene_name()으로 대체될 수 있습니다.
        self.__current_scene_idx = base_scene_idx
        self.__current_view_idx = base_view_idx
        self.__scene_path_list: List[Path] = [] # Path 객체 리스트
        self.__scene_name_list: List[str] = []
        self.base_view_idx = base_view_idx
        self.__refined_label_root = Path('') # Path 객체로 변경
        self.preloaded_scene_data = {}

    def get_set_path(self) -> Path: return self.__set_path
    def set_set_path(self, path: Path): self.__set_path = path
    def get_set_name(self) -> str: return self.__set_name
    def set_set_name(self, name: str): self.__set_name = name
    def get_scene_index(self) -> int: return self.__current_scene_idx
    def set_scene_index(self, idx: int): self.__current_scene_idx = idx
    def get_view_name(self) -> str: return self.__current_view_idx
    def set_view_name(self, name: str): self.__current_view_idx = name
    def get_max_name_length(self) -> int:
        return len(self.__scene_name_list[0]) if self.__scene_name_list else 0

    def get_scene_name(self) -> Optional[str]:
        if not self.__scene_name_list or self.__current_scene_idx >= len(self.__scene_name_list):
            return None
        return self.__scene_name_list[self.__current_scene_idx]

    def set_scene_name(self, name: str):
        if name in self.__scene_name_list:
            self.__current_scene_idx = self.__scene_name_list.index(name)

    def get_scene_name_list(self) -> List[str]:
        return self.__scene_name_list

    def get_scene_path(self) -> Optional[Path]: # 반환 타입을 Path로 변경
        if not self.__scene_path_list or self.__current_scene_idx >= len(self.__scene_path_list):
            return None
        return self.__scene_path_list[self.__current_scene_idx]

    def get_view_path(self) -> Optional[Path]: # 반환 타입을 Path로 변경
        # 뷰 폴더가 없어졌으므로, 이미지가 있는 씬 폴더 경로를 반환합니다.
        return self.get_scene_path()

    def get_ir_path(self) -> Optional[Path]: # 반환 타입을 Path로 변경
        # 현재 구조에는 IR 정보가 없으므로 None 반환
        return None 

    def get_scene_path_list(self) -> List[Path]: # 반환 타입을 Path 리스트로 변경
        return self.__scene_path_list

    def set_scene_path_list(self, path_list: List[Path]): # 인자 타입을 Path 리스트로 변경
        self.__scene_path_list = path_list

    # <<< --- 핵심 수정: 새로운 폴더 구조를 파싱하도록 완전히 변경 ---
    def set_path_and_name(self, path: str): # path 인자는 여전히 문자열
        """데이터셋의 루트 경로를 설정하고 씬 목록을 파싱합니다.
           path는 이제 /.../ROOT_FOLDER/train 또는 /.../ROOT_FOLDER/test 입니다."""
        
        selected_set_path = Path(path) # 사용자가 직접 입력한 'train' 또는 'test' 폴더 경로
        dataset_type = selected_set_path.name # "train" 또는 "test"
        
        # 1. 실제 이미지 씬 폴더 (e.g., /.../train/train_imgs)
        imgs_subfolder_name = f"{dataset_type}_imgs"
        imgs_path = selected_set_path / imgs_subfolder_name
        
        # 2. 라벨 폴더 (가장 높은 버전의 _labels_vX.Y 폴더를 찾음)
        labels_path: Optional[Path] = None
        
        # 'train_labels_v' 또는 'test_labels_v'로 시작하는 모든 폴더를 찾음
        candidate_label_folders = [
            d for d in os.listdir(selected_set_path)
            if d.startswith(f"{dataset_type}_labels_v") and (selected_set_path / d).is_dir()
        ]
        
        if candidate_label_folders:
            # 폴더 목록을 버전 순서대로 정렬 (예: v1.0, v1.5, v2.0 순)
            # 정렬 규칙: 숫자로 된 부분을 정확히 파싱하여 비교
            def parse_version(folder_name):
                match = re.search(r'v(\d+)\.(\d+)', folder_name)
                if match:
                    return int(match.group(1)), int(match.group(2))
                return 0, 0 # 매칭되지 않으면 낮은 버전으로 간주

            candidate_label_folders.sort(key=parse_version)
            
            # 정렬된 목록의 가장 마지막 항목(가장 높은 버전)을 선택
            highest_version_folder_name = candidate_label_folders[-1]
            labels_path = selected_set_path / highest_version_folder_name
            
        # 필수 폴더들이 존재하는지 확인
        if not imgs_path.is_dir():
            print(f"오류: 이미지 씬 폴더를 찾을 수 없습니다: {imgs_path}")
            self.__scene_name_list = []
            self.__scene_path_list = []
            self.set_refined_label_root(Path('')) # Path 객체로 빈 값 설정
            return
            
        if not labels_path or not labels_path.is_dir():
            print(f"오류: 라벨 폴더를 찾을 수 없거나 유효하지 않습니다. 예상 경로: {selected_set_path}/{dataset_type}_labels_vX.Y")
            self.__scene_name_list = []
            self.__scene_path_list = []
            self.set_refined_label_root(Path('')) # Path 객체로 빈 값 설정
            return

        # self.__set_path를 실제 이미지 씬 폴더로 설정 (여기서 씬 목록을 파싱)
        self.set_set_path(imgs_path) # Path 객체로 설정
        self.set_set_name(imgs_path.name) # "train_imgs" 또는 "test_imgs"

        # 이미지 씬 폴더(train_imgs/test_imgs)에서 씬 목록(숫자 이름 폴더)을 가져옴
        scene_names = sorted([d for d in os.listdir(imgs_path) if (imgs_path / d).is_dir() and d.isdigit()])

        self.__scene_name_list = scene_names
        self.__scene_path_list = [imgs_path / name for name in scene_names] # Path 객체 리스트
        self.set_refined_label_root(labels_path) # Path 객체로 설정
        print(f"감지된 라벨 폴더: {labels_path}")
        print(f"{len(self.__scene_name_list)}개의 씬을 찾았습니다.")
    # --- 핵심 수정 끝 ---

    def get_view_name_list(self) -> List[str]:
        """현재 씬 폴더의 이미지 파일명에서 뷰 이름(각도) 목록을 추출합니다."""
        current_scene_path = self.get_scene_path()
        if not current_scene_path or not current_scene_path.is_dir(): # is_dir() 사용
            return []

        view_angles = []
        # 정규표현식을 사용하여 파일명에서 씬 번호와 뷰 각도를 추출
        # 예: EO_0000_0.png -> 0, EO_0001_330.png -> 330
        scene_name = self.get_scene_name()
        if scene_name is None: return [] # 씬 이름이 없으면 반환
        
        pattern = re.compile(r"EO_" + re.escape(scene_name) + r"_(\d+)\.png$") # scene_name 이 특수문자를 포함할 경우를 대비해 re.escape
        for filename in os.listdir(current_scene_path):
            match = pattern.match(filename)
            if match:
                view_angles.append(int(match.group(1))) # 각도를 정수로 변환하여 추가

        # 숫자 순서대로 정렬 후 다시 문자열 리스트로 변환
        return [str(angle) for angle in sorted(view_angles)]
    
    def view_exists(self, view_name: str) -> bool:
        """현재 씬의 뷰 목록에 특정 뷰가 존재하는지 확인합니다."""
        return view_name in self.get_view_name_list()

    def update_best_view_idx(self):
        try:
            look_angles = self.get_view_name_list()
            if not look_angles:
                self.__current_view_idx = '0' # 기본값 설정
                return '0'

            median_idx = len(look_angles) // 2
            best_view = look_angles[median_idx]

            if self.__current_view_idx != best_view:
                print(f'current_view_idx (base_look_angle) {self.__current_view_idx} has been replaced with {best_view}')
                self.__current_view_idx = best_view
                self.base_view_idx = best_view
            return self.__current_view_idx
        except Exception as e:
            print(f'error at update_best_view_idx() in multiviewset.py: {e}')
            self.__current_view_idx = '0' # 오류 발생 시 기본값 설정
            return '0'

    def get_refined_label_root(self) -> Path: # 반환 타입을 Path로 변경
        return self.__refined_label_root

    def set_refined_label_root(self, path: Path): # 인자 타입을 Path로 변경
        if path and not path.exists(): # path.exists() 사용
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

        modified_path = refined_root_path / modified_csv_name # Path 객체 연산
        original_path = refined_root_path / original_csv_name # Path 객체 연산

        if modified_path.exists():
            return modified_path
        
        if original_path.exists():
            return original_path

        return None

    def get_refined_csv(self) -> Optional[pd.DataFrame]:
        csv_path = self.get_current_refined_csv_path()
        if csv_path:
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Refined CSV 파일 읽기 오류: {csv_path} - {e}")
        return None

    def get_refined_eo_path(self) -> Optional[Path]: # 반환 타입을 Path로 변경
        scene_name = self.get_scene_name()
        view_name = self.get_view_name()
        scene_path = self.get_scene_path()
        if scene_name is None or view_name is None or scene_path is None: return None

        # 새 파일명 형식: EO_0000_0.png
        target_img_name = f"EO_{scene_name}_{view_name}.png"
        target_path = scene_path / target_img_name # Path 객체 연산

        if target_path.exists(): # .exists() 사용
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

        if img_path and img_path.exists(): # img_path.exists() 사용
            image = cv2.imread(str(img_path)) # cv2.imread는 문자열 경로를 받음
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
        # 이 함수는 현재 로직에서 직접 사용되지 않지만, 호환성을 위해 남겨둡니다.
        view_names = self.get_view_name_list()
        # 뷰 '경로'는 없으므로 None으로 채운 리스트를 반환합니다.
        return [int(v) for v in view_names], [None] * len(view_names)