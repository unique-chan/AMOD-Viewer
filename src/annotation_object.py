# annotation_object.py

import numpy as np
import pandas as pd
from typing import Dict, Any

import numpy as np
import pandas as pd
import cv2 # cv2.pointPolygonTest 사용을 위해 추가
from typing import Dict, Any, Optional

class AnnotationObject:
    def __init__(self, row_data: pd.Series, parent_viewer: Optional[Any] = None):
        # row_data (pandas Series)는 CSV의 한 행 데이터를 담고 있습니다.
        self.row_data = row_data
        
        # 'id' 컬럼이 있으면 사용하고, 없으면 새로운 고유 ID를 생성합니다.
        self.id = row_data['id'] if 'id' in row_data else f"new_obj_{id(self)}"
        
        # 원본 8개 좌표 (x1, y1, ..., x4, y4)를 추출하여 NumPy 배열로 저장합니다.
        # .get()을 사용하여 컬럼이 없을 경우 기본값 0.0을 사용합니다.
        points_cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        self.original_points = np.array([row_data.get(col, 0.0) for col in points_cols], dtype=np.float32).reshape(4, 2)

        # 객체의 선택 및 수정 상태를 나타내는 플래그
        self.is_selected = False
        self.is_modified = False
        
        # 객체별 변환 상태를 초기화합니다.
        # CSV 데이터에 'tx', 'ty', 'sw', 'sh', 'angle' 컬럼이 있다면 그 값을 사용하고,
        # 없으면 기본값 (0.0, 1.0, 0.0)으로 초기화합니다.
        self.translation = np.array([row_data.get('tx', 0.0), row_data.get('ty', 0.0)], dtype=np.float32)
        self.scale = np.array([row_data.get('sw', 1.0), row_data.get('sh', 1.0)], dtype=np.float32)
        self.rotation_angle = float(row_data.get('angle', 0.0)) # 각도 (degree)

        # 부모 뷰어 인스턴스를 저장합니다 (필요한 경우 사용).
        self.parent_viewer = parent_viewer 

    def get_transformed_points(self):
        """현재 객체의 변환 상태를 반영하여 8개 좌표를 계산하여 반환합니다."""
        tx, ty = self.translation[0], self.translation[1]
        sw, sh = self.scale[0], self.scale[1]
        angle_rad = np.deg2rad(self.rotation_angle) # 각도를 라디안으로 변환

        # 1. 원본 좌표의 중심을 계산합니다.
        center_x, center_y = np.mean(self.original_points, axis=0)
        # 2. 중심을 (0,0)으로 이동합니다.
        temp_points = self.original_points - np.array([center_x, center_y])

        # 3. 스케일 변환 행렬을 적용합니다.
        scale_matrix = np.array([[sw, 0], [0, sh]])
        scaled_points = np.dot(temp_points, scale_matrix)

        # 4. 회전 변환 행렬을 적용합니다.
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_points = np.dot(scaled_points, rotation_matrix)

        # 5. 다시 원래 중심으로 이동하고, 최종 이동(translate) 값을 적용합니다.
        transformed_points = rotated_points + np.array([center_x + tx, center_y + ty])
        
        # 변환된 좌표는 float 형태로 유지하고 반환합니다.
        # int 변환은 그릴 때나 특정 연산에 필요할 때 수행하는 것이 좋습니다.
        return transformed_points 

    def reset_transform(self):
        """이 객체의 변환 상태(이동, 스케일, 회전)를 초기값으로 리셋합니다."""
        self.translation = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = np.array([1.0, 1.0], dtype=np.float32)
        self.rotation_angle = 0.0
        self.is_modified = False

    def mark_as_modified(self):
        """객체가 수정되었음을 표시합니다."""
        self.is_modified = True

    def check_selection(self, point):
        """주어진 점(point)이 현재 변환된 BBOX 내부에 있는지 확인합니다."""
        transformed_points = self.get_transformed_points()
        
        # cv2.pointPolygonTest는 정수 좌표를 기대하므로 변환된 좌표를 int로 캐스팅합니다.
        polygon = transformed_points.astype(np.int32)
        
        # cv2.pointPolygonTest를 사용하여 점이 폴리곤 내부에 있는지 확인합니다.
        # return 값은 내부에 있으면 양수, 외부에 있으면 음수, 선 위에 있으면 0입니다.
        result = cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False)
        return result >= 0 # 내부에 있거나 선 위에 있으면 True
        
    def apply_transform_to_original(self):
        """현재 변환된 상태를 original_points에 적용하고 변환을 리셋합니다."""
        # 현재 변환된 좌표를 새로운 원본 좌표로 설정합니다.
        self.original_points = self.get_transformed_points().astype(np.float32) 
        self.reset_transform() # 변환 상태는 초기화됩니다.
        self.is_modified = False # 수정 완료 후 수정 상태를 해제합니다.