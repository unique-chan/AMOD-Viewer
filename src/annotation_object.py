import numpy as np
import pandas as pd
import cv2
from typing import Optional, Any

class AnnotationObject:
    def __init__(self, row_data: pd.Series, parent_viewer: Optional[Any] = None):
        self.row_data = row_data
        

        raw_id_value = str(row_data['id']) 
        
        # 'id_' 프리픽스를 제거하고 숫자만 추출
    
        self.id = raw_id_value.split('_')[-1] 
        

        points_cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        loaded_coords = np.array([row_data.get(col, 0.0) for col in points_cols], dtype=np.float32).reshape(4, 2)
        
        rect_info = cv2.minAreaRect(loaded_coords)
        
        self.original_center = np.array(rect_info[0], dtype=np.float32)
        self.original_size = np.array(rect_info[1], dtype=np.float32) 
        
        self._initial_angle = rect_info[2]

        self.rotation_angle = 0.0 

        self.original_points = cv2.boxPoints(rect_info).astype(np.float32)

        self.parent_viewer = parent_viewer
        self.is_selected = False
        
        self.translation = np.array([row_data.get('tx', 0.0), row_data.get('ty', 0.0)], dtype=np.float32)
        self.scale = np.array([1.0, 1.0], dtype=np.float32)
        
        self.mark_as_modified()


    def get_transformed_points(self) -> np.ndarray:
        """
        현재 객체의 변환 상태(이동, 스케일, 회전)를 적용한 4개의 코너 포인트를 반환합니다.
        UI에서 시계 방향을 양수 각도로 간주하는 경우에 맞춰 각도를 조정합니다.
        """
        scaled_width = self.original_size[0] * self.scale[0]
        scaled_height = self.original_size[1] * self.scale[1]
        
        final_center = self.original_center + self.translation

        
        
        angle_for_cv2 = self._initial_angle + self.rotation_angle
        
        transformed_rect = (
            (final_center[0], final_center[1]), 
            (scaled_width, scaled_height), 
            angle_for_cv2 
        )
        
        transformed_points = cv2.boxPoints(transformed_rect)
        
        return transformed_points.astype(np.float32)

    def reset_transform(self):
        self.translation = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = np.array([1.0, 1.0], dtype=np.float32)
        self.rotation_angle = 0.0 
        self.mark_as_modified()

    def mark_as_modified(self):
        is_t_modified = not np.allclose(self.translation, [0.0, 0.0], atol=1e-5)
        is_s_modified = not np.allclose(self.scale, [1.0, 1.0], atol=1e-5)
        is_r_modified = not np.isclose(self.rotation_angle, 0.0, atol=1e-5)

        self.is_modified = is_t_modified or is_s_modified or is_r_modified

    def check_selection(self, point: tuple) -> bool:
        transformed_points = self.get_transformed_points()
        polygon = transformed_points.astype(np.int32)
        result = cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False)
        return result >= 0
        
    def apply_transform_to_original(self):
        current_transformed_points = self.get_transformed_points().astype(np.float32)
        self.original_points = current_transformed_points
        
        updated_rect_info = cv2.minAreaRect(self.original_points)
        self.original_center = np.array(updated_rect_info[0], dtype=np.float32)
        self.original_size = np.array(updated_rect_info[1], dtype=np.float32)
        
        self._initial_angle = updated_rect_info[2]
        self.reset_transform()