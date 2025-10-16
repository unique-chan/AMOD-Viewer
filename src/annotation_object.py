# annotation_object.py (최종 수정: UI 시계 방향 = 양수 각도, 이미지도 시계 방향 회전)

import numpy as np
import pandas as pd
import cv2
from typing import Optional, Any

class AnnotationObject:
    def __init__(self, row_data: pd.Series, parent_viewer: Optional[Any] = None):
        self.row_data = row_data
        self.id = row_data.name 

        points_cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        loaded_coords = np.array([row_data.get(col, 0.0) for col in points_cols], dtype=np.float32).reshape(4, 2)
        
        rect_info = cv2.minAreaRect(loaded_coords)
        
        self.original_center = np.array(rect_info[0], dtype=np.float32)
        self.original_size = np.array(rect_info[1], dtype=np.float32) 
        
        # _initial_angle은 cv2.minAreaRect에서 반환된 객체 본연의 각도입니다.
        # 이 각도는 일반적으로 [-90, 0) 범위이며, x축에서 시계 방향으로 측정될 때 음수 값을 가집니다.
        # 우리는 이 각도를 cv2.boxPoints가 기대하는 방식으로 그대로 사용합니다.
        self._initial_angle = rect_info[2]

        # `self.rotation_angle`은 사용자가 추가한 회전량으로, 초기값은 0으로 설정됩니다.
        # UI에서 시계 방향을 양수로 간주하는 경우, 이 값도 시계 방향으로 증가합니다.
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

        # ★★★ 핵심 수정: cv2.boxPoints에 전달할 각도 조정 (UI 시계 방향 = 양수) ★★★
        # self._initial_angle: 객체 본연의 각도 (OpenCV 체계: 시계 방향으로 측정, 보통 음수 값)
        # self.rotation_angle: UI에서 입력된 사용자 추가 회전량 (시계 방향 = 양수)
        
        # UI의 '시계 방향 양수 증가'가 실제 물체의 '시계 방향 회전'과 일치하려면,
        # self.rotation_angle을 부호 변경 없이 _initial_angle에 더합니다.
        # cv2.boxPoints의 angle은 일반적으로 시계 방향으로 양수 값을 더하면 시계 방향으로 회전합니다.
        # (혹은 -90~0 범위에서 시계 방향으로 증가합니다.)
        
        # UI에서 시계 방향을 양수로 설정했다면, 이 값을 그대로 더하면 됩니다.
        # 만약 여전히 반대 방향으로 회전한다면, `self.rotation_angle`에 -1을 곱해야 합니다.
        
        # 현재는 `UI 시계 방향 양수 = 이미지 시계 방향 회전`을 목표로 함.
        # 만약 UI에서 시계방향 버튼 누를 때 `self.rotation_angle`이 양수로 증가한다면,
        # cv2.boxPoints의 각도도 시계 방향으로 증가해야 합니다.
        # cv2.minAreaRect가 반환하는 각도는 이미 시계 방향을 기준으로 정의되는 경향이 있으므로,
        # UI의 시계 방향 양수 값(self.rotation_angle)을 그대로 더해주는 것이 맞습니다.
        
        # 만약 계속 반대로 회전한다면, cv2.boxPoints의 각도 정의가 일반적인 수학적 정의(CCW=양수)와 같다는 뜻이므로,
        # self.rotation_angle에 -1을 곱해야 합니다.
        
        # 이 코드 블록은 `UI 시계 방향 양수 = 이미지 시계 방향 회전`을 가정하고 있습니다.
        # (즉, UI에서 시계 방향 화살표를 누를 때 `self.rotation_angle`이 양수로 증가해야 합니다.)
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