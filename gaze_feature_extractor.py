import cv2
import mediapipe as mp
import numpy as np
import math

from global_info import GlobalInfo


class GazeFeatureExtractor:
    def __init__(self):
        # 初始化MediaPipe面部检测和面部网格
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # 设置为1，只检测最大人脸
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 3D模型点（标准化坐标）
        self.model_points = np.array([
            [0.0, 0.0, 0.0],  # 鼻尖
            [0.0, -330.0, -65.0],  # 下巴
            [-225.0, 170.0, -135.0],  # 左眼角
            [225.0, 170.0, -135.0],  # 右眼角
            [-150.0, -150.0, -125.0],  # 左嘴角
            [150.0, -150.0, -125.0]  # 右嘴角
        ], dtype=np.float64)

        # 眼部关键点索引
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        self.LEFT_PUPIL_INDEX = 468
        self.RIGHT_PUPIL_INDEX = 473
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    def calculate_head_pose(self, image, face_landmarks):
        """计算头部姿态角"""
        height, width = image.shape[:2]

        # 获取2D图像点
        image_points = []
        for idx in [1, 152, 263, 33, 308, 78]:  # 对应特征点索引
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            image_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float64)

        # 相机参数估算
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # 将旋转向量转换为欧拉角
            rmat, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = self.rotation_matrix_to_euler_angles(rmat)

            return {
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'rotation_vector': rotation_vector,
                'translation_vector': translation_vector
            }
        return None

    def rotation_matrix_to_euler_angles(self, R):
        """将旋转矩阵转换为欧拉角"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # 转换为度数
        x = math.degrees(x)
        y = math.degrees(y)
        z = math.degrees(z)

        return x, y, z

    def euclidean_distance(self, point1, point2):
        """计算两点间欧氏距离"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def get_eye_features(self, face_landmarks, image_shape):
        """获取眼部特征点"""
        height, width = image_shape[:2]

        # 左眼特征
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            left_eye_points.append([x, y])

        # 右眼特征
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            right_eye_points.append([x, y])

        # 瞳孔中心
        left_pupil = face_landmarks.landmark[self.LEFT_PUPIL_INDEX]
        left_pupil_x, left_pupil_y = int(left_pupil.x * width), int(left_pupil.y * height)

        right_pupil = face_landmarks.landmark[self.RIGHT_PUPIL_INDEX]
        right_pupil_x, right_pupil_y = int(right_pupil.x * width), int(right_pupil.y * height)

        # 虹膜关键点
        left_iris_points = []
        for idx in self.LEFT_IRIS_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            left_iris_points.append([x, y])

        right_iris_points = []
        for idx in self.RIGHT_IRIS_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            right_iris_points.append([x, y])

        # 眼球中心（近似为眼睑中点）
        left_eye_center = (
            (left_eye_points[0][0] + left_eye_points[3][0]) // 2,
            (left_eye_points[0][1] + left_eye_points[3][1]) // 2
        )
        right_eye_center = (
            (right_eye_points[0][0] + right_eye_points[3][0]) // 2,
            (right_eye_points[0][1] + right_eye_points[3][1]) // 2
        )

        # 眼睛宽度
        left_eye_width = self.euclidean_distance(left_eye_points[0], left_eye_points[3])
        right_eye_width = self.euclidean_distance(right_eye_points[0], right_eye_points[3])


        return {
            'left_eye_points': left_eye_points,
            'right_eye_points': right_eye_points,
            'left_pupil_center': (left_pupil_x, left_pupil_y),
            'right_pupil_center': (right_pupil_x, right_pupil_y),
            'left_iris_points': left_iris_points,
            'right_iris_points': right_iris_points,
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center
        }

    def calculate_face_position_and_area(self, face_landmarks, image_shape):
        """计算人脸位置和面积"""
        height, width = image_shape[:2]

        # 获取所有人脸关键点的坐标
        x_coords = [lm.x * width for lm in face_landmarks.landmark]
        y_coords = [lm.y * height for lm in face_landmarks.landmark]

        # 计算边界框
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 计算面积（像素）
        area = (x_max - x_min) * (y_max - y_min)

        # 计算人脸中心位置
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # 计算相对于图像的比例位置
        relative_center_x = center_x / width
        relative_center_y = center_y / height

        # 计算人脸宽度和高度
        face_width = x_max - x_min
        face_height = y_max - y_min

        # 计算相对大小（相对于图像尺寸）
        relative_width = face_width / width
        relative_height = face_height / height

        return {
            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
            'area_pixels': int(area),
            'center': (int(center_x), int(center_y)),
            'relative_center': (relative_center_x, relative_center_y),
            'width': int(face_width),
            'height': int(face_height),
            'relative_size': (relative_width, relative_height)
        }

    def find_largest_face(self, multi_face_landmarks, image_shape):
        """找到最大的人脸"""
        if not multi_face_landmarks:
            return None

        height, width = image_shape[:2]
        largest_face = None
        max_area = 0

        for face_landmarks in multi_face_landmarks:
            # 计算人脸边界框
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]

            x_min, x_max = min(x_coords) * width, max(x_coords) * width
            y_min, y_max = min(y_coords) * height, max(y_coords) * height

            area = (x_max - x_min) * (y_max - y_min)

            if area > max_area:
                max_area = area
                largest_face = face_landmarks

        return largest_face

    def process_frame(self, image):
        """处理单帧图像，仅处理最大人脸"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        head_pose_data = None
        eye_features = None
        face_position_data = None

        if results.multi_face_landmarks:
            # 找到最大的人脸
            largest_face = self.find_largest_face(results.multi_face_landmarks, image.shape)

            if largest_face:
                # 计算头部姿态
                head_pose_data = self.calculate_head_pose(image, largest_face)

                # 获取眼部特征
                eye_features = self.get_eye_features(largest_face, image.shape)

                # 计算人脸位置和面积
                face_position_data = self.calculate_face_position_and_area(largest_face, image.shape)

                # 绘制面部网格
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=largest_face,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=largest_face,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())

        return image, head_pose_data, eye_features, face_position_data


    def extract_features_from_image(self, image):
        """从图像中提取全部特征（124 维，对齐 Mac 版）。

        特征布局（索引）：
          0..4   : 人脸 bbox（xmin, ymin, xmax, ymax, face_area）
          5..7   : 头部姿态角（pitch, yaw, roll）
          8..15  : 左眼虹膜特征（rel_x, rel_y, iris_x, iris_y, openness, ratio_x, ratio_y, iris_aspect）
          16..23 : 右眼虹膜特征（同上）
          24     : 双眼虹膜水平偏移差（辐辏信号）
          25..27 : 左眼 3D 视线方向向量（归一化）
          28..30 : 右眼 3D 视线方向向量（归一化）
          31     : 瞳孔间距 IPD（3D）
          32..123: 原始关键点坐标（16 左眼 + 16 右眼 + 5 左虹膜 + 5 右虹膜 + 4 其他）× 2(x,y) = 92
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        # 取第一个检测到的人脸（对齐 Mac 版行为）
        face_landmarks = results.multi_face_landmarks[0]
        lms = face_landmarks.landmark
        h, w = image.shape[:2]

        features = []

        # ===== A. 人脸位置（像素单位，idx 0..4）=====
        px_coords = np.array([(lm.x * w, lm.y * h) for lm in lms])
        xmin, ymin = np.min(px_coords, axis=0).astype(int)
        xmax, ymax = np.max(px_coords, axis=0).astype(int)
        face_area = int((ymax - ymin) * (xmax - xmin))
        features.extend([xmin, ymin, xmax, ymax, face_area])

        # ===== B. 姿态角（idx 5..7）=====
        head_pose = self.calculate_head_pose(image, face_landmarks)
        if head_pose:
            features.extend([head_pose['pitch'], head_pose['yaw'], head_pose['roll']])
        else:
            features.extend([0.0, 0.0, 0.0])

        # ===== C. 左眼虹膜特征（idx 8..15）=====
        iris_l = lms[468]       # 左虹膜中心
        eye_l_outer = lms[33]   # 左眼外角
        eye_l_inner = lms[133]  # 左眼内角
        eye_l_top = lms[159]    # 左眼上眼睑
        eye_l_bot = lms[145]    # 左眼下眼睑

        eye_l_width = abs(eye_l_inner.x - eye_l_outer.x)
        eye_l_height = abs(eye_l_top.y - eye_l_bot.y)
        eye_l_mid_x = (eye_l_inner.x + eye_l_outer.x) / 2
        eye_l_mid_y = (eye_l_inner.y + eye_l_outer.y) / 2
        eye_l_rel_x = iris_l.x - eye_l_mid_x
        eye_l_rel_y = iris_l.y - eye_l_mid_y
        eye_l_openness = eye_l_height / (eye_l_width + 1e-6)
        eye_l_ratio_x = (iris_l.x - eye_l_outer.x) / (eye_l_inner.x - eye_l_outer.x + 1e-6)
        eye_l_ratio_y = (iris_l.y - eye_l_top.y) / (eye_l_bot.y - eye_l_top.y + 1e-6)

        iris_l_left = lms[469]
        iris_l_top_pt = lms[470]
        iris_l_right = lms[471]
        iris_l_bot_pt = lms[472]
        iris_l_vis_w = abs(iris_l_left.x - iris_l_right.x)
        iris_l_vis_h = abs(iris_l_top_pt.y - iris_l_bot_pt.y)
        iris_l_aspect = iris_l_vis_h / (iris_l_vis_w + 1e-6)

        features.extend([eye_l_rel_x, eye_l_rel_y, iris_l.x, iris_l.y,
                         eye_l_openness, eye_l_ratio_x, eye_l_ratio_y, iris_l_aspect])

        # ===== D. 右眼虹膜特征（idx 16..23）=====
        iris_r = lms[473]
        eye_r_outer = lms[263]
        eye_r_inner = lms[362]
        eye_r_top = lms[386]
        eye_r_bot = lms[374]

        eye_r_width = abs(eye_r_inner.x - eye_r_outer.x)
        eye_r_height = abs(eye_r_top.y - eye_r_bot.y)
        eye_r_mid_x = (eye_r_inner.x + eye_r_outer.x) / 2
        eye_r_mid_y = (eye_r_inner.y + eye_r_outer.y) / 2
        eye_r_rel_x = iris_r.x - eye_r_mid_x
        eye_r_rel_y = iris_r.y - eye_r_mid_y
        eye_r_openness = eye_r_height / (eye_r_width + 1e-6)
        eye_r_ratio_x = (iris_r.x - eye_r_outer.x) / (eye_r_inner.x - eye_r_outer.x + 1e-6)
        eye_r_ratio_y = (iris_r.y - eye_r_top.y) / (eye_r_bot.y - eye_r_top.y + 1e-6)

        iris_r_left = lms[474]
        iris_r_top_pt = lms[475]
        iris_r_right = lms[476]
        iris_r_bot_pt = lms[477]
        iris_r_vis_w = abs(iris_r_left.x - iris_r_right.x)
        iris_r_vis_h = abs(iris_r_top_pt.y - iris_r_bot_pt.y)
        iris_r_aspect = iris_r_vis_h / (iris_r_vis_w + 1e-6)

        features.extend([eye_r_rel_x, eye_r_rel_y, iris_r.x, iris_r.y,
                         eye_r_openness, eye_r_ratio_x, eye_r_ratio_y, iris_r_aspect])

        # ===== E. 双眼虹膜水平偏移差（辐辏信号，idx 24）=====
        features.append(eye_l_rel_x - eye_r_rel_x)

        # ===== F. 左眼 3D 视线方向向量（idx 25..27）=====
        eye_l_center_3d = np.array([
            (eye_l_inner.x + eye_l_outer.x) / 2,
            (eye_l_inner.y + eye_l_outer.y) / 2,
            (eye_l_inner.z + eye_l_outer.z) / 2,
        ])
        iris_l_3d = np.array([iris_l.x, iris_l.y, iris_l.z])
        gaze_vec_l = iris_l_3d - eye_l_center_3d
        gaze_vec_l_norm = gaze_vec_l / (np.linalg.norm(gaze_vec_l) + 1e-6)
        features.extend(gaze_vec_l_norm.tolist())

        # ===== G. 右眼 3D 视线方向向量（idx 28..30）=====
        eye_r_center_3d = np.array([
            (eye_r_inner.x + eye_r_outer.x) / 2,
            (eye_r_inner.y + eye_r_outer.y) / 2,
            (eye_r_inner.z + eye_r_outer.z) / 2,
        ])
        iris_r_3d = np.array([iris_r.x, iris_r.y, iris_r.z])
        gaze_vec_r = iris_r_3d - eye_r_center_3d
        gaze_vec_r_norm = gaze_vec_r / (np.linalg.norm(gaze_vec_r) + 1e-6)
        features.extend(gaze_vec_r_norm.tolist())

        # ===== H. 瞳孔间距 IPD（3D，idx 31）=====
        ipd_3d = float(np.linalg.norm(iris_l_3d - iris_r_3d))
        features.append(ipd_3d)

        # ===== I. 原始关键点坐标（idx 32..123）=====
        for i in (self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES +
                  self.LEFT_IRIS_INDICES + self.RIGHT_IRIS_INDICES +
                  [168, 4, 8, 9]):
            features.append(lms[i].x)
            features.append(lms[i].y)

        return np.array(features, dtype=np.float32)


def main():
    estimator = GazeFeatureExtractor()

    # 打开摄像头（使用 GlobalInfo 中配置的索引）
    cap = cv2.VideoCapture(int(GlobalInfo.camera_index))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法获取摄像头画面")
            break

        # 处理图像
        processed_image, head_pose, eye_features, face_position = estimator.process_frame(image)

        # 显示头部姿态信息
        if head_pose:
            cv2.putText(processed_image, f"Yaw: {head_pose['yaw']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_image, f"Pitch: {head_pose['pitch']:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_image, f"Roll: {head_pose['roll']:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示眼部特征信息
        if eye_features:
            # 绘制左眼关键点
            for point in eye_features['left_eye_points']:
                cv2.circle(processed_image, tuple(point), 2, (255, 0, 0), -1)
            cv2.circle(processed_image, eye_features['left_pupil_center'], 3, (0, 0, 255), -1)

            # 绘制右眼关键点
            for point in eye_features['right_eye_points']:
                cv2.circle(processed_image, tuple(point), 2, (255, 0, 0), -1)
            cv2.circle(processed_image, eye_features['right_pupil_center'], 3, (0, 0, 255), -1)

            # 显示人脸位置和面积信息
            if face_position:
                center = face_position['center']
                area = face_position['area_pixels']
                rel_center = face_position['relative_center']
                rel_size = face_position['relative_size']

                # 绘制人脸中心点
                cv2.circle(processed_image, center, 5, (255, 255, 0), -1)

                # 显示详细信息
                cv2.putText(processed_image, f"Center: ({center[0]}, {center[1]})", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(processed_image, f"Relative Pos: ({rel_center[0]:.2f}, {rel_center[1]:.2f})", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(processed_image, f"Size: {face_position['width']}x{face_position['height']}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(processed_image, f"Relative Size: ({rel_size[0]:.2f}, {rel_size[1]:.2f})", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示图像
        cv2.imshow('Head Pose Estimation - Largest Face Only', processed_image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC键退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
