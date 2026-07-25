import os

class GlobalInfo:
    video_steam = None
    screen_width = None
    screen_height = None
    train_and_predict_instance = None
    model_state_var = None
    sample_count_var = None  # tkinter StringVar，用于实时显示累计样本数
    predict_executor = None  # ThreadPoolExecutor，用于复用推理线程
    sample_upper_limit = 100000  # 10w
    train_data = None
    current_frame = None
    gaze_feature_extractor = None
    current_features = None
    online_training_batchSize = 2048
    online_training_epoch = 40
    offline_training_batchSize = 2048
    offline_training_epoch = 400
    # —— 在线自动训练配置 ——
    auto_train_threshold = 1024
    enable_auto_train = True
    red_dot_x = 0
    red_dot_y = 0
    mode_select = None
    enable_move_cursor = False
    move_cursor_x = 0
    move_cursor_y = 0
    show_red_dot_win = False
    # 红点显示独立开关
    show_red_dot_enabled = False
    # ================== 新增鼠标交互模式配置 ==================
    # —— 模式 gaze_jump ——
    gaze_jump_jump_threshold = 300
    gaze_jump_cooldown_ms = 2000
    gaze_jump_min_user_move = 5
    # —— 模式 gaze_follow ——
    gaze_follow_idle_seconds = 3.0
    gaze_follow_user_move_pixel = 3
    gaze_follow_step_interval_ms = 30
    gaze_follow_ease = 0.35
    # —— 模式 gaze_glide ——
    gaze_glide_max_multiplier = 5.0
    gaze_glide_near_threshold = 300
    gaze_glide_far_threshold = 500
    gaze_glide_cos_threshold = 0.6
    gaze_glide_stroke_reset_ms = 200
    gaze_glide_min_stroke_len = 20
    # —— 柔和刹车相关 ——
    gaze_glide_dist_exponent = 2.0
    gaze_glide_overshoot_anchor_ratio = 0.5
    root = None
    # 用户数据目录（项目本地）
    path_dir = os.path.dirname(os.path.abspath(__file__))
    # 摄像头索引（多摄像头用户可配置）
    camera_index = 0

