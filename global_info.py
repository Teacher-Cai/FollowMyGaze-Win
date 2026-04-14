class GlobalInfo:
    video_steam = None
    screen_width = None
    screen_height = None
    train_and_predict_instance = None
    camera_index_select = 0
    samples_max_num = 100000

    # gui show
    model_state_var = None
    total_samples_num_int = None

    # data
    sample_upper_limit = 1000000
    train_data = None

    # features
    current_frame = None
    gaze_feature_extractor = None
    current_features = None

    # training setting
    online_training_batchSize = 256
    offline_training_batchSize = 256
    offline_training_epoch = 500
    online_training_epoch = 1
    red_dot_x = 0
    red_dot_y = 0
    mode_select = None
    enable_move_cursor = False
    move_cursor_x = 0
    move_cursor_y = 0

