import threading
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

import utils
from global_info import GlobalInfo

root = tk.Tk()
root.title("看我眼神")
root.geometry("800x700")

# show video steam
video_label = tk.Label(root)
video_label.pack()

under_video_frame = tk.Frame(root)
under_video_frame.pack(pady=20)

# tips frame
model_state_frame = tk.Frame(under_video_frame, highlightthickness=1, highlightbackground="black", relief="raised")
model_state_frame.grid(row=0, column=0)
model_state_label = tk.Label(model_state_frame, text="提示信息：")
model_state_label.grid(row=0, column=0)

model_state_str = tk.StringVar()
model_state_str.set("……")
model_state_entry = tk.Label(model_state_frame, textvariable=model_state_str, width=5)
model_state_entry.grid(row=0, column=1)
GlobalInfo.model_state_var = model_state_str

total_samples_nums_int = tk.IntVar()
GlobalInfo.total_samples_num_int = total_samples_nums_int
total_samples_nums_pre = tk.Label(model_state_frame, text='累积样本数：')
total_samples_nums_pre.grid(row=1, column=0)
total_samples_nums_post = tk.Label(model_state_frame, textvariable=total_samples_nums_int)
total_samples_nums_post.grid(row=1, column=1)

camera_index_int = tk.IntVar()
camera_index_int.set(0)
GlobalInfo.camera_index_select = camera_index_int
camera_index_select_unit_label = tk.Label(model_state_frame, text='摄像头选择：')
camera_index_select_unit_label.grid(row=2, column=0)
camera_index_select_unit_entry = tk.Entry(model_state_frame, textvariable=camera_index_int, width=3)
camera_index_select_unit_entry.grid(row=2, column=1, pady=10)


samples_max_num_int = tk.IntVar()
samples_max_num_int.set(10)
GlobalInfo.samples_max_num = samples_max_num_int
samples_max_num_unit_label = tk.Label(model_state_frame, text='样本量上限(w)：')
samples_max_num_unit_label.grid(row=3, column=0)
samples_max_num_unit_entry = tk.Entry(model_state_frame, textvariable=samples_max_num_int, width=5)
samples_max_num_unit_entry.grid(row=3, column=1, pady=10)


# mode select
# 创建 Style 对象
style = ttk.Style()
# 配置 Radiobutton 的样式
style.configure("Big.TRadiobutton",
                font=("Arial", 7),  # 设置字体
                padding=1,  # 内边距（影响整体大小）
                indicatorsize=15)  # 圆形选择器的直径（像素）

mode_select_frame = tk.Frame(under_video_frame, highlightthickness=1, highlightbackground="black", relief="raised")
mode_select_frame.grid(row=0, column=1, padx=20)

mode_select_label = tk.Label(mode_select_frame, text="模式选择：")
mode_select_label.grid(row=0, column=0, ipadx=0)
which_mode = tk.StringVar(value='silent_train')
GlobalInfo.mode_select = which_mode
silent_train_button = ttk.Radiobutton(mode_select_frame, value="silent_train", variable=which_mode, text="后台训练",
                                      style="Big.TRadiobutton")
silent_train_button.grid(row=0, column=1, ipadx=0)

move_cursor_button = ttk.Radiobutton(mode_select_frame, value='move_cursor', variable=which_mode, text="光标随动",
                                     style="Big.TRadiobutton")
move_cursor_button.grid(row=0, column=2, ipadx=0)

red_dot_button = tk.Button(mode_select_frame, text='红点测试', command=lambda: utils.show_red_dot(root))
red_dot_button.grid(row=1, column=1, pady=10)


def train_model_with_all_data_func():
    GlobalInfo.train_and_predict_instance.train_model(is_online_mode=False)


train_model_with_all_data_button = tk.Button(mode_select_frame, text='用全部数据训练',
                                             command=train_model_with_all_data_func)
train_model_with_all_data_button.grid(row=2, column=1, pady=10)


def update_video_feed():
    ret, frame = GlobalInfo.video_steam.read()
    if ret:
        GlobalInfo.current_frame = frame
        features = GlobalInfo.gaze_feature_extractor.extract_features_from_image(frame)
        GlobalInfo.current_features = features
        camera_thread = threading.Thread(target=utils.predict_and_draw_pot, args=(features,))
        camera_thread.daemon = True
        camera_thread.start()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        pil_image = utils.resize_image(pil_image, 600, 500)
        photo = ImageTk.PhotoImage(image=pil_image)
        video_label.config(image=photo)
        video_label.image = photo
        root.after(100, update_video_feed)


listening_click_thread = threading.Thread(target=utils.start_listening_click_dnn)
listening_click_thread.daemon = True
listening_click_thread.start()

utils.move_cursor_when_press_key()


def when_close():
    GlobalInfo.train_data.save_current_sample_to_local()
    root.destroy()


root.protocol('WM_DELETE_WINDOW', when_close)


def run_gui():
    update_video_feed()
    root.mainloop()


if __name__ == "__main__":
    run_gui()
