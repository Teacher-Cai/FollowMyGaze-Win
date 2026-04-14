import threading
import time
import tkinter as tk

import pyautogui
from PIL import Image
from pynput import keyboard
from pynput import mouse

from data_process_dnn import GazeDataset
from global_info import GlobalInfo
from threading import Event, Thread


def resize_image(image, max_width, max_height):
    width, height = image.size
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)

    return image


def show_red_dot(root=None):
    def toggle_dot():
        if hasattr(toggle_dot, "dot_id"):
            canvas.delete(toggle_dot.dot_id)  # 删除现有红点
            del toggle_dot.dot_id  # 清除属性
        else:
            # root_dot.attributes("-topmost", True)  # 置于所有窗口最上方
            x = GlobalInfo.red_dot_x  # 20 是红点直径
            y = GlobalInfo.red_dot_y  # 距离顶部 10 像素
            toggle_dot.dot_id = canvas.create_oval(x, y, x + 20, y + 20, fill="red")
            canvas.lift(toggle_dot.dot_id)  # 确保红点在最上层
        root_dot.after(100, toggle_dot)  # 1 秒后再次调用

    # 创建透明、无边框的顶层窗口
    root_dot = tk.Toplevel(root)
    # root_dot.overrideredirect(True)  # 去除窗口边框和标题栏
    root_dot.attributes("-topmost", True)  # 置于所有窗口最上方
    root_dot.attributes("-alpha", 0.8)  # 设置红点的透明度， 1不透明
    # root_dot.attributes("-transparentcolor", "white")  # 设置白色为透明色
    screen_width, screen_height = pyautogui.size()
    root_dot.geometry(f"{screen_width}x{screen_height}")

    # 创建画布并绘制红点
    canvas = tk.Canvas(root_dot, width=screen_width, height=screen_height, bg='white', highlightthickness=0)
    canvas.pack()
    toggle_dot()


def predict_and_draw_pot(frame):
    if not GlobalInfo.train_and_predict_instance.is_predict and frame is not None:
        x, y = GlobalInfo.train_and_predict_instance.predict_gaze(frame)
        GlobalInfo.red_dot_x = x
        GlobalInfo.red_dot_y = y
        mode_selected = GlobalInfo.mode_select.get()

        if mode_selected == 'silent_train':
            GlobalInfo.enable_move_cursor = False

        elif mode_selected == 'move_cursor':
            GlobalInfo.enable_move_cursor = True


def start_listening_click():
    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left and GlobalInfo.current_frame is not None:
            coords = [float(x), float(y)]
            GlobalInfo.train_data.save_sample(GlobalInfo.current_frame, coords)
            print(coords, "saving sample...")

            if len(GlobalInfo.train_data) >= 16:
                train_thread = threading.Thread(target=GlobalInfo.train_and_predict_instance.train_model)
                train_thread.daemon = False
                train_thread.start()

    listener = mouse.Listener(on_click=on_click)
    listener.start()


def start_listening_click_dnn():
    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left and GlobalInfo.current_features is not None:
            coords = [float(x), float(y)]
            GlobalInfo.train_data.save_sample(GlobalInfo.current_features, coords)

            # if len(GlobalInfo.train_data) >= GlobalInfo.online_training_batchSize:
            #     GlobalInfo.train_data.use_current_samples()
            #     train_thread = threading.Thread(target=GlobalInfo.train_and_predict_instance.train_model)
            #     train_thread.daemon = False
            #     train_thread.start()

    listener = mouse.Listener(on_click=on_click)
    listener.start()


Target_key = keyboard.Key.alt_l
stop_action_event = Event()
action_thread = None


def action_loop():
    while not stop_action_event.is_set():
        pyautogui.moveTo(GlobalInfo.red_dot_x, GlobalInfo.red_dot_y)
        time.sleep(0.1)


def move_cursor_when_press_key():
    def on_press(key):
        global action_thread
        if key == Target_key and GlobalInfo.enable_move_cursor\
                and (action_thread is None or not action_thread.is_alive()):
            stop_action_event.clear()
            action_thread = Thread(target=action_loop, daemon=True)
            action_thread.start()
            print("111")

    def on_release(key):
        if key == Target_key:
            stop_action_event.set()
            print("222")

    def start_keyboard_listener():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = threading.Thread(target=start_keyboard_listener)
    listener_thread.daemon = True  # 设置为守护线程（主线程退出时自动结束）
    listener_thread.start()


if __name__ == "__main__":
    move_cursor_when_press_key()
