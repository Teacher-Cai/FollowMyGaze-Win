from pynput.keyboard import Key, Listener
import queue
import threading
import time

# 创建事件队列
event_queue = queue.Queue()

def on_press(key):
    event_queue.put(('press', key, time.time()))

def on_release(key):
    event_queue.put(('release', key, time.time()))

def process_events():
    """处理事件队列的线程函数"""
    while True:
        event_type, key, timestamp = event_queue.get()
        if event_type == 'press':
            print(f"按键按下: {key} (时间: {timestamp})")
        elif event_type == 'release':
            print(f"按键释放: {key} (时间: {timestamp})")
        event_queue.task_done()


# 启动事件处理线程
threading.Thread(target=process_events, daemon=True).start()

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
