"""GUI 主模块。
设计原则：
- 所有 side effect（Tk 实例、控件、监听线程）都封装在 run_gui() 中，
  import 本模块不会自动启动 UI，方便测试和被脚本引用。
- 摄像头帧循环在主线程通过 root.after 驱动，只有预测推理
  提交到后台线程池，避免阻塞 UI（Windows 上 Tkinter 非线程安全）。
- 训练相关按钮/自动训练全部走后台线程，绝不阻塞 Tkinter mainloop。
"""
import logging
import os
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import utils
from gaze_cursor_modes import CursorModeManager
from global_info import GlobalInfo
from user_config import load_gui_config, save_gui_config

logger = logging.getLogger(__name__)

# —— 视频循环参数 ——
_FRAME_INTERVAL_MS = 40            # 目标 ~25fps
_CAMERA_FAIL_TIMEOUT_SEC = 3.0     # 连续读帧失败超过此秒数则触发重连
_MAX_CAMERA_FAILS = int(_CAMERA_FAIL_TIMEOUT_SEC * 1000 / _FRAME_INTERVAL_MS)

# —— 视频显示的初始尺寸（会随窗口大小动态调整） ——
_INITIAL_VIDEO_W = 600
_INITIAL_VIDEO_H = 500

# —— 红点参数 ——
_RED_DOT_SIZE = 20
_RED_DOT_REFRESH_MS = 40           # 位置刷新间隔（≈25fps 跟随预测）


class RedDotOverlay:
    """持久的红点悬浮窗（全屏透明层 + 单个红点圆形）。
    - start(): 创建 Toplevel + Canvas + 圆形；启动位置刷新循环
    - stop() : 销毁窗口，停止刷新循环
    - 位置由 GlobalInfo.red_dot_x / red_dot_y 驱动，_tick 每 30ms 移动圆形
    - 幂等：重复 start/stop 无副作用
    """
    def __init__(self, root, icon_img=None):
        self._root = root
        self._icon_img = icon_img
        self._win = None       # tk.Toplevel
        self._canvas = None    # tk.Canvas
        self._dot_id = None    # canvas item id
        self._active = False   # 逻辑开关；stop 后 _tick 自动退出
        self._tick_scheduled = False

    def is_active(self):
        return self._active

    def start(self):
        if self._active:
            return
        try:
            self._win = tk.Toplevel(self._root)
            self._win.title("FollowMyGaze - 红点追踪")
            if self._icon_img is not None:
                try:
                    self._win.iconphoto(True, self._icon_img)
                except Exception:
                    pass
            self._win.attributes("-topmost", True)
            w = GlobalInfo.screen_width or self._win.winfo_screenwidth()
            h = GlobalInfo.screen_height or self._win.winfo_screenheight()
            self._win.geometry(f"{w}x{h}+0+0")
            bg_color = "white"
            alpha = 1.0
            self._win.attributes("-alpha", alpha)
            self._win.attributes("-fullscreen", True)
            self._win.protocol('WM_DELETE_WINDOW', self.stop)
            self._win.bind('<Escape>', lambda e: self.stop())
            self._win.focus_force()
            self._canvas = tk.Canvas(self._win, width=w, height=h,
                                      bg=bg_color, highlightthickness=0)
            self._canvas.pack()
            self._canvas.bind('<Button-1>', lambda e: self.stop())
            self._dot_id = self._canvas.create_oval(
                0, 0, _RED_DOT_SIZE, _RED_DOT_SIZE, fill="red", outline="")
            self._active = True
            self._schedule_tick()
            logger.info("RedDotOverlay started")
        except Exception:
            logger.exception("RedDotOverlay.start failed")
            self._cleanup()

    def stop(self):
        if not self._active and self._win is None:
            return
        self._active = False
        self._cleanup()
        logger.info("RedDotOverlay stopped")

    def _cleanup(self):
        try:
            if self._win is not None:
                self._win.destroy()
        except Exception:
            pass
        self._win = None
        self._canvas = None
        self._dot_id = None

    def _schedule_tick(self):
        if self._active and not self._tick_scheduled:
            self._tick_scheduled = True
            self._root.after(_RED_DOT_REFRESH_MS, self._tick)

    def _tick(self):
        self._tick_scheduled = False
        if not self._active or self._canvas is None or self._dot_id is None:
            return
        try:
            x = GlobalInfo.red_dot_x
            y = GlobalInfo.red_dot_y
            self._canvas.coords(self._dot_id,
                                x, y, x + _RED_DOT_SIZE, y + _RED_DOT_SIZE)
        except Exception:
            logger.exception("RedDotOverlay._tick failed")
        finally:
            self._schedule_tick()


class GazeApp:
    """所有 UI 状态封装到实例，避免模块级 global。
    仅由 run_gui() 构造一次。
    """
    def __init__(self):
        # ---------- 根窗口 ----------
        self.root = tk.Tk()
        self.root.title("FollowMyGaze")
        self.root.geometry("1200x1200+0+0")
        self.root.resizable(True, True)
        GlobalInfo.root = self.root

        # —— 设置窗口图标 ——
        self._app_icon_img = None
        try:
            from utils import resource_path
            icon_ico = resource_path("icon.ico")
            icon_png = resource_path("icon.png")
            if os.path.exists(icon_ico):
                self.root.iconbitmap(icon_ico)
            elif os.path.exists(icon_png):
                self._app_icon_img = ImageTk.PhotoImage(Image.open(icon_png))
                self.root.iconphoto(True, self._app_icon_img)
        except Exception:
            logger.exception("set window icon failed")

        # ---------- 后台线程池 ----------
        self._frame_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='frame')
        GlobalInfo.predict_executor = self._frame_executor

        # ---------- 视频区 ----------
        self.video_label = tk.Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._video_w = _INITIAL_VIDEO_W
        self._video_h = _INITIAL_VIDEO_H
        self.video_label.bind("<Configure>", self._on_video_resize)

        # ---------- 信息条 ----------
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=6)
        tk.Label(info_frame, text="提示信息：").grid(row=0, column=0)
        self.model_state_str = tk.StringVar(value="……")
        tk.Label(info_frame, textvariable=self.model_state_str,
                 width=16, anchor='w').grid(row=0, column=1)
        GlobalInfo.model_state_var = self.model_state_str

        self.sample_count_str = tk.StringVar(value="累计样本：本地0+本次0=0")
        tk.Label(info_frame, textvariable=self.sample_count_str,
                 width=32, anchor='w').grid(row=0, column=2, padx=(20, 0))
        GlobalInfo.sample_count_var = self.sample_count_str

        self.mode_status_str = tk.StringVar(value="当前模式：后台训练")
        tk.Label(info_frame, textvariable=self.mode_status_str,
                 width=18).grid(row=0, column=3, padx=(20, 0))

        # ---------- 模式单选 ----------
        self._build_mode_selector()

        # ---------- 红点悬浮层（持久）----------
        self.red_dot_overlay = RedDotOverlay(self.root, icon_img=self._app_icon_img)

        # ---------- 视线-鼠标交互模式管理器 ----------
        self.cursor_mode_manager = CursorModeManager()

        # ---------- 开关：红点显示 / 自动训练 ----------
        self._build_switches()

        # ---------- 训练按钮 + 进度条 ----------
        self._build_train_area()

        # ---------- 摄像头恢复计数 ----------
        self._camera_fail_count = 0

        # ---------- 关闭标志 ----------
        self._closing = False

        # —— 保存 main.py 中默认打开的摄像头索引 ——
        self._camera_opened_index = int(GlobalInfo.camera_index)

        # —— 加载 GUI 配置，初始化界面值 ——
        self._load_gui_config()

        # 关闭协议
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ================== 界面构造 ==================
    def _build_mode_selector(self):
        style = ttk.Style()
        style.configure("Big.TRadiobutton",
                        font=("Arial", 9), padding=5, indicatorsize=15)
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Label(frame, text="模式选择：").grid(row=0, column=0)
        self.which_mode = tk.StringVar(value='silent_train')
        GlobalInfo.mode_select = self.which_mode
        ttk.Radiobutton(frame, value="silent_train", variable=self.which_mode,
                        text="后台训练", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=1)
        ttk.Radiobutton(frame, value="move_cursor", variable=self.which_mode,
                        text="光标随动(Alt)", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=2)
        ttk.Radiobutton(frame, value="gaze_jump", variable=self.which_mode,
                        text="视线跳转", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=3)
        ttk.Radiobutton(frame, value="gaze_follow", variable=self.which_mode,
                        text="视线跟随", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=4)
        ttk.Radiobutton(frame, value="gaze_glide", variable=self.which_mode,
                        text="视线滑翔", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=5)

    def _build_switches(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=6)
        self.red_dot_btn = tk.Button(frame, text="显示红点追踪",
                                     width=16,
                                     command=self._on_red_dot_open)
        self.red_dot_btn.grid(row=0, column=0, padx=8)
        self.auto_train_var = tk.BooleanVar(value=bool(GlobalInfo.enable_auto_train))
        tk.Checkbutton(frame, text=f"自动训练（每 {GlobalInfo.auto_train_threshold} 样本）",
                       variable=self.auto_train_var,
                       command=self._on_auto_train_toggle
                       ).grid(row=0, column=1, padx=8)
        tk.Label(frame, text="摄像头：").grid(row=0, column=2, padx=(16, 4))
        self.camera_index_var = tk.IntVar(
            value=int(GlobalInfo.camera_index))
        self.camera_index_spin = tk.Spinbox(
            frame, from_=0, to=9, width=4, textvariable=self.camera_index_var,
            command=self._on_camera_index_changed)
        self.camera_index_spin.grid(row=0, column=3, padx=4)
        self.camera_index_hint = tk.Label(
            frame, text=f"当前：{int(GlobalInfo.camera_index)}",
            fg="green", width=10, anchor='w')
        self.camera_index_hint.grid(row=0, column=4, padx=4)

        # ─── 视线跳转 ───
        jump_frame = tk.LabelFrame(self.root, text="视线跳转", padx=6, pady=4)
        jump_frame.pack(fill='x', padx=8, pady=(0, 6))
        tk.Label(jump_frame, text="触发距离(px)：").grid(row=0, column=0, sticky='w')
        self.jump_threshold_var = tk.StringVar(
            value=str(int(GlobalInfo.gaze_jump_jump_threshold)))
        self.jump_threshold_var.trace_add(
            'write', lambda *_: self._on_jump_threshold_changed())
        self.jump_threshold_entry = tk.Entry(
            jump_frame, textvariable=self.jump_threshold_var,
            width=8, justify='right')
        self.jump_threshold_entry.grid(row=0, column=1, padx=6)
        tk.Label(jump_frame, text="px").grid(row=0, column=2)
        self.jump_threshold_hint = tk.Label(
            jump_frame, text=f"已生效：{int(GlobalInfo.gaze_jump_jump_threshold)} px",
            fg="green", width=20, anchor='w')
        self.jump_threshold_hint.grid(row=0, column=3, padx=(10, 0))
        tk.Label(jump_frame, text="冷却时间(ms)：").grid(row=1, column=0, sticky='w')
        self.jump_cooldown_var = tk.IntVar(
            value=int(GlobalInfo.gaze_jump_cooldown_ms))
        tk.Scale(jump_frame, from_=500, to=5000, resolution=100,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.jump_cooldown_var,
                 command=self._on_jump_cooldown_changed,
                 showvalue=False).grid(row=1, column=1, padx=6)
        self.jump_cooldown_hint = tk.Label(
            jump_frame, text=f"当前：{int(GlobalInfo.gaze_jump_cooldown_ms)} ms",
            fg="green", width=14, anchor='w')
        self.jump_cooldown_hint.grid(row=1, column=2, sticky='w')

        # ─── 视线跟随 ───
        follow_frame = tk.LabelFrame(self.root, text="视线跟随", padx=6, pady=4)
        follow_frame.pack(fill='x', padx=8, pady=(0, 6))
        tk.Label(follow_frame, text="操作后暂停(秒)：").grid(row=0, column=0, sticky='w')
        self.follow_idle_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_follow_idle_seconds))
        tk.Scale(follow_frame, from_=1.0, to=10.0, resolution=0.5,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.follow_idle_var,
                 command=self._on_follow_idle_changed,
                 showvalue=False).grid(row=0, column=1, padx=6)
        self.follow_idle_hint = tk.Label(
            follow_frame, text=f"当前：{float(GlobalInfo.gaze_follow_idle_seconds):.1f}s",
            fg="green", width=14, anchor='w')
        self.follow_idle_hint.grid(row=0, column=2, sticky='w')
        tk.Label(follow_frame, text="跟随顺滑度：").grid(row=1, column=0, sticky='w')
        self.follow_ease_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_follow_ease))
        tk.Scale(follow_frame, from_=0.10, to=0.80, resolution=0.05,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.follow_ease_var,
                 command=self._on_follow_ease_changed,
                 showvalue=False).grid(row=1, column=1, padx=6)
        self.follow_ease_hint = tk.Label(
            follow_frame,
            text=f"当前：{float(GlobalInfo.gaze_follow_ease):.2f}（小=顺滑）",
            fg="green", width=18, anchor='w')
        self.follow_ease_hint.grid(row=1, column=2, sticky='w')

        # ─── 视线滑翔 ───
        glide_frame = tk.LabelFrame(self.root, text="视线滑翔", padx=6, pady=4)
        glide_frame.pack(fill='x', padx=8, pady=(0, 6))
        tk.Label(glide_frame, text="加速倍数：").grid(row=0, column=0, sticky='w')
        self.glide_mul_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_glide_max_multiplier))
        self.glide_mul_scale = tk.Scale(
            glide_frame, from_=5.0, to=30.0, resolution=1.0,
            orient=tk.HORIZONTAL, length=240,
            variable=self.glide_mul_var,
            command=self._on_glide_mul_changed, showvalue=False)
        self.glide_mul_scale.grid(row=0, column=1, padx=6)
        self.glide_mul_hint = tk.Label(
            glide_frame,
            text=f"当前：{GlobalInfo.gaze_glide_max_multiplier:.1f}×",
            fg="green", width=14, anchor='w')
        self.glide_mul_hint.grid(row=0, column=2, padx=(6, 0))
        tk.Label(glide_frame, text="减速起始距离(px)：").grid(row=1, column=0, sticky='w')
        self.glide_near_var = tk.IntVar(
            value=int(GlobalInfo.gaze_glide_near_threshold))
        tk.Scale(glide_frame, from_=100, to=500, resolution=20,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.glide_near_var,
                 command=self._on_glide_near_changed,
                 showvalue=False).grid(row=1, column=1, padx=6)
        self.glide_near_hint = tk.Label(
            glide_frame, text=f"当前：{int(GlobalInfo.gaze_glide_near_threshold)} px",
            fg="green", width=14, anchor='w')
        self.glide_near_hint.grid(row=1, column=2, sticky='w')
        tk.Label(glide_frame, text="全速加速距离(px)：").grid(row=2, column=0, sticky='w')
        self.glide_far_var = tk.IntVar(
            value=int(GlobalInfo.gaze_glide_far_threshold))
        tk.Scale(glide_frame, from_=300, to=1000, resolution=50,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.glide_far_var,
                 command=self._on_glide_far_changed,
                 showvalue=False).grid(row=2, column=1, padx=6)
        self.glide_far_hint = tk.Label(
            glide_frame, text=f"当前：{int(GlobalInfo.gaze_glide_far_threshold)} px",
            fg="green", width=14, anchor='w')
        self.glide_far_hint.grid(row=2, column=2, sticky='w')
        tk.Label(glide_frame, text="减速陡峭度：").grid(row=3, column=0, sticky='w')
        self.glide_exp_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_glide_dist_exponent))
        tk.Scale(glide_frame, from_=1.0, to=4.0, resolution=0.5,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.glide_exp_var,
                 command=self._on_glide_exp_changed,
                 showvalue=False).grid(row=3, column=1, padx=6)
        self.glide_exp_hint = tk.Label(
            glide_frame,
            text=f"当前：{float(GlobalInfo.gaze_glide_dist_exponent):.1f}（大=早刹）",
            fg="green", width=18, anchor='w')
        self.glide_exp_hint.grid(row=3, column=2, sticky='w')

        # —— 加载 GUI 配置 ——
        self._load_gui_config_runtime()

    def _build_train_area(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)
        self.train_btn = tk.Button(
            frame, text="用全部数据训练",
            command=self._on_train_button)
        self.train_btn.grid(row=0, column=0, padx=8)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            frame, orient='horizontal', length=280, mode='determinate',
            maximum=100.0, variable=self.progress_var)
        self.progress_bar.grid(row=0, column=1, padx=8)
        self.progress_text = tk.StringVar(value="")
        tk.Label(frame, textvariable=self.progress_text, width=22
                 ).grid(row=0, column=2, padx=8)

    # ================== 配置加载 ==================
    def _load_gui_config(self):
        """初始化时加载 GUI 配置，设置界面初值。"""
        cfg = load_gui_config()
        if not cfg:
            return
        try:
            if 'auto_train' in cfg:
                val = bool(cfg['auto_train'])
                self.auto_train_var.set(val)
                GlobalInfo.enable_auto_train = val
        except Exception:
            logger.exception("restore auto_train failed")
        try:
            if 'jump_threshold' in cfg:
                v = int(cfg['jump_threshold'])
                self.jump_threshold_var.set(str(v))
                GlobalInfo.gaze_jump_jump_threshold = v
                self.jump_threshold_hint.config(text=f"已生效：{v} px")
        except Exception:
            logger.exception("restore jump_threshold failed")
        try:
            if 'jump_cooldown_ms' in cfg:
                v = int(cfg['jump_cooldown_ms'])
                self.jump_cooldown_var.set(v)
                GlobalInfo.gaze_jump_cooldown_ms = v
                self.jump_cooldown_hint.config(text=f"当前：{v} ms")
        except Exception:
            logger.exception("restore jump_cooldown failed")
        try:
            if 'follow_idle_seconds' in cfg:
                v = float(cfg['follow_idle_seconds'])
                self.follow_idle_var.set(v)
                GlobalInfo.gaze_follow_idle_seconds = v
                self.follow_idle_hint.config(text=f"当前：{v:.1f}s")
        except Exception:
            logger.exception("restore follow_idle failed")
        try:
            if 'follow_ease' in cfg:
                v = float(cfg['follow_ease'])
                self.follow_ease_var.set(v)
                GlobalInfo.gaze_follow_ease = v
                self.follow_ease_hint.config(text=f"当前：{v:.2f}（小=顺滑）")
        except Exception:
            logger.exception("restore follow_ease failed")
        try:
            if 'glide_multiplier' in cfg:
                v = float(cfg['glide_multiplier'])
                self.glide_mul_var.set(v)
                GlobalInfo.gaze_glide_max_multiplier = v
                self.glide_mul_hint.config(text=f"当前：{v:.1f}×")
        except Exception:
            logger.exception("restore glide_multiplier failed")
        try:
            if 'glide_near_threshold' in cfg:
                v = int(cfg['glide_near_threshold'])
                self.glide_near_var.set(v)
                GlobalInfo.gaze_glide_near_threshold = v
                self.glide_near_hint.config(text=f"当前：{v} px")
        except Exception:
            logger.exception("restore glide_near failed")
        try:
            if 'glide_far_threshold' in cfg:
                v = int(cfg['glide_far_threshold'])
                self.glide_far_var.set(v)
                GlobalInfo.gaze_glide_far_threshold = v
                self.glide_far_hint.config(text=f"当前：{v} px")
        except Exception:
            logger.exception("restore glide_far failed")
        try:
            if 'glide_dist_exponent' in cfg:
                v = float(cfg['glide_dist_exponent'])
                self.glide_exp_var.set(v)
                GlobalInfo.gaze_glide_dist_exponent = v
                self.glide_exp_hint.config(text=f"当前：{v:.1f}（大=早刹）")
        except Exception:
            logger.exception("restore glide_dist_exponent failed")
        try:
            if 'camera_index' in cfg:
                v = int(cfg['camera_index'])
                self.camera_index_var.set(v)
                GlobalInfo.camera_index = v
                self.camera_index_hint.config(text=f"当前：{v}")
        except Exception:
            logger.exception("restore camera_index failed")

    def _reload_camera_if_needed(self):
        """启动时，如果保存的 camera_index 与当前打开的不一致，则重新打开摄像头。"""
        try:
            current = int(GlobalInfo.camera_index)
            if self._camera_opened_index != current:
                logger.info("camera_index changed from %s to %s, reopening...",
                            self._camera_opened_index, current)
                self._reopen_camera()
        except Exception:
            logger.exception("reload camera if needed failed")

    def _load_gui_config_runtime(self):
        """运行时从 GlobalInfo 同步最新值到界面。"""
        try:
            self.auto_train_var.set(bool(GlobalInfo.enable_auto_train))
        except Exception:
            pass
        try:
            self.jump_threshold_var.set(str(int(GlobalInfo.gaze_jump_jump_threshold)))
            self.jump_threshold_hint.config(
                text=f"已生效：{int(GlobalInfo.gaze_jump_jump_threshold)} px")
        except Exception:
            pass
        try:
            self.jump_cooldown_var.set(int(GlobalInfo.gaze_jump_cooldown_ms))
            self.jump_cooldown_hint.config(
                text=f"当前：{int(GlobalInfo.gaze_jump_cooldown_ms)} ms")
        except Exception:
            pass
        try:
            self.follow_idle_var.set(float(GlobalInfo.gaze_follow_idle_seconds))
            self.follow_idle_hint.config(
                text=f"当前：{float(GlobalInfo.gaze_follow_idle_seconds):.1f}s")
        except Exception:
            pass
        try:
            self.follow_ease_var.set(float(GlobalInfo.gaze_follow_ease))
            self.follow_ease_hint.config(
                text=f"当前：{float(GlobalInfo.gaze_follow_ease):.2f}（小=顺滑）")
        except Exception:
            pass
        try:
            self.glide_mul_var.set(float(GlobalInfo.gaze_glide_max_multiplier))
            self.glide_mul_hint.config(
                text=f"当前：{float(GlobalInfo.gaze_glide_max_multiplier):.1f}×")
        except Exception:
            pass
        try:
            self.glide_near_var.set(int(GlobalInfo.gaze_glide_near_threshold))
            self.glide_near_hint.config(
                text=f"当前：{int(GlobalInfo.gaze_glide_near_threshold)} px")
        except Exception:
            pass
        try:
            self.glide_far_var.set(int(GlobalInfo.gaze_glide_far_threshold))
            self.glide_far_hint.config(
                text=f"当前：{int(GlobalInfo.gaze_glide_far_threshold)} px")
        except Exception:
            pass
        try:
            self.glide_exp_var.set(float(GlobalInfo.gaze_glide_dist_exponent))
            self.glide_exp_hint.config(
                text=f"当前：{float(GlobalInfo.gaze_glide_dist_exponent):.1f}（大=早刹）")
        except Exception:
            pass
        try:
            self.camera_index_var.set(int(GlobalInfo.camera_index))
            self.camera_index_hint.config(text=f"当前：{int(GlobalInfo.camera_index)}")
        except Exception:
            pass

    # ================== 事件回调 ==================
    def _on_mode_changed(self):
        val = self.which_mode.get()
        labels = {
            'silent_train': '后台训练',
            'move_cursor': '光标随动(Alt)',
            'gaze_jump': '视线跳转',
            'gaze_follow': '视线跟随',
            'gaze_glide': '视线滑翔',
        }
        self.mode_status_str.set(f"当前模式：{labels.get(val, val)}")

        if val == 'move_cursor':
            GlobalInfo.enable_move_cursor = True
        else:
            GlobalInfo.enable_move_cursor = False

        try:
            self.cursor_mode_manager.switch_to(val)
        except Exception:
            logger.exception("switch cursor mode failed")

    def _on_red_dot_open(self):
        if self.red_dot_overlay.is_active():
            self.red_dot_overlay.stop()
        else:
            self.red_dot_overlay.start()

    def _on_auto_train_toggle(self):
        GlobalInfo.enable_auto_train = bool(self.auto_train_var.get())
        self.model_state_str.set(
            "自动训练开启" if GlobalInfo.enable_auto_train else "自动训练关闭")
        save_gui_config(auto_train=GlobalInfo.enable_auto_train)

    def _on_jump_threshold_changed(self):
        raw = self.jump_threshold_var.get().strip()
        if not raw:
            self.jump_threshold_hint.config(
                text="输入为空", fg="gray")
            return
        try:
            v = int(raw)
            if v <= 0:
                raise ValueError("must be positive")
        except (TypeError, ValueError):
            self.jump_threshold_hint.config(
                text=f"已生效：{int(GlobalInfo.gaze_jump_jump_threshold)} px (非法输入)",
                fg="red")
            return
        GlobalInfo.gaze_jump_jump_threshold = v
        self.jump_threshold_hint.config(text=f"已生效：{v} px", fg="green")

    def _on_jump_cooldown_changed(self, *_):
        try:
            v = int(float(self.jump_cooldown_var.get()))
            GlobalInfo.gaze_jump_cooldown_ms = v
            self.jump_cooldown_hint.config(text=f"当前：{v} ms")
        except (TypeError, ValueError):
            pass

    def _on_follow_idle_changed(self, *_):
        try:
            v = float(self.follow_idle_var.get())
            GlobalInfo.gaze_follow_idle_seconds = v
            self.follow_idle_hint.config(text=f"当前：{v:.1f}s")
        except (TypeError, ValueError):
            pass

    def _on_follow_ease_changed(self, *_):
        try:
            v = float(self.follow_ease_var.get())
            GlobalInfo.gaze_follow_ease = v
            self.follow_ease_hint.config(text=f"当前：{v:.2f}（小=顺滑）")
        except (TypeError, ValueError):
            pass

    def _on_glide_mul_changed(self, *_):
        try:
            v = float(self.glide_mul_var.get())
            GlobalInfo.gaze_glide_max_multiplier = v
            self.glide_mul_hint.config(text=f"当前：{v:.1f}×")
        except (TypeError, ValueError):
            pass

    def _on_glide_near_changed(self, *_):
        try:
            v = int(float(self.glide_near_var.get()))
            GlobalInfo.gaze_glide_near_threshold = v
            self.glide_near_hint.config(text=f"当前：{v} px")
        except (TypeError, ValueError):
            pass

    def _on_glide_far_changed(self, *_):
        try:
            v = int(float(self.glide_far_var.get()))
            GlobalInfo.gaze_glide_far_threshold = v
            self.glide_far_hint.config(text=f"当前：{v} px")
        except (TypeError, ValueError):
            pass

    def _on_glide_exp_changed(self, *_):
        try:
            v = float(self.glide_exp_var.get())
            GlobalInfo.gaze_glide_dist_exponent = v
            self.glide_exp_hint.config(text=f"当前：{v:.1f}（大=早刹）")
        except (TypeError, ValueError):
            pass

    def _on_camera_index_changed(self):
        try:
            v = int(self.camera_index_var.get())
            if v < 0 or v > 9:
                raise ValueError("camera index out of range")
            GlobalInfo.camera_index = v
            self.camera_index_hint.config(text=f"当前：{v}")
            save_gui_config(camera_index=v)
            # 重新打开摄像头使用新索引
            self._reopen_camera()
        except (TypeError, ValueError):
            pass

    # ================== 训练 UI 回调 ==================
    def _training_ui_callback(self, event, *args):
        """train_model 的 UI 回调，通过 root.after 切回主线程更新界面。"""
        if event == "training_started":
            self.root.after(0, lambda: self.progress_text.set("训练中..."))
        elif event == "training_progress":
            epoch, loss = args[0], args[1]
            total = GlobalInfo.offline_training_epoch
            pct = min(100.0, epoch / max(1, total) * 100.0)
            self.root.after(0, lambda p=pct, e=epoch, t=total, l=loss: (
                self.progress_var.set(p),
                self.progress_text.set(f"epoch {e}/{t} loss={l:.2f}")
            ))
        elif event == "training_completed":
            self.root.after(0, lambda: (
                self.progress_var.set(100.0),
                self.progress_text.set("训练完成"),
                self.model_state_str.set("模型已更新")
            ))

    def _set_training_button_state(self, disabled: bool):
        state = 'disabled' if disabled else 'normal'
        self.root.after(0, lambda: self.train_btn.config(state=state))

    def _on_train_button(self):
        controller = GlobalInfo.train_and_predict_instance
        if controller is None:
            self.model_state_str.set("模型未初始化")
            return
        if getattr(controller, 'training', False):
            self.model_state_str.set("正在训练中，请稍候…")
            return

        prev_cb = controller.ui_callback
        controller.ui_callback = self._training_ui_callback

        def _run():
            try:
                self._set_training_button_state(disabled=True)
                self.progress_var.set(0.0)
                self.progress_text.set("训练中...")
                self.model_state_str.set("正在训练...")
                controller.train_model(is_online_mode=False)
                self.root.after(0, lambda: self.model_state_str.set("训练完成"))
                self.root.after(0, lambda: self.progress_text.set("训练完成"))
            except Exception as e:
                logger.exception("manual train failed")
                self.root.after(0, lambda: self.model_state_str.set(f"训练失败：{e}"))
                self.root.after(0, lambda: self.progress_text.set("训练失败"))
            finally:
                controller.ui_callback = prev_cb
                self._set_training_button_state(disabled=False)

        threading.Thread(target=_run, daemon=True, name="ManualTrainThread").start()

    def _on_video_resize(self, event):
        self._video_w = max(200, event.width - 10)
        self._video_h = max(150, event.height - 10)

    # ================== 摄像头帧循环 ==================
    def _schedule_frame(self):
        if self._closing:
            return
        self._process_frame()
        self.root.after(_FRAME_INTERVAL_MS, self._schedule_frame)

    def _process_frame(self):
        if self._closing:
            return
        try:
            ret, frame = GlobalInfo.video_steam.read()
            if not ret:
                self._camera_fail_count += 1
                if self._camera_fail_count >= _MAX_CAMERA_FAILS:
                    logger.warning("camera failed for %.1fs, trying reopen",
                                   _CAMERA_FAIL_TIMEOUT_SEC)
                    self._reopen_camera()
                return
            self._camera_fail_count = 0
            GlobalInfo.current_frame = frame

            features = GlobalInfo.gaze_feature_extractor.extract_features_from_image(frame)
            GlobalInfo.current_features = features

            if features is not None:
                self._frame_executor.submit(utils.predict_and_draw_pot, features)

            self.cursor_mode_manager.update()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            pil_image = utils.resize_image(pil_image, self._video_w, self._video_h)
            photo = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        except Exception:
            logger.exception("process_frame error")

    def _reopen_camera(self):
        try:
            GlobalInfo.video_steam.release()
        except Exception:
            pass
        try:
            GlobalInfo.video_steam = cv2.VideoCapture(int(GlobalInfo.camera_index))
            if GlobalInfo.video_steam.isOpened():
                GlobalInfo.video_steam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("camera reinitialized successfully")
                self.model_state_str.set("摄像头重连成功")
            else:
                logger.warning("camera reopen failed, will retry")
                self.model_state_str.set("摄像头重连失败...")
        except Exception:
            logger.exception("reopen camera failed")
            self.model_state_str.set("摄像头重连异常")

    # ================== 关闭 ==================
    def _on_close(self):
        """关闭窗口的完整流程（在独立线程中执行，避免阻塞主循环）。"""
        self._closing = True

        # 1. 先保存配置
        try:
            save_gui_config(
                auto_train=GlobalInfo.enable_auto_train,
                jump_threshold=int(GlobalInfo.gaze_jump_jump_threshold),
                jump_cooldown_ms=int(GlobalInfo.gaze_jump_cooldown_ms),
                follow_idle_seconds=float(GlobalInfo.gaze_follow_idle_seconds),
                follow_ease=float(GlobalInfo.gaze_follow_ease),
                glide_multiplier=float(GlobalInfo.gaze_glide_max_multiplier),
                glide_near_threshold=int(GlobalInfo.gaze_glide_near_threshold),
                glide_far_threshold=int(GlobalInfo.gaze_glide_far_threshold),
                glide_dist_exponent=float(GlobalInfo.gaze_glide_dist_exponent),
                camera_index=int(GlobalInfo.camera_index),
            )
        except Exception:
            logger.exception("save gui config failed")

        # 2. 停掉光标模式管理器
        try:
            self.cursor_mode_manager.stop_all()
        except Exception:
            logger.exception("stop cursor mode manager failed")

        # 3. 停掉红点悬浮层
        try:
            self.red_dot_overlay.stop()
        except Exception:
            logger.exception("stop red dot overlay failed")

        # 4. 等待训练结束（最多 30 秒）
        def _wait_for_training():
            try:
                controller = GlobalInfo.train_and_predict_instance
                if controller is not None and getattr(controller, 'training', False):
                    self.root.after(0, lambda: self.model_state_str.set(
                        "正在保存训练结果，请稍候..."))
                    deadline = time.time() + 30.0
                    while getattr(controller, 'training', False) and time.time() < deadline:
                        time.sleep(0.2)
                    if getattr(controller, 'training', False):
                        logger.warning("training still running after 30s timeout")
            except Exception:
                logger.exception("wait for training failed")

            # 5. 保存样本
            try:
                if GlobalInfo.train_data is not None:
                    GlobalInfo.train_data.save_current_sample_to_local()
            except Exception:
                logger.exception("save sample failed")

            # 6. 关闭 MediaPipe landmark 检测器
            try:
                extractor = getattr(GlobalInfo, 'gaze_feature_extractor', None)
                if extractor is not None:
                    landmark = getattr(extractor, 'face_mesh', None)
                    if landmark is not None:
                        landmark.close()
            except Exception:
                logger.exception("landmark close failed")

            # 7. 关闭视频流
            try:
                if GlobalInfo.video_steam is not None:
                    GlobalInfo.video_steam.release()
            except Exception:
                logger.exception("video stream release failed")

            # 8. 关闭线程池
            try:
                self._frame_executor.shutdown(wait=False)
            except Exception:
                logger.exception("executor shutdown failed")

            # 9. 销毁窗口
            try:
                self.root.after(0, self.root.destroy)
            except Exception:
                logger.exception("root destroy failed")

        threading.Thread(target=_wait_for_training, daemon=True,
                         name="CloseThread").start()

    # ================== 启动 ==================
    def run(self):
        # 如果保存的摄像头索引与默认打开的不一致，则重新打开
        self._reload_camera_if_needed()

        # 初始化 sample count 显示
        try:
            train_data = GlobalInfo.train_data
            if train_data is not None and hasattr(train_data, '_push_sample_count_to_ui'):
                train_data._push_sample_count_to_ui()
        except Exception:
            logger.exception("init sample count display failed")

        # 启动鼠标点击监听
        listening_click_thread = threading.Thread(
            target=utils.start_listening_click_dnn,
            daemon=True, name="ClickListener")
        listening_click_thread.start()

        # 启动键盘监听
        utils.move_cursor_when_press_key()

        # 启动帧循环
        self._schedule_frame()

        # 设置初始模式
        self._on_mode_changed()

        # 启动主循环
        self.root.mainloop()


def run_gui():
    app = GazeApp()
    app.run()


if __name__ == "__main__":
    run_gui()
