"""视线-鼠标交互模式：
- GazeJumpController: 手动移动鼠标时，若光标距视线过远则一次性瞬移到视线位置
- GazeFollowController: 光标自动跟随视线，用户操作时让位
- GazeGlideController: 鼠标沿视线方向移动时按 cos + 距离加权加速滑翔
两个 controller 都可 start/stop，通过 GUI 的 mode 切换驱动。
使用 pynput 监听鼠标事件；用 pyautogui 控制鼠标位置。
关键：程序自身的 pyautogui.moveTo 也会触发 on_move 回调。我们用
`_last_programmatic_pos` 记录最近一次程序设置的坐标，并允许一个
`gaze_follow_user_move_pixel` 的容差来判定"用户手动"vs"程序回调"。
"""
import logging
import math
import threading
import time
import pyautogui
from pynput import mouse
from global_info import GlobalInfo

# 关掉 pyautogui 的 fail-safe（鼠标到屏幕角就抛异常）——本模块自己做边界保护。
pyautogui.FAILSAFE = False
# pyautogui 默认每次 moveTo 会 sleep 0.1s，串起来会明显延迟，禁掉
pyautogui.PAUSE = 0

logger = logging.getLogger(__name__)


def _gaze_xy():
    """当前视线预测点（可能为 0，需要调用方判断）"""
    return GlobalInfo.red_dot_x, GlobalInfo.red_dot_y


def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


# =============================================================
# Mode: gaze_jump — 距离过远时手动移动鼠标 → 瞬移到视线位置
# =============================================================
class GazeJumpController:
    """监听鼠标手动移动。当满足以下条件时，把光标瞬移到视线预测点：
    - 用户手动移动了鼠标（单次移动 >= min_user_move 像素）
    - 当前光标位置与视线预测点的距离 > jump_threshold
    - 距上一次瞬移已过了 cooldown_ms（防止视线抖动导致来回跳）
    效果：
    - 手轻推一下鼠标 → 直接跳到目光所在处
    - 到达后视线点附近的精细定位交给用户继续手动移动
    """
    def __init__(self):
        self._listener = None
        self._last_pos = None
        self._suppress_next = False
        self._last_jump_time = 0.0
        self._lock = threading.Lock()

    def start(self):
        if self._listener is not None:
            return
        self._last_pos = pyautogui.position()
        self._suppress_next = False
        self._last_jump_time = 0.0
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.daemon = True
        self._listener.start()
        logger.info("GazeJumpController started")

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("stop jump listener failed")
            self._listener = None
        logger.info("GazeJumpController stopped")

    def _on_move(self, x, y):
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        with self._lock:
            if self._last_pos is None:
                self._last_pos = (x, y)
                return
            # 过滤：这是我们自己 moveTo 触发的回调
            if self._suppress_next:
                self._suppress_next = False
                self._last_pos = (x, y)
                return
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            if not (math.isfinite(dx) and math.isfinite(dy)):
                self._last_pos = (x, y)
                return
            # 用户是否在明显地移动？太小的抖动忽略
            move_mag = math.hypot(dx, dy)
            if move_mag < GlobalInfo.gaze_jump_min_user_move:
                self._last_pos = (x, y)
                return
            # 冷却期内不再瞬移
            now = time.time()
            cooldown = GlobalInfo.gaze_jump_cooldown_ms / 1000.0
            if now - self._last_jump_time < cooldown:
                self._last_pos = (x, y)
                return
            # 需要有视线预测
            gx, gy = _gaze_xy()
            # 光标距视线距离 → 超阈值才瞬移
            distance = _dist(x, y, gx, gy)
            if not math.isfinite(distance):
                self._last_pos = (x, y)
                return
            if distance < GlobalInfo.gaze_jump_jump_threshold:
                self._last_pos = (x, y)
                return
            # 触发瞬移
            new_x, new_y = gx, gy
            self._suppress_next = True
            self._last_jump_time = now
            try:
                pyautogui.moveTo(new_x, new_y, _pause=False)
                logger.debug("gaze jump: (%d,%d) -> gaze (%d,%d), dist=%.1f",
                             int(x), int(y), new_x, new_y, distance)
            except Exception:
                logger.exception("jump moveTo failed")
            self._last_pos = (new_x, new_y)


# =============================================================
# Mode: gaze_follow — 光标自动跟随视线，用户操作时让位
# =============================================================
class GazeFollowController:
    """后台线程持续把光标移向视线点；检测到用户操作 → 暂停 idle_seconds。
    状态机：
      FOLLOWING   → 用户 on_move (delta > threshold) → PAUSED (记录 last_user_time)
      PAUSED      → 距 last_user_time > idle_seconds → FOLLOWING
    区分程序 vs 用户移动：用 _last_programmatic_pos 比对；
    on_move 的坐标如果和 _last_programmatic_pos 距离 <= user_move_pixel 视为程序移动。
    """
    def __init__(self):
        self._listener = None
        self._thread = None
        self._stop_event = threading.Event()
        self._paused = False
        self._last_user_time = 0.0
        self._last_programmatic_pos = None
        self._last_seen_pos = None
        self._lock = threading.Lock()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._paused = False
        self._last_user_time = 0.0
        self._last_programmatic_pos = None
        self._last_seen_pos = None
        self._listener = mouse.Listener(
            on_click=self._on_activity, on_scroll=self._on_activity)
        self._listener.daemon = True
        self._listener.start()
        self._thread = threading.Thread(
            target=self._follow_loop, daemon=True, name="GazeFollow")
        self._thread.start()
        logger.info("GazeFollowController started")

    def stop(self):
        self._stop_event.set()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("stop follow listener failed")
            self._listener = None
        logger.info("GazeFollowController stopped")

    def _on_activity(self, *args, **kwargs):
        with self._lock:
            self._paused = True
            self._last_user_time = time.time()

    def _follow_loop(self):
        interval = max(0.005, GlobalInfo.gaze_follow_step_interval_ms / 1000.0)
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("gaze follow tick failed")
            time.sleep(interval)

    def _tick(self):
        cur_x, cur_y = pyautogui.position()
        tol = GlobalInfo.gaze_follow_user_move_pixel
        with self._lock:
            prog = self._last_programmatic_pos
            user_moving = False
            if prog is not None:
                drift = _dist(cur_x, cur_y, prog[0], prog[1])
                if math.isfinite(drift) and drift > tol:
                    user_moving = True
            self._last_seen_pos = (cur_x, cur_y)
            if user_moving:
                self._paused = True
                self._last_user_time = time.time()
            paused = self._paused
            idle_elapsed = time.time() - self._last_user_time
        if paused:
            if idle_elapsed < GlobalInfo.gaze_follow_idle_seconds:
                with self._lock:
                    self._last_programmatic_pos = (cur_x, cur_y)
                return
            with self._lock:
                self._paused = False
        gx, gy = _gaze_xy()
        if not (math.isfinite(gx) and math.isfinite(gy)) or (gx == 0 and gy == 0):
            with self._lock:
                self._last_programmatic_pos = (cur_x, cur_y)
            return
        ease = max(0.0, min(1.0, GlobalInfo.gaze_follow_ease))
        nx = int(round(cur_x + (gx - cur_x) * ease))
        ny = int(round(cur_y + (gy - cur_y) * ease))
        if abs(nx - cur_x) < 1 and abs(ny - cur_y) < 1:
            with self._lock:
                self._last_programmatic_pos = (cur_x, cur_y)
                self._last_seen_pos = (cur_x, cur_y)
            return
        try:
            pyautogui.moveTo(nx, ny, _pause=False)
        except Exception:
            logger.exception("follow moveTo failed")
            with self._lock:
                self._last_programmatic_pos = (cur_x, cur_y)
            return
        with self._lock:
            self._last_programmatic_pos = (nx, ny)
            self._last_seen_pos = (nx, ny)


# =============================================================
# Mode: gaze_glide — 鼠标沿视线方向移动时加速滑翔
# =============================================================
class GazeGlideController:
    """当用户移动鼠标的方向与"当前光标位置→视线点"的方向一致时，放大位移。
    关键设计（修复"越过后被拉回"的 bug）：
      gaze_dir 从"当前光标位置"指向视线点，而不是从 stroke 起点指向视线点。
      这样一旦光标越过视线点，gaze_dir 立刻翻转，cos 变负，加速自动停止。
    速度系数 factor = 1 + (max_mul - 1) * dist_factor * dir_factor
      - dist_factor: 距视线距离越远越接近 1；<= near_threshold 时为 0
      - dir_factor:  cos(user_dir, gaze_dir) 归一化到 [0,1]
    自身回声过滤：用最近一次程序 moveTo 的目标位置 + 容差比对。
    """
    def __init__(self):
        self._listener = None
        self._last_pos = None
        self._last_move_time = 0.0
        self._last_programmatic = None
        self._lock = threading.Lock()

    def start(self):
        if self._listener is not None:
            return
        pos = pyautogui.position()
        self._last_pos = pos
        self._last_move_time = 0.0
        self._last_programmatic = None
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.daemon = True
        self._listener.start()
        logger.info("GazeGlideController started")

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("stop glide listener failed")
            self._listener = None
        logger.info("GazeGlideController stopped")

    def _dist_factor(self, distance):
        """距离因子：距目标越近越接近 0（柔和刹车），越远越接近 1。"""
        near = GlobalInfo.gaze_glide_near_threshold
        far = GlobalInfo.gaze_glide_far_threshold
        if distance <= near:
            return 0.0
        if distance >= far:
            return 1.0
        linear = (distance - near) / max(1e-6, (far - near))
        exponent = getattr(GlobalInfo, "gaze_glide_dist_exponent", 1.0)
        try:
            exponent = float(exponent)
        except Exception:
            exponent = 1.0
        if exponent <= 0:
            exponent = 1.0
        return max(0.0, min(1.0, linear ** exponent))

    def _dir_factor(self, mouse_dir, gaze_dir):
        """cos 加权，返回 [0, 1]。cos <= cos_threshold → 0；cos = 1 → 1。"""
        mx, my = mouse_dir
        gx, gy = gaze_dir
        mn = math.hypot(mx, my)
        gn = math.hypot(gx, gy)
        if mn <= 1e-6 or gn <= 1e-6:
            return 0.0
        cos_v = (mx * gx + my * gy) / (mn * gn)
        cos_v = max(-1.0, min(1.0, cos_v))
        cos_th = GlobalInfo.gaze_glide_cos_threshold
        if cos_v <= cos_th:
            return 0.0
        return cos_v

    def _on_move(self, x, y):
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        with self._lock:
            now = time.time()
            try:
                real_x, real_y = pyautogui.position()
            except Exception:
                real_x, real_y = x, y
            if not (math.isfinite(real_x) and math.isfinite(real_y)):
                real_x, real_y = x, y
            if self._last_pos is None:
                self._last_pos = (real_x, real_y)
                self._last_move_time = now
                return
            # 自身回声过滤
            prog = self._last_programmatic
            if prog is not None:
                if _dist(x, y, prog[0], prog[1]) <= max(2, GlobalInfo.gaze_glide_min_stroke_len):
                    self._last_pos = (real_x, real_y)
                    self._last_move_time = now
                    self._last_programmatic = None
                    return
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            if not (math.isfinite(dx) and math.isfinite(dy)):
                self._last_pos = (real_x, real_y)
                self._last_move_time = now
                return
            if dx == 0 and dy == 0:
                return
            self._last_move_time = now
            gx, gy = _gaze_xy()
            # 距离因子
            distance = _dist(real_x, real_y, gx, gy)
            df = self._dist_factor(distance)
            if df <= 0:
                self._last_pos = (real_x, real_y)
                return
            # 方向因子：用户移动方向 vs 光标→视线方向
            gaze_dx = gx - real_x
            gaze_dy = gy - real_y
            dir_f = self._dir_factor((dx, dy), (gaze_dx, gaze_dy))
            if dir_f <= 0:
                self._last_pos = (real_x, real_y)
                return
            # 放大位移
            max_mul = GlobalInfo.gaze_glide_max_multiplier
            factor = 1.0 + (max_mul - 1.0) * df * dir_f
            new_dx = dx * factor
            new_dy = dy * factor
            new_x = real_x + new_dx
            new_y = real_y + new_dy
            # 边界保护
            sw, sh = GlobalInfo.screen_width, GlobalInfo.screen_height
            new_x = max(0, min(sw - 1, int(new_x)))
            new_y = max(0, min(sh - 1, int(new_y)))
            self._suppress_next = True
            try:
                pyautogui.moveTo(new_x, new_y, _pause=False)
            except Exception:
                logger.exception("glide moveTo failed")
            self._last_pos = (real_x, real_y)
            self._last_programmatic = (new_x, new_y)

    _suppress_next = False


# =============================================================
# CursorModeManager: 统一管理三种模式的切换
# =============================================================
class CursorModeManager:
    """根据 GlobalInfo.mode_select 的值，启停对应的 controller。"""
    def __init__(self):
        self._jump = GazeJumpController()
        self._follow = GazeFollowController()
        self._glide = GazeGlideController()
        self._current_mode = None

    def update(self):
        """根据当前 mode_select 切换模式。由 GUI 帧循环调用。"""
        mode = None
        if GlobalInfo.mode_select is not None:
            try:
                mode = GlobalInfo.mode_select.get()
            except Exception:
                mode = None
        if mode == self._current_mode:
            return
        self.switch_to(mode)

    def switch_to(self, mode):
        """直接切换到指定模式。"""
        if mode == self._current_mode:
            return
        self._stop_all()
        if mode == 'gaze_jump':
            self._jump.start()
        elif mode == 'gaze_follow':
            self._follow.start()
        elif mode == 'gaze_glide':
            self._glide.start()
        self._current_mode = mode
        logger.info("CursorModeManager switched to: %s", mode)

    def _stop_all(self):
        self._jump.stop()
        self._follow.stop()
        self._glide.stop()

    def stop_all(self):
        self._stop_all()
        self._current_mode = None
