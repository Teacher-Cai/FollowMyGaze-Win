# FollowMyGaze
> 用眼睛替手腕省下每天上千次的鼠标位移，降低腱鞘炎风险、提升工作效率。
> 最终目标：让视线成为你操作电脑的主力，逐步替代鼠标。
<p>
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows-lightgrey">
  <img alt="python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-red">
  <img alt="mediapipe" src="https://img.shields.io/badge/MediaPipe-FaceLandmarker-green">
  <img alt="status" src="https://img.shields.io/badge/status-experimental-orange">
</p>
FollowMyGaze 是一个跑在个人电脑上的**个性化视线追踪 + 视线-鼠标混合交互**工具。
它以普通摄像头作为输入，用 [MediaPipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) 提取人脸/眼部几何特征，训练一个基于 **Mixture-of-Experts + 物理先验门控** 的 PyTorch 小模型，把预测到的视线坐标用于红点显示、光标随动、视线跳转、视线跟随和视线滑翔等多种交互模式。

> 本仓库为 **Windows 版本**，从 [FollowMyGaze-Mac](https://github.com/Teacher-Cai/FollowMyGaze-Mac) 适配而来。保留了 Mac 版的 GazeMoE 模型架构与交互模式，针对 Windows 平台的 Tkinter 线程模型、摄像头 API 等做了适配。

## 项目定位
**本项目的核心目标是减少鼠标的操作。** 现代办公场景中，鼠标点击与拖动是极度高频、极度重复的动作，长期累积会带来腕管综合征、腱鞘炎等劳损风险，也拖慢了操作效率。FollowMyGaze 希望：
- **减少鼠标操作** —— 把最累的大范围光标位移从手腕转移到眼睛。
- **降低腱鞘炎 / 鼠标手风险** —— 减少手腕、手指的重复微动作和静态负荷。
- **提高工作效率** —— 视线擅长快速大范围转移，手只负责最后一厘米的精确定位。
- **最终替代鼠标** —— 长期目标是让视线成为操作电脑的主力输入方式。
**一个额外的副作用是：它同时是一个轻量的深度学习实验平台。** 因为"模型好不好，眼睛立刻能看出来"，从数据采集、特征工程、模型训练、shadow-model 部署到实时推理与交互层一应俱全，非常适合"改一行代码、立即用眼睛验证"的快速实验。
整个系统的核心特点：
- **强个性化**：你自己每次点击都会成为一条训练样本，模型持续在线学习你的头姿+眼动习惯。
- **强可解释性**：手工设计的 124 维几何特征 + 显式拆分头姿/虹膜两个专家，改哪儿动了哪儿一目了然。
- **强工程完备性**：从数据采集、特征工程、模型训练、shadow-model 部署、实时推理、到鼠标交互层，麻雀虽小五脏俱全。
> ⚠️ 当前项目定位是个人实验/原型工具，不是通用 SDK。模型效果强依赖于个人采样质量。
---
## 目录
- [项目定位](#项目定位)
- [项目背景](#项目背景)
  - [视角一：降低鼠标手 / 腱鞘炎风险](#视角一给长时间用电脑的人降低鼠标手--腱鞘炎风险)
  - [视角二：轻量的深度学习实验平台](#视角二作为轻量的深度学习实验平台)
- [功能特性](#功能特性)
- [与 Mac 版的差异](#与-mac-版的差异)
- [快速上手](#快速上手)
  - [环境要求](#环境要求)
  - [启动](#启动)
  - [基本使用流程](#基本使用流程)
- [交互模式](#交互模式)
- [模型设计思路](#模型设计思路)
- [训练与推理机制](#训练与推理机制)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [开发建议](#开发建议)
- [未来展望](#未来展望)
- [License](#license)
---
## 项目背景
FollowMyGaze 的定位不是"再造一个通用视线追踪 SDK"，而是同时服务于两类看起来不相关、但都受益于**"视线驱动 + 个性化学习"**的场景。理解这两个视角，能帮助你决定要不要把它跑起来、以及以什么姿势跑起来。
### 视角一：给长时间用电脑的人，降低鼠标手 / 腱鞘炎风险
现代办公场景中，鼠标操作已经是**极度高频、极度重复**的动作：
- 每天数千次点击
- 大量小幅度腕部微调（把光标从 A 拖到 B）
- 长时间保持"前臂前伸 + 手腕悬空 + 拇指外展"的静态姿势
这些动作长期累积会引发几种常见问题：
- **腕管综合征**（Carpal Tunnel Syndrome）：正中神经受压，出现手指麻木、握力下降
- **腱鞘炎**（De Quervain / trigger finger）：反复摩擦让肌腱鞘发炎，拇指、食指、腕部酸痛
- **鼠标肘 / 网球肘**：外上髁反复受力发炎
- **肩颈紧张**：长期悬肘操作让斜方肌和肩胛提肌持续紧绷
**FollowMyGaze 的价值不在于"取代鼠标"，而在于把每一次光标位移中最累的部分——大范围移动——从手腕转移到眼睛上。** 具体机制：
1. **消除大部分长距离拖动**
   看向哪儿，光标就跳/滑到哪儿。屏幕上从左上到右下的大跨度移动，原本需要抬腕 + 前臂配合完成，现在几乎不需要用手就能完成 90% 以上的距离。
2. **手只做"最后一厘米"精细定位**
   `视线滑翔` / `视线跳转` 模式都是**远距离视线接管、近距离交给手**。这正好利用了两者的优势：视线擅长快速大范围转移，手擅长小范围精确定位。
3. **减少手指/手腕微动作次数**
   同样是从代码编辑器切到浏览器点某个按钮，传统方式是"移动鼠标 → 点击"，可能需要 10+ 次小幅调整；用视线滑翔后，一次快速手推 + 视线聚焦即可到达目标附近。
4. **降低静态负荷时间**
   传统鼠标操作要求手时刻"待命"在鼠标上；FollowMyGaze 的多种模式允许你**手完全离开鼠标**去做别的事（阅读、思考），只在真需要点击时再触碰。
5. **保留传统鼠标的可控性**
   项目故意没做成"完全脱手视线控制"，而是提供**多种混合模式**（Alt 触发、视线跳转触发、视线滑翔加速）——你的手仍然处于主控地位，视线只做辅助放大器。这样既减负，又不牺牲精度和可预测性。
> 这不是医疗器械声明，仅是从操作学和人体工学角度的合理推断。真正有 RSI（重复性劳损）相关症状请就医。
**适合这些人群试用**：
- 每天键鼠时间超过 6 小时的程序员、设计师、运营
- 已经有轻微腕部不适、想主动降低使用强度的人
- 使用大屏幕 / 双显示器、鼠标跨屏距离远的用户
- 希望在阅读、看视频时减少手部占用的用户
### 视角二：作为轻量的深度学习实验平台
如果说健康层面的价值面向**每一位重度用户**，那么工程层面的价值就是面向**每一位对深度学习感兴趣的开发者**。
绝大多数深度学习入门项目停留在"训练一个 MNIST / CIFAR 分类器"，你能看到的只是准确率数字的变化。FollowMyGaze 提供了一个截然不同的学习闭环：**模型好不好，眼睛立刻能看出来**。
这带来了几个独特的教学 / 实验优势：
1. **即时的、可视的反馈**
   模型改一版、训练一次，打开红点追踪窗口，直接用眼睛判断：红点跟你视线的偏差有多大？在屏幕四角是否失灵？是否有明显抖动或漂移？——这比看 loss 曲线直观得多，也更容易触发直觉性思考。
2. **完整覆盖深度学习工程链路**
   项目里刚好包含了工程上"从数据到部署"的所有关键环节，且每一环都足够小、可读、可改：
   - **数据采集**：`utils.py` 的鼠标点击监听
   - **特征工程**：`gaze_feature_extractor.py` 的手工几何特征（124 维）
   - **模型结构**：`gaze_feature_based_model.py` 的 MoE + CrossNet + ResBlock
   - **损失设计**：`train_and_predict_dnn.py` 的多任务加权损失
   - **训练循环**：Adam + BN + Dropout，全量/在线两套配置
   - **在线学习**：`data_process_dnn.py` 的自动触发训练
   - **模型部署**：Shadow model + 原子替换 + EMA 平滑
   - **实时推理**：Tk 主线程 + 后台线程池 + 摄像头 30fps
   - **交互层**：多种鼠标控制模式验证模型效果
3. **每个改动都能立即验证**
   - 加一维新特征 → 训练 → 看红点是否更稳
   - 改门控阈值策略 → 训练 → 看头姿变化时的漂移
   - 换个损失权重 → 训练 → 看不同眼动状态下的误差分布
   - 调 EMA alpha → 感受平滑度和延迟的权衡
4. **数据规模适中，实验成本低**
   本地采集几千个样本，训练一次几十秒到几分钟。既不像 toy 数据集那样脱离真实工程，又不像大型项目那样一个实验要等一天。**"改代码 → 训练 → 用眼睛验证"整个循环控制在 5 分钟内**，非常适合快速迭代。
5. **暴露真实工程问题**
   这不是一个纯净的算法练习题，而是一个**必须解决工程细节**的完整系统：
   - 训练线程和推理线程如何共享模型？（→ Shadow model 模式）
   - BatchNorm 在推理时会不会污染 running stats？（→ eval/train 隔离）
   - 摄像头掉帧怎么办？（→ 自动重连 + 帧任务防堆积）
   - 事件流延迟导致鼠标被拉回怎么办？（→ 相对位移 vs 绝对坐标）
   - 训练时 UI 卡死怎么办？（→ 全部后台线程 + `root.after`）
   这些坑几乎是所有实时 AI 系统都会遇到的，值得亲手踩一遍。
6. **鼓励物理直觉驱动的模型设计**
   注视点 = 头姿贡献 + 虹膜偏转贡献。这个物理直觉直接决定了模型是 MoE 结构，门控用 `||rel||` 物理先验来路由。这种"从领域先验推导网络结构"的思路，是从 toy 例子里学不到的，但正是工业界模型设计的核心方法论。
**适合这样使用**：
- 深度学习新手：把 `SimpleDNN` 和 `GazeMoE` 拿出来对比，理解为什么加 residual、加 BN、加 gate
- 中级开发者：试试改特征、改结构、改损失，训练 → 用眼睛验证
- 想学习"实时 AI 系统工程"的人：整个 `gui.py` + `train_and_predict_dnn.py` 就是一个小型 online-learning 平台的完整实现
---
## 功能特性
围绕上面两个视角，项目提供以下能力：
- 🎥 **实时摄像头预览**：Tkinter GUI 显示摄像头画面。
- 🧠 **MediaPipe 人脸特征提取**：通过 MediaPipe FaceMesh 提取人脸、虹膜、头部姿态相关特征。
- 🖱️ **点击采样**：左键点击屏幕时，将当前视线特征与点击坐标保存为训练样本。
- 💾 **样本持久化**：样本保存到本地 `~/FollowMyGaze/samples/samples.pkl`。
- 🔢 **样本计数显示**：显示 `本地样本 + 本次会话样本 = 总样本数`。
- 🏋️ **手动训练**：一键"用全部数据训练"，后台跑全量样本。
- ⚙️ **自动训练**：每累计 `auto_train_threshold` 个新样本后自动持久化并后台全量训练。
- 🌗 **影子模型训练**：训练使用 shadow model，完成后原子替换推理模型，训练/推理线程互不干扰。
- 🔴 **红点追踪窗口**：显示当前预测视线位置，方便肉眼校验模型效果。
- ✋ **多种鼠标交互模式**：后台训练 / Alt 光标随动 / 视线跳转 / 视线跟随 / 视线滑翔。
- 🎛️ **参数持久化**：GUI 各交互参数在退出时保存到 `user_config.py` 指定的配置文件，下次启动自动恢复。
---
## 与 Mac 版的差异
| 方面 | Mac 版 | Windows 版 |
|------|--------|------------|
| 特征维度 | 124 维 | 124 维（对齐 Mac 版） |
| 特征提取文件 | `gaze_feature_extractor_mac.py` | `gaze_feature_extractor.py` |
| 系统权限 | 需要摄像头 + 辅助功能 + 输入监控 | 仅需摄像头（Windows 上 `pynput` 和 `pyautogui` 无需额外权限） |
| UI 线程模型 | 后台线程可更新 UI | 必须由主线程 `root.after` 驱动 UI |
| 打包方式 | `.app` (ad-hoc 签名) | `.exe` (PyInstaller) |
| GazeMoE 模型 | 相同架构 | 相同架构（完全相同） |
| 交互模式 | 全部 5 种 | 全部 5 种（相同实现） |
---
## 快速上手
> **TL;DR（3 步跑起来）**
> ```bash
> pip install opencv-python pyautogui mediapipe numpy pandas torch pillow pynput
> python main.py
> ```
> 然后看向屏幕点几下左键采样 → 点"用全部数据训练" → 打开红点追踪看效果。
### 环境要求
推荐环境：
- Windows 10/11
- Python 3.9+
- 摄像头
主要 Python 依赖：
```bash
pip install opencv-python pyautogui mediapipe numpy pandas torch pillow pynput
```
CUDA 用户建议按 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 安装匹配自己 GPU 的 PyTorch 版本。
### 启动
在项目根目录执行：
```bash
python main.py
```
启动后会出现主窗口：
- **上方**：摄像头画面
- **中部**：提示信息、样本计数、当前模式
- **模式选择**：后台训练 / 光标随动(Alt) / 视线跳转 / 视线跟随 / 视线滑翔
- **控制区**：红点追踪按钮、自动训练开关，以及按模式分组的参数面板：
  - **视线跳转**：触发距离、冷却时间
  - **视线跟随**：操作后暂停、跟随顺滑度
  - **视线滑翔**：加速倍数、减速起始距离、全速加速距离、减速陡峭度
- **底部**：手动训练按钮和训练进度条
> 参数面板的取值会在退出时自动持久化（`user_config.py`），下次启动自动恢复。
### 基本使用流程
**1. 启动程序**
```bash
python main.py
```
确保摄像头画面正常显示。
**2. 采集样本**
保持头部和眼睛自然状态，看向屏幕上的某个位置，然后用鼠标左键点击该位置。
每次左键点击会保存：
- 当前帧提取到的 gaze 特征
- 鼠标点击坐标 `(x, y)`
样本会先进入本次会话缓存，并在以下时机持久化：
- 每达到 `GlobalInfo.auto_train_threshold` 个新样本时自动保存
- 程序退出时保存未持久化样本
样本文件路径：
```text
~/FollowMyGaze/samples/samples.pkl
```
**3. 训练模型**
方式一：手动训练。点击 GUI 中的 "用全部数据训练"，程序会加载本地所有样本训练模型，并显示训练进度。
方式二：自动训练。默认开启：
```python
GlobalInfo.enable_auto_train = True
GlobalInfo.auto_train_threshold = 1024
```
每累计 1024 个新样本会触发一次后台全量训练，使用 `online_training_epoch` 作为 epoch 数。训练完成后的模型保存到：
```text
~/FollowMyGaze/gaze_model_resnet.pth
```
下次启动时会自动加载该模型。
**4. 查看预测效果**
点击 "显示红点追踪" 会打开一个覆盖屏幕的红点窗口，红点位置由当前模型预测的视线坐标驱动。
关闭方式：`Esc` / 点击红点窗口 / 关闭窗口。
**建议的第一次使用节奏**：
1. 采集 200~500 个样本，覆盖屏幕四角和中央
2. 用 "用全部数据训练" 手动训一次
3. 打开红点追踪窗口感受模型效果
4. 效果不理想就继续采样 + 训练；效果 OK 就切换交互模式使用
---
## 交互模式
模式切换由 `gaze_cursor_modes.py::CursorModeManager` 统一管理。所有模式都能与红点追踪并存显示。
### 1. 后台训练
GUI 默认模式。只采样和预测，不主动移动鼠标。适合：
- 采集训练数据
- 观察红点预测效果
- 安全调试
### 2. 光标随动(Alt)
- 模型持续预测视线坐标
- 按住 `Alt` 键时，鼠标会循环移动到预测视线位置
- 松开 `Alt` 后停止移动
适合低频触发式控制，不会一直抢鼠标。
### 3. 视线跳转（`gaze_jump`）
**触发逻辑**：用户手动移动鼠标时，如果当前鼠标位置与视线预测点距离超过阈值，鼠标会**一次性跳转**到视线位置附近。跳转后进入冷却期，避免连续触发。
GUI 参数（"视线跳转" 面板）：`触发距离(px)` 为文本输入框，输入合法正整数后立即生效；`冷却时间(ms)` 为滑块实时调节。
相关配置：
```python
gaze_jump_jump_threshold = 300
gaze_jump_cooldown_ms = 2000
gaze_jump_min_user_move = 5
```
### 4. 视线跟随（`gaze_follow`）
**触发逻辑**：当用户一段时间没有主动操作鼠标时，光标自动缓动跟随视线位置；检测到用户主动移动鼠标后暂停自动跟随，用户停止操作超过 `idle_seconds` 后恢复跟随。
相关配置：
```python
gaze_follow_idle_seconds = 3.0
gaze_follow_user_move_pixel = 3
gaze_follow_step_interval_ms = 30
gaze_follow_ease = 0.35
```
GUI 参数（"视线跟随" 面板）：`操作后暂停(秒)`、`跟随顺滑度`，均为滑块实时生效。
### 5. 视线滑翔（`gaze_glide`）
**触发逻辑**：当用户移动鼠标时，系统会判断：
1. 当前鼠标位置与视线位置是否足够远
2. 鼠标移动方向是否与 "当前鼠标位置 → 视线位置" 的方向一致
3. 两个方向的夹角余弦值是否超过阈值
如果满足条件，则对用户这一次鼠标移动附加额外相对位移，实现 "朝视线方向滑翔加速"。
速度系数近似为：
```text
factor = 1 + (max_multiplier - 1) × dist_factor × dir_factor
```
其中：
- `dist_factor`：距离越远越接近 1，越接近视线越接近 0
- `dir_factor`：方向越一致越接近 1，不一致时为 0
- `max_multiplier`：最大加速倍数
GUI 参数：`加速倍数`，滑块实时调节。除了加速倍数，同一个"视线滑翔"面板里还可调节：减速起始距离、全速加速距离、减速陡峭度。
相关配置：
```python
gaze_glide_max_multiplier = 5.0       # 最大加速倍数（滑块可调）
gaze_glide_near_threshold = 300       # 距视线 <= 该值 → 不加速
gaze_glide_far_threshold = 500        # 距视线 >= 该值 → 距离因子取最大
gaze_glide_cos_threshold = 0.6        # cos <= 该值视为方向不一致
gaze_glide_stroke_reset_ms = 200      # 鼠标停顿超过此时长 → 重置 stroke 起点
gaze_glide_min_stroke_len = 20        # stroke 过短时暂不启用方向判定
gaze_glide_dist_exponent = 2.0        # 距离因子曲线指数，>1 时近处衰减更快（急刹车）
gaze_glide_overshoot_anchor_ratio = 0.5  # 越界锚：extra 位移最多落到"距目标 near*ratio"处
```
**实现注意点**（也是踩坑经验）：
- 使用 `pyautogui.move(dx, dy)` 做**相对位移叠加**，而非 `moveTo(x, y)` 绝对坐标跳转，避免事件队列滞后导致鼠标被拉回。
- 使用**实时光标位置** `pyautogui.position()` 计算距离和视线方向，减少快速手动移动时的竞态问题。
- 近距离不加速，方便在目标附近精细定位。
---
## 模型设计思路
视线预测本质是一个**多因素回归**问题：屏幕上的注视点，既取决于**头部相对屏幕的姿态**（头怎么摆的），又取决于**眼球相对头部的偏转**（眼珠往哪儿看）。这两个因素在物理上是**加性叠加**的：
```text
gaze_on_screen ≈ f_head(头部姿态) + f_iris(虹膜偏转)
```
FollowMyGaze 的模型（`gaze_feature_based_model.py::GazeMoE`）不把这两个因素揉在一起端到端硬回归，而是显式建模为两个专家网络：
### 模型架构
```
输入 x (124 维几何特征)
       │
       ├──→ Head 专家 ──→ y_head (头部姿态贡献的注视点位置)
       │
       ├──→ Iris 专家 ──→ y_iris (虹膜偏转贡献的位移量)
       │
       └──→ Gate 路由 ──→ gate ∈ [0, 1] (基于 ||瞳孔偏移|| 的物理先验门控)
                              │
                    y_final = y_head + gate × y_iris
```
### 两个专家
- **Head 专家**：输入 124 维 → CrossNet 特征交叉 → 4 个 ResidualBlock → 输出 `y_head`（2 维）。负责从头部姿态、人脸位置等全局特征中预测注视点的"基准位置"。
- **Iris 专家**：与 Head 专家完全同构。负责从虹膜/瞳孔偏移等局部特征中预测"虹膜偏转带来的额外位移"`y_iris`。Iris 专家的最后一层会拼接 `y_head.detach()` 作为 anchor，避免梯度污染。
### 门控路由（Gate）
门控不是可学习的自由参数，而是基于**物理先验**：
- 从 124 维特征中取出瞳孔相对眼眶中心的偏移量（索引 8, 9, 16, 17：左右眼 eye_rel_x/y）
- 计算 `||rel||`（偏移量模长），用 Otsu 自适应阈值二值化
- 训练时通过 STE（Straight-Through Estimator）做 hard 二值化，推理时用 soft gate
- 门控还包含一个可学习的微调残差（默认 scale=0，即关闭）
**物理直觉**：当虹膜居中时（gate≈0），注视点完全由头部姿态决定；当虹膜偏转时（gate≈1），注视点 = 头部姿态 + 虹膜偏转。
### 多任务损失
训练时不仅监督最终输出 `y_final`，还通过辅助损失引导两个专家各司其职：
- `loss_final`：MSE(y_final, label) — 主目标
- `loss_head_aux`：虹膜居中时强监督 y_head ≈ label，偏转时弱监督
- `loss_iris_aux`：虹膜偏转时让 y_iris 学到完整贡献量，居中时强制归零
### 特征标准化
输入特征通过 EMA 在线更新的 Z-score 标准化（均值和标准差作为模型 buffer 随 state_dict 存取），确保训练和推理时特征分布一致。
---
## 训练与推理机制
### 影子模型（Shadow Model）
训练和推理共享同一个模型实例会导致线程冲突。解决方案：
1. 训练时克隆一份 shadow model，在 shadow 上训练
2. 推理线程继续使用原 `self.model`，不受训练影响
3. 训练完成后 shadow.eval()，原子替换 `self.model = shadow`
4. 下一帧推理自动使用新模型
### 在线学习
- 用户每点一次左键，特征 + 坐标成为一条训练样本
- 每累计 `auto_train_threshold`（默认 1024）个样本，自动触发后台全量训练
- 训练使用 `online_training_epoch`（默认 40）轮，训练完成后模型自动生效
- 可通过 GUI 开关控制是否启用自动训练
### 推理机制
- 推理在主线程通过 `root.after` 驱动的帧循环中触发（Windows Tkinter 非线程安全）
- 实际推理计算提交到 `ThreadPoolExecutor` 后台线程池，避免阻塞 UI
- 预测坐标经过 EMA（指数移动平均，α=0.9）平滑，减少帧间抖动
- 摄像头掉帧时自动重连，帧任务不堆积
---
## 配置说明
所有可调参数集中在 `global_info.py` 的 `GlobalInfo` 类中：

### 训练参数
```python
online_training_batchSize = 2048    # 在线训练 batch size
online_training_epoch = 40          # 在线训练轮数
offline_training_batchSize = 2048   # 全量训练 batch size
offline_training_epoch = 400        # 全量训练轮数
auto_train_threshold = 1024         # 自动训练触发阈值（样本数）
enable_auto_train = True            # 是否启用自动训练
sample_upper_limit = 100000         # 样本上限
```

### 交互模式参数
参见上文各交互模式的相关配置代码块。GUI 参数面板可实时调节，退出时自动持久化到 `user_config.json`。

### 多任务损失权重（`train_and_predict_dnn.py`）
```python
LOSS_W_FINAL = 1.0        # 最终预测主 loss
LOSS_W_HEAD_AUX = 0.8     # y_head 辅助 loss
LOSS_W_IRIS_AUX = 0.5     # y_iris 辅助 loss
```
---
## 项目结构
```
FollowMyGaze/
├── main.py                      # 入口：初始化数据目录、摄像头、启动 GUI
├── gui.py                       # Tkinter GUI：主窗口、参数面板、帧循环、模式切换
├── global_info.py               # 全局配置与状态变量（GlobalInfo 类）
├── user_config.py               # 用户配置持久化（JSON 读写）
├── gaze_feature_extractor.py    # MediaPipe 特征提取：124 维几何特征
├── gaze_feature_based_model.py  # 模型定义：SimpleDNN / ResNet / GazeMoE
├── train_and_predict_dnn.py     # 训练/推理控制器：影子模型、多任务损失、EMA 平滑
├── data_process_dnn.py          # 数据集管理：采样、持久化、自动训练触发
├── gaze_cursor_modes.py         # 鼠标交互模式：随动、跳转、跟随、滑翔
├── utils.py                     # 工具函数：红点窗口、点击监听、预测调度
├── data_process.py              # [旧] 图像级数据集处理
├── train_and_predict.py         # [旧] 图像级训练控制器
├── cnn_model.py                 # [旧] 图像级 CNN 模型
├── test.py                      # 测试脚本
├── asset/
│   └── app_gui.png              # GUI 截图
└── README.md
```
核心文件依赖关系：
```
main.py
  ├── gui.py
  │     ├── gaze_cursor_modes.py  ← 交互模式
  │     ├── utils.py              ← 红点 / 点击监听 / 预测调度
  │     └── user_config.py        ← 配置持久化
  ├── gaze_feature_extractor.py   ← 特征提取
  └── train_and_predict_dnn.py    ← 训练/推理
        ├── gaze_feature_based_model.py  ← GazeMoE 模型
        └── data_process_dnn.py          ← 数据管理
```
---
## 常见问题
### Q: 摄像头打不开 / 黑屏？
A: 检查是否有其他程序占用了摄像头。Windows 的"相机"应用或浏览器视频通话可能独占摄像头。关闭其他程序后重启 `python main.py`。程序内置了摄像头自动重连机制。
### Q: 红点不跟视线走？
A: 可能原因：1) 样本太少（建议 200+）；2) 没训练过模型（点"用全部数据训练"）；3) 面部光照不足导致 MediaPipe 特征提取不稳定。先确保采样质量，再训练。
### Q: 训练时 GUI 卡顿？
A: 训练在后台线程执行，UI 不应卡死。如果出现卡顿，通常是因为 CPU 资源不足。可以降低 `online_training_batchSize` 或 `online_training_epoch`。
### Q: 模型文件在哪里？
A: 模型保存在 `~/FollowMyGaze/gaze_model_resnet.pth`。数据样本在 `~/FollowMyGaze/samples/samples.pkl`。
### Q: 如何重置所有数据？
A: 删除 `~/FollowMyGaze/` 目录即可清空所有样本和模型。
### Q: 鼠标交互模式不生效？
A: 检查是否从"后台训练"切换到了其他模式。`视线跳转` 需要手动移动鼠标才能触发；`视线跟随` 需要停止操作鼠标几秒后才开始跟随；`视线滑翔` 需要朝视线方向移动鼠标才加速。
### Q: 与 Mac 版有什么区别？
A: 见[与 Mac 版的差异](#与-mac-版的差异)。核心差异：无需系统权限配置、Tkinter 主线程驱动 UI。GazeMoE 模型架构、特征维度和交互模式完全相同。
---
## 开发建议
### 快速实验流程
1. 改模型（`gaze_feature_based_model.py`）或改损失（`train_and_predict_dnn.py`）
2. 删除旧模型文件 `~/FollowMyGaze/gaze_model_resnet.pth`
3. 运行 `python main.py`，采集样本 → 训练 → 红点验证
4. 整个循环控制在 5 分钟内
### 调试技巧
- 设置 `GlobalInfo.enable_auto_train = False` 避免自动训练干扰调试
- 红点追踪窗口是最直观的模型效果反馈
- 训练日志中 `gate_mean`、`gate_min`、`gate_max` 可以观察门控是否正常工作（理想情况是两极分化：大部分≈0 或≈1）
### 添加新特征
在 `gaze_feature_extractor.py::extract_features_from_image` 中追加特征后，记得同步更新：
- `GazeMoE` 的 `input_size` 参数
- `GATE_FEATURE_INDICES` 如果新特征改变了瞳孔偏移的索引位置
---
## 未来展望
- [ ] 眨眼点击：用眨眼动作替代鼠标左键
- [ ] 更丰富的特征工程：探索端到端 CNN 特征 + 几何特征的混合方案
- [ ] 眼动类型识别：区分扫视、注视、平滑追踪
---
## License
MIT License - 详见 [LICENSE](LICENSE) 文件。