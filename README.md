# FollowMyGaze

**语言 / Language：** 中文 | [English](#english-version)

> 用眼睛替手腕省下每天上千次的鼠标位移，降低腱鞘炎风险、提升工作效率。
> 最终目标：让视线成为你操作电脑的主力，逐步替代鼠标。视频介绍：https://www.bilibili.com/video/BV11ng969Ete/
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

---
# English Version

> Save your wrist thousands of mouse displacements every day — lower the risk of tenosynovitis and get more done.
> Ultimate goal: make gaze the primary way you operate your computer, gradually replacing the mouse. Video intro: https://www.bilibili.com/video/BV11ng969Ete/

**Language / 语言：** [中文](#followmygaze) | English

FollowMyGaze is a **personalized gaze tracking + gaze–mouse hybrid interaction** tool that runs on your own computer.
It takes an ordinary webcam as input, uses [MediaPipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) to extract face/eye geometric features, trains a small PyTorch model built on **Mixture-of-Experts with physics-prior gating**, and feeds the predicted gaze coordinates into several interaction modes: red-dot display, cursor follow, gaze jump, gaze follow and gaze glide.

> This repository is the **Windows version**, ported from [FollowMyGaze-Mac](https://github.com/Teacher-Cai/FollowMyGaze-Mac). It keeps the Mac version's GazeMoE architecture and interaction modes, and adapts them to Windows (Tkinter threading model, camera API, etc.).

## Project Positioning
**The core goal of this project is to reduce mouse usage.** In modern office work, clicking and dragging are extremely frequent, extremely repetitive actions; over time they lead to carpal tunnel syndrome, tenosynovitis and other repetitive strain injuries, and they slow you down. FollowMyGaze aims to:
- **Reduce mouse operations** — move the most tiring part (long-range cursor travel) from your wrist to your eyes.
- **Lower the risk of tenosynovitis / mouse hand** — cut down repeated micro-movements and static load on the wrist and fingers.
- **Improve efficiency** — gaze is great at fast long-range transfers; the hand only handles the last centimeter of fine positioning.
- **Eventually replace the mouse** — the long-term goal is for gaze to become the primary input method.
**A side benefit: it is also a lightweight deep learning playground.** Because "you can tell whether a model is good with your own eyes", the project covers data collection, feature engineering, model training, shadow-model deployment, real-time inference and the interaction layer end to end — ideal for experiments where you "change one line, then verify it instantly with your eyes".
Key characteristics of the whole system:
- **Highly personalized**: every click you make becomes a training sample; the model keeps learning your head-pose + eye-movement habits online.
- **Highly interpretable**: 124 hand-crafted geometric features plus two explicit experts (head pose / iris) — you can see at a glance what changed when you changed something.
- **End-to-end engineering**: from data collection, feature engineering, model training, shadow-model deployment and real-time inference to the mouse interaction layer — small but complete.
> ⚠️ This project is currently a personal experiment / prototype tool, not a general-purpose SDK. Model quality depends heavily on your own sampling quality.
---
## Contents
- [Project Positioning](#project-positioning)
- [Background](#background)
  - [Perspective 1: Lowering mouse-hand / tenosynovitis risk](#perspective-1-lowering-mouse-hand--tenosynovitis-risk)
  - [Perspective 2: A lightweight deep learning experimentation platform](#perspective-2-a-lightweight-deep-learning-experimentation-platform)
- [Features](#features)
- [Differences from the Mac Version](#differences-from-the-mac-version)
- [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Launch](#launch)
  - [Basic Workflow](#basic-workflow)
- [Interaction Modes](#interaction-modes)
- [Model Design](#model-design)
- [Training and Inference](#training-and-inference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [Development Tips](#development-tips)
- [Roadmap](#roadmap)
- [License](#license-1)
---
## Background
FollowMyGaze is not meant to be "yet another general-purpose gaze tracking SDK". It serves two audiences that look unrelated but both benefit from **"gaze-driven + personalized learning"**. Understanding both perspectives helps you decide whether to run it, and how.
### Perspective 1: Lowering mouse-hand / tenosynovitis risk
For people who use a computer for long hours, mouse operations are **extremely frequent and extremely repetitive**:
- thousands of clicks every day
- lots of small wrist adjustments (dragging the cursor from A to B)
- holding a static posture of "forearm extended + wrist floating + thumb abducted" for long periods
Over time these accumulate into common problems:
- **Carpal tunnel syndrome**: compression of the median nerve, causing finger numbness and reduced grip strength
- **Tenosynovitis** (De Quervain / trigger finger): repeated friction inflames the tendon sheath, causing soreness in the thumb, index finger and wrist
- **Mouse elbow / tennis elbow**: repeated loading and inflammation of the lateral epicondyle
- **Neck and shoulder tension**: holding the elbow up keeps the trapezius and levator scapulae tight
**The value of FollowMyGaze is not "replacing the mouse", but moving the most tiring part of every cursor movement — long-range travel — from the wrist to the eyes.** Concretely:
1. **Eliminate most long drags**
   Look somewhere and the cursor jumps/glides there. A large movement from the top-left to the bottom-right of the screen used to need a lifted wrist and forearm coordination; now 90%+ of the distance needs almost no hand movement.
2. **The hand only does the "last centimeter"**
   `gaze_glide` / `gaze_jump` both mean **long distance handled by gaze, short distance left to the hand**. This plays to both strengths: gaze is fast over long ranges, the hand is precise over short ranges.
3. **Fewer finger/wrist micro-movements**
   Switching from your editor to a browser and clicking a button traditionally means "move mouse → click", possibly with 10+ small adjustments. With gaze glide, one quick push plus gaze focus lands you near the target.
4. **Less static load time**
   Traditional mouse use requires your hand to stand by on the mouse at all times; the modes here let you **take your hand off the mouse completely** to do other things (read, think), touching it only when you actually need to click.
5. **Keep the controllability of a normal mouse**
   The project deliberately avoids "fully hands-free gaze control" and instead offers **several hybrid modes** (Alt trigger, gaze jump trigger, gaze glide acceleration) — your hand stays in charge and gaze only acts as an amplifier. That reduces strain without sacrificing precision or predictability.
> This is not a medical device claim; it is a reasonable inference from an ergonomics standpoint. If you have real RSI (repetitive strain injury) symptoms, see a doctor.
**Who might benefit**:
- programmers, designers and operations staff who spend more than 6 hours a day on keyboard and mouse
- people with mild wrist discomfort who want to reduce their usage intensity
- users with large monitors / dual displays where the cursor has to travel far
- users who want to free their hands while reading or watching videos
### Perspective 2: A lightweight deep learning experimentation platform
If the health angle is for **every heavy user**, the engineering angle is for **every developer interested in deep learning**.
Most introductory deep learning projects stop at "train an MNIST / CIFAR classifier", where all you can see is a changing accuracy number. FollowMyGaze offers a completely different learning loop: **you can tell whether a model is good with your own eyes.**
This brings several unique teaching / experimentation advantages:
1. **Immediate, visual feedback**
   Change the model, train once, open the red-dot tracking window and judge with your eyes: how far off is the red dot from your gaze? Does it fail at the screen corners? Is there obvious jitter or drift? This is far more intuitive than watching a loss curve, and much more likely to trigger real intuition.
2. **Full coverage of the deep learning engineering pipeline**
   The project contains every key step from data to deployment, and each step is small enough to read and modify:
   - **Data collection**: the mouse click listener in `utils.py`
   - **Feature engineering**: the 124 hand-crafted geometric features in `gaze_feature_extractor.py`
   - **Model architecture**: MoE + CrossNet + ResBlock in `gaze_feature_based_model.py`
   - **Loss design**: the multi-task weighted loss in `train_and_predict_dnn.py`
   - **Training loop**: Adam + BN + Dropout, with full and online configurations
   - **Online learning**: automatic training triggers in `data_process_dnn.py`
   - **Model deployment**: shadow model + atomic swap + EMA smoothing
   - **Real-time inference**: Tk main thread + background thread pool + 30 fps camera
   - **Interaction layer**: several mouse control modes to validate model quality
3. **Every change can be verified immediately**
   - add a new feature → train → see whether the red dot is steadier
   - change the gating threshold strategy → train → watch drift when head pose changes
   - change a loss weight → train → look at the error distribution for different eye states
   - tune the EMA alpha → feel the trade-off between smoothness and latency
4. **Moderate data size, low experiment cost**
   A few thousand local samples, and one training run takes tens of seconds to a few minutes. Unlike toy datasets it stays close to real engineering; unlike large projects you don't wait a day per experiment. **The whole "edit code → train → verify with your eyes" loop stays under 5 minutes**, which is ideal for fast iteration.
5. **Exposes real engineering problems**
   This is not a clean algorithm exercise but a complete system where **engineering details must be solved**:
   - How do the training and inference threads share a model? (→ shadow model)
   - Does BatchNorm pollute running stats at inference time? (→ eval/train isolation)
   - What if the camera drops frames? (→ auto reconnect + frame-task backpressure)
   - What if event-stream latency drags the mouse back? (→ relative displacement instead of absolute coordinates)
   - What if the UI freezes during training? (→ everything on background threads + `root.after`)
   These pitfalls show up in almost every real-time AI system, and they are worth hitting yourself.
6. **Encourages physics-intuition-driven model design**
   Point of regard = head pose contribution + iris deflection contribution. That physical intuition directly determines the MoE structure, with gating routed by the `||rel||` physics prior. This "derive the network from domain priors" approach cannot be learned from toy examples, yet it is exactly how models are designed in industry.
**How to use it**:
- Deep learning beginners: compare `SimpleDNN` and `GazeMoE` to understand why residual connections, BN and gating are added
- Intermediate developers: try changing features, architecture or loss, then train → verify with your eyes
- People learning "real-time AI systems engineering": `gui.py` + `train_and_predict_dnn.py` together are a complete small online-learning platform
---
## Features
Built around the two perspectives above, the project offers:
- 🎥 **Real-time camera preview**: a Tkinter GUI showing the camera feed.
- 🧠 **MediaPipe face feature extraction**: face, iris and head-pose related features via MediaPipe FaceMesh.
- 🖱️ **Click sampling**: a left click saves the current gaze features plus the click coordinates as a training sample.
- 💾 **Sample persistence**: samples are stored in `~/FollowMyGaze/samples/samples.pkl`.
- 🔢 **Sample counter**: shows `local samples + session samples = total samples`.
- 🏋️ **Manual training**: one-click "train on all data", running on all samples in the background.
- ⚙️ **Automatic training**: every `auto_train_threshold` new samples are persisted and trained in the background.
- 🌗 **Shadow model training**: training uses a shadow model and atomically replaces the inference model when done, so training and inference never interfere.
- 🔴 **Red-dot tracking window**: shows the current predicted gaze position for easy visual verification.
- ✋ **Multiple mouse interaction modes**: background training / Alt cursor follow / gaze jump / gaze follow / gaze glide.
- 🎛️ **Parameter persistence**: GUI parameters are saved on exit to the config file specified by `user_config.py` and restored on the next launch.
---
## Differences from the Mac Version
| Aspect | Mac version | Windows version |
|------|--------|------------|
| Feature dimension | 124 | 124 (aligned with Mac) |
| Feature extraction file | `gaze_feature_extractor_mac.py` | `gaze_feature_extractor.py` |
| System permissions | Camera + Accessibility + Input Monitoring | Camera only (`pynput` and `pyautogui` need no extra permissions on Windows) |
| UI threading model | background threads may update the UI | the main thread must drive the UI via `root.after` |
| Packaging | `.app` (ad-hoc signed) | `.exe` (PyInstaller) |
| GazeMoE model | same architecture | same architecture (identical) |
| Interaction modes | all 5 | all 5 (same implementation) |
---
## Quick Start
> **TL;DR (3 steps)**
> ```bash
> pip install opencv-python pyautogui mediapipe numpy pandas torch pillow pynput
> python main.py
> ```
> Then look at the screen and left-click a few times to sample → click "train on all data" → open red-dot tracking to see the result.
### Requirements
Recommended environment:
- Windows 10/11
- Python 3.9+
- A webcam
Main Python dependencies:
```bash
pip install opencv-python pyautogui mediapipe numpy pandas torch pillow pynput
```
CUDA users should install the PyTorch build matching their GPU, following the [official PyTorch guide](https://pytorch.org/get-started/locally/).
### Launch
From the project root:
```bash
python main.py
```
The main window appears:
- **Top**: camera preview
- **Middle**: hints, sample counter, current mode
- **Mode selection**: background training / cursor follow (Alt) / gaze jump / gaze follow / gaze glide
- **Controls**: red-dot tracking button, auto-train toggle, and parameter panels grouped by mode:
  - **Gaze jump**: trigger distance, cooldown
  - **Gaze follow**: pause after user action, follow smoothness
  - **Gaze glide**: acceleration multiplier, deceleration start distance, full-speed distance, deceleration steepness
- **Bottom**: manual training button and training progress bar
> Parameter values are persisted on exit (`user_config.py`) and restored on the next launch.
### Basic Workflow
**1. Start the program**
```bash
python main.py
```
Make sure the camera preview appears.
**2. Collect samples**
Keep your head and eyes in a natural state, look at a point on the screen, then left-click that point with the mouse.
Each left click saves:
- the gaze features extracted from the current frame
- the mouse click coordinates `(x, y)`
Samples first go into the session buffer and are persisted:
- automatically every time `GlobalInfo.auto_train_threshold` new samples accumulate
- on exit, for any samples not yet persisted
Sample file path:
```text
~/FollowMyGaze/samples/samples.pkl
```
**3. Train the model**
Option 1: manual training. Click "train on all data" in the GUI; the program loads all local samples, trains the model and shows progress.
Option 2: automatic training, enabled by default:
```python
GlobalInfo.enable_auto_train = True
GlobalInfo.auto_train_threshold = 1024
```
Every 1024 new samples trigger a background full training run, using `online_training_epoch` epochs. The trained model is saved to:
```text
~/FollowMyGaze/gaze_model_resnet.pth
```
It is loaded automatically on the next launch.
**4. Check the predictions**
Click "show red-dot tracking" to open a red-dot overlay across the screen, driven by the gaze coordinates predicted by the current model.
Close it with `Esc` / by clicking the red-dot window / by closing the window.
**A suggested first session**:
1. Collect 200–500 samples covering the four corners and the center of the screen
2. Run "train on all data" once manually
3. Open the red-dot tracking window to feel how the model performs
4. If it is not good enough, keep sampling and training; if it is OK, switch to an interaction mode
---
## Interaction Modes
Mode switching is managed by `gaze_cursor_modes.py::CursorModeManager`. All modes can coexist with red-dot tracking.
### 1. Background training
The GUI default mode. It only samples and predicts, and never moves the mouse. Good for:
- collecting training data
- watching the red-dot predictions
- safe debugging
### 2. Cursor follow (Alt)
- the model keeps predicting gaze coordinates
- while `Alt` is held, the mouse repeatedly moves to the predicted gaze position
- releasing `Alt` stops the movement
Suited to low-frequency, triggered control — it never keeps grabbing the mouse.
### 3. Gaze jump (`gaze_jump`)
**Trigger logic**: when you move the mouse manually and the distance between the current mouse position and the predicted gaze point exceeds a threshold, the mouse **jumps once** to near the gaze position. After a jump it enters a cooldown period to avoid repeated triggering.
GUI parameters ("Gaze jump" panel): `trigger distance (px)` is a text field that takes effect once a valid positive integer is entered; `cooldown (ms)` is a slider adjusted live.
Related configuration:
```python
gaze_jump_jump_threshold = 300
gaze_jump_cooldown_ms = 2000
gaze_jump_min_user_move = 5
```
### 4. Gaze follow (`gaze_follow`)
**Trigger logic**: when the user has not touched the mouse for a while, the cursor eases toward the gaze position automatically; once user movement is detected the auto-follow pauses, and it resumes after the user has been idle for more than `idle_seconds`.
Related configuration:
```python
gaze_follow_idle_seconds = 3.0
gaze_follow_user_move_pixel = 3
gaze_follow_step_interval_ms = 30
gaze_follow_ease = 0.35
```
GUI parameters ("Gaze follow" panel): `pause after user action (s)` and `follow smoothness`, both sliders taking effect live.
### 5. Gaze glide (`gaze_glide`)
**Trigger logic**: when the user moves the mouse, the system checks:
1. whether the current mouse position is far enough from the gaze position
2. whether the mouse movement direction is consistent with the direction "current mouse position → gaze position"
3. whether the cosine of the angle between the two directions exceeds a threshold
If so, extra relative displacement is added to this mouse movement, producing "glide acceleration toward the gaze direction".
The speed factor is approximately:
```text
factor = 1 + (max_multiplier - 1) × dist_factor × dir_factor
```
where:
- `dist_factor`: closer to 1 the farther away, closer to 0 the closer to the gaze point
- `dir_factor`: closer to 1 the more aligned the direction, 0 when misaligned
- `max_multiplier`: the maximum acceleration multiplier
GUI parameters: `acceleration multiplier`, a live slider. Besides that, the same "Gaze glide" panel also exposes: deceleration start distance, full-speed distance, deceleration steepness.
Related configuration:
```python
gaze_glide_max_multiplier = 5.0       # max acceleration multiplier (slider-adjustable)
gaze_glide_near_threshold = 300       # distance to gaze <= this → no acceleration
gaze_glide_far_threshold = 500        # distance to gaze >= this → distance factor maxed out
gaze_glide_cos_threshold = 0.6        # cos <= this is treated as direction mismatch
gaze_glide_stroke_reset_ms = 200      # mouse pause longer than this → reset stroke origin
gaze_glide_min_stroke_len = 20        # strokes shorter than this skip direction checking
gaze_glide_dist_exponent = 2.0        # distance-factor curve exponent; >1 decays faster up close (hard braking)
gaze_glide_overshoot_anchor_ratio = 0.5  # overshoot anchor: extra displacement lands at most "near*ratio" from the target
```
**Implementation notes** (lessons learned):
- Use `pyautogui.move(dx, dy)` for **relative displacement accumulation** rather than `moveTo(x, y)` absolute jumps, so that event-queue lag does not drag the mouse back.
- Use the **live cursor position** `pyautogui.position()` to compute distance and gaze direction, reducing race conditions during fast manual movement.
- No acceleration up close, so you can position precisely near the target.
---
## Model Design
Gaze prediction is fundamentally a **multi-factor regression** problem: the point of regard on screen depends both on **the head's pose relative to the screen** (how the head is positioned) and on **the eyes' deflection relative to the head** (where the pupils are looking). Physically, the two are **additively** combined:
```text
gaze_on_screen ≈ f_head(head pose) + f_iris(iris deflection)
```
Instead of cramming both factors into a single end-to-end regression, FollowMyGaze's model (`gaze_feature_based_model.py::GazeMoE`) models them explicitly as two expert networks:
### Architecture
```
input x (124 geometric features)
       │
       ├──→ Head expert ──→ y_head (point of regard contributed by head pose)
       │
       ├──→ Iris expert ──→ y_iris (displacement contributed by iris deflection)
       │
       └──→ Gate routing ──→ gate ∈ [0, 1] (physics-prior gate based on ||pupil offset||)
                              │
                    y_final = y_head + gate × y_iris
```
### The two experts
- **Head expert**: 124-dim input → CrossNet feature crossing → 4 ResidualBlocks → outputs `y_head` (2-dim). Predicts the "base position" of the point of regard from global features such as head pose and face position.
- **Iris expert**: structurally identical to the Head expert. Predicts the "extra displacement caused by iris deflection" `y_iris` from local features such as iris/pupil offsets. Its last layer concatenates `y_head.detach()` as an anchor to avoid gradient contamination.
### Gate routing
The gate is not a freely learnable parameter but is based on a **physics prior**:
- extract the pupil offset relative to the eye socket center from the 124 features (indices 8, 9, 16, 17: left/right eye_rel_x/y)
- compute `||rel||` (offset magnitude) and binarize it with an Otsu adaptive threshold
- during training, hard binarization uses STE (Straight-Through Estimator); at inference a soft gate is used
- the gate also includes a learnable fine-tuning residual (default scale=0, i.e. disabled)
**Physical intuition**: when the iris is centered (gate≈0), the point of regard is determined entirely by head pose; when the iris deflects (gate≈1), the point of regard = head pose + iris deflection.
### Multi-task loss
Training does not only supervise the final output `y_final`; auxiliary losses guide the two experts to do their own jobs:
- `loss_final`: MSE(y_final, label) — the main objective
- `loss_head_aux`: strongly supervise y_head ≈ label when the iris is centered, weakly when deflected
- `loss_iris_aux`: teach y_iris the full contribution when the iris is deflected, force it to zero when centered
### Feature normalization
Input features are Z-score normalized with EMA-updated statistics (mean and std stored as model buffers in the state_dict), so that feature distributions match between training and inference.
---
## Training and Inference
### Shadow model
Having training and inference share one model instance causes thread conflicts. The solution:
1. clone a shadow model and train on the shadow
2. the inference thread keeps using the original `self.model`, unaffected by training
3. when training finishes, call shadow.eval() and atomically replace `self.model = shadow`
4. the next inference frame automatically uses the new model
### Online learning
- every left click becomes a training sample (features + coordinates)
- every `auto_train_threshold` (default 1024) samples automatically trigger a background full training run
- training uses `online_training_epoch` (default 40) epochs, and the model takes effect when done
- a GUI toggle controls whether automatic training is enabled
### Inference
- inference runs in the main thread inside the frame loop driven by `root.after` (Tkinter is not thread-safe on Windows)
- the actual computation is submitted to a `ThreadPoolExecutor` so the UI never blocks
- predicted coordinates are smoothed with an EMA (exponential moving average, α=0.9) to reduce frame-to-frame jitter
- the camera reconnects automatically after dropped frames, and frame tasks do not pile up
---
## Configuration
All tunable parameters live in the `GlobalInfo` class in `global_info.py`:

### Training parameters
```python
online_training_batchSize = 2048    # online training batch size
online_training_epoch = 40          # online training epochs
offline_training_batchSize = 2048   # full training batch size
offline_training_epoch = 400        # full training epochs
auto_train_threshold = 1024         # auto-training trigger threshold (sample count)
enable_auto_train = True            # whether automatic training is enabled
sample_upper_limit = 100000         # sample cap
```

### Interaction mode parameters
See the configuration blocks for each interaction mode above. The GUI parameter panels adjust them live and persist them to `user_config.json` on exit.

### Multi-task loss weights (`train_and_predict_dnn.py`)
```python
LOSS_W_FINAL = 1.0        # main loss on the final prediction
LOSS_W_HEAD_AUX = 0.8     # auxiliary loss on y_head
LOSS_W_IRIS_AUX = 0.5     # auxiliary loss on y_iris
```
---
## Project Structure
```
FollowMyGaze/
├── main.py                      # entry point: init data dir, camera, start GUI
├── gui.py                       # Tkinter GUI: main window, parameter panels, frame loop, mode switching
├── global_info.py               # global config and state (GlobalInfo class)
├── user_config.py               # user config persistence (JSON read/write)
├── gaze_feature_extractor.py    # MediaPipe feature extraction: 124 geometric features
├── gaze_feature_based_model.py  # model definitions: SimpleDNN / ResNet / GazeMoE
├── train_and_predict_dnn.py     # training/inference controller: shadow model, multi-task loss, EMA smoothing
├── data_process_dnn.py          # dataset management: sampling, persistence, auto-training triggers
├── gaze_cursor_modes.py         # mouse interaction modes: follow, jump, follow, glide
├── utils.py                     # utilities: red-dot window, click listener, prediction scheduling
├── data_process.py              # [legacy] image-level dataset processing
├── train_and_predict.py         # [legacy] image-level training controller
├── cnn_model.py                 # [legacy] image-level CNN model
├── test.py                      # test script
├── asset/
│   └── app_gui.png              # GUI screenshot
└── README.md
```
Core file dependencies:
```
main.py
  ├── gui.py
  │     ├── gaze_cursor_modes.py  ← interaction modes
  │     ├── utils.py              ← red dot / click listener / prediction scheduling
  │     └── user_config.py        ← config persistence
  ├── gaze_feature_extractor.py   ← feature extraction
  └── train_and_predict_dnn.py    ← training/inference
        ├── gaze_feature_based_model.py  ← GazeMoE model
        └── data_process_dnn.py          ← data management
```
---
## FAQ
### Q: The camera does not open / shows a black screen?
A: Check whether another program is using the camera. The Windows "Camera" app or a browser video call may hold it exclusively. Close the others and restart `python main.py`. The program has a built-in camera auto-reconnect mechanism.
### Q: The red dot does not follow my gaze?
A: Possible causes: 1) too few samples (200+ recommended); 2) the model has never been trained (click "train on all data"); 3) insufficient lighting on the face makes MediaPipe feature extraction unstable. Make sure sampling quality is good, then train.
### Q: The GUI stutters while training?
A: Training runs on a background thread, so the UI should not freeze. If it does, it is usually a lack of CPU resources. Try lowering `online_training_batchSize` or `online_training_epoch`.
### Q: Where are the model files?
A: The model is saved to `~/FollowMyGaze/gaze_model_resnet.pth`. Samples are in `~/FollowMyGaze/samples/samples.pkl`.
### Q: How do I reset all data?
A: Delete the `~/FollowMyGaze/` directory to clear all samples and models.
### Q: The mouse interaction modes do not work?
A: Check that you switched away from "background training". `Gaze jump` requires you to move the mouse manually to trigger; `gaze follow` only starts after you have been idle for a few seconds; `gaze glide` only accelerates when you move the mouse toward the gaze direction.
### Q: What is different from the Mac version?
A: See [Differences from the Mac Version](#differences-from-the-mac-version). The key differences: no system permission setup, and the UI is driven by the Tkinter main thread. The GazeMoE architecture, feature dimension and interaction modes are identical.
---
## Development Tips
### Fast experiment loop
1. change the model (`gaze_feature_based_model.py`) or the loss (`train_and_predict_dnn.py`)
2. delete the old model file `~/FollowMyGaze/gaze_model_resnet.pth`
3. run `python main.py`, collect samples → train → verify with the red dot
4. the whole loop stays under 5 minutes
### Debugging tips
- set `GlobalInfo.enable_auto_train = False` so automatic training does not interfere with debugging
- the red-dot tracking window is the most direct feedback on model quality
- `gate_mean`, `gate_min`, `gate_max` in the training log show whether gating works properly (ideally it is polarized: mostly ≈0 or ≈1)
### Adding new features
After appending features in `gaze_feature_extractor.py::extract_features_from_image`, remember to update:
- the `input_size` parameter of `GazeMoE`
- `GATE_FEATURE_INDICES` if the new features shift the pupil offset indices
---
## Roadmap
- [ ] Blink click: use a blink to replace the left mouse button
- [ ] Richer feature engineering: explore a hybrid of end-to-end CNN features + geometric features
- [ ] Eye movement type recognition: distinguish saccades, fixations and smooth pursuit
---
## License
MIT License - see the [LICENSE](LICENSE) file for details.
