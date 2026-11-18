# 分层动态笔触重建（VSTK0004）

这个独立工程从参考视频离线分析出一套区域感知的绘图指令，在浏览器运行时用 Three.js/WebGL 的椭圆、线段、二次贝塞尔、软笔刷和花瓣图元重新绘制。网页不播放视频，也不逐帧贴原视频画面；只加载一张固定的低频基底用于填补图元间隙。

参考视频为 1504×704、24 fps、124 帧；分析统一缩放到 752×352。与旧版不同，这里没有把单一全局仿射相机变换施加到所有笔触：

- 背景、拱门、左右雕像、座椅和座椅支撑结构锁在统一坐标中。运行时位移、缩放、旋转均为零，31 个相机矩阵全是单位矩阵。
- 胸部与躯干围绕胸部锚点做细小周期缩放及亮度呼吸。
- 头发按发丝曲线传播相位，并逐笔触弯曲控制点。
- 左右翅膀围绕各自锚点浮动，弯曲与透明度幅度随到锚点的距离增加。
- 花瓣有生成时间、下落速度、横向摆动、自旋、生命周期和淡入淡出；闪动粒子有独立相位。
- 每条笔触都有 `region`、`motion_model`、8 个运动参数，以及 31 组时间关键帧（从旧版 13 组提升），在着色器中用 smoothstep 插值。

## 文件

- `index.html` / `styles.css`：播放器页面
- `src/player.js`：Three.js 图元绘制、静态锁定和逐区域运动着色器
- `src/instructions.js`：浏览器与 Node 共用的 VSTK0004 解析器
- `src/main.js`：播放控制、错误诊断与统计信息
- `assets/strokes.bin`：36,300 条二进制绘图指令
- `assets/strokes.manifest.json`：指令格式、区域、运动模型及相机矩阵说明
- `assets/base.jpg`：固定坐标的时域中值低频基底
- `tools/analyze_motion.py`：时域方差、局部差分、Farneback 光流、静态锚点 Shi-Tomasi/ORB/RANSAC 与受限 ECC
- `tools/extract_strokes.py`：固定基底、区域分类、逐笔触关键帧和二进制打包
- `tools/verify-browser.mjs` / `tools/verify-all.mjs`：本地 HTTP 加载、五时刻截图、控制台/WebGL/shader 检查
- `tools/verify_projection.py`：从同一个 `strokes.bin` 做 CPU 投影，供无头环境的像素级 ROI 测量
- `tools/verify_roi.py`：静态漂移与动态区域时序变化统计
- `analysis/motion_report.json`、`analysis/regions.png`：离线运动分析与区域可视化
- `verification/roi_report.json`、`verification/comparison.png`：验收数据与参考帧对比

## 本地运行

```powershell
npm install
npm run serve
```

默认访问 `http://127.0.0.1:8765/`。页面支持暂停、拖动时间轴和回到起点。

## 指令格式

`assets/strokes.bin` 以 `VSTK0004` 开头。头部为 `version u16, keyframes u16, json_length u32, stroke_count u32`，随后是 JSON 元数据、31×6 个相机矩阵浮点数，以及每条笔触：

```text
kind u8, region u8, motion_model u8, flags u8, alpha f32,
params 12 f32, color 3 f32, motion 8 f32, keyframes 31*8 f32
```

关键帧字段是 `dx, dy, scale, rotate, delta_alpha, brightness, time, wobble`。静态笔触的关键帧恒为 `[0, 0, 1, 0, 0, 1, time, 0]`；动态笔触的关键帧由区域运动模型和逐笔触锚点、速度、相位生成。运动参数记录胸部锚点、发丝根部与相位、翅膀锚点、花瓣速度与自旋等。完整字段见 [manifest](assets/strokes.manifest.json)。

## 重新生成

准备 Python 环境并安装 `numpy opencv-python-headless pillow`。系统 PATH 中需有 ffmpeg。

```powershell
powershell -File tools\prepare_frames.ps1 -Video "C:\path\to\reference.mp4" -Work analysis
python tools\analyze_motion.py --cache analysis\frames_bgr24.raw --output analysis\motion_report.json --region-preview analysis\regions.png
python tools\extract_strokes.py --video "C:\path\to\reference.mp4" --cache analysis\frames_bgr24.raw --analysis analysis\motion_report.json --analysis-script tools\analyze_motion.py --output assets\strokes.bin
```

原始 BGR24 缓存也可从只读基线目录的 `analysis\frames_bgr24.raw` 复用；本工程不修改该文件或原有 GitHub 仓库/Pages。

## 验证

```powershell
npm test
python tools\verify_projection.py
python tools\verify_roi.py --screenshots verification\projection --reference-frames "C:\path\to\frames_bgr24.raw" --output verification\roi_report.json
```

`npm test` 覆盖 JS 语法、指令格式/范围/单位相机、五个时间点的本地浏览器加载、WebGL 错误和 shader 编译。浏览器截图在 `verification/screenshots/`。`verify_projection.py` 从相同指令做确定性 CPU 投影，用于静态锚点漂移和动态 ROI 的像素统计；它不是网页运行时资源。

验收时检查 0、1、2.5、4、5 秒。浏览器/Chromium 无头 SwiftShader 的保存绘图缓冲可能复用首帧，因此动态像素统计以 CPU 投影为准；浏览器检查负责确认页面确实由 VSTK0004 指令在 WebGL 中绘制、无 `video` 元素、无控制台/shader 错误。

## 许可

Three.js 使用 MIT 许可，文本见 `assets/THREE-LICENSE.txt`。参考视频及由其生成的绘图数据仅作为本项目的输入和演示资产。
