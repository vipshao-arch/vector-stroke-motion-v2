# 分层动态笔触重建 v2

这个独立工程从参考视频离线分析出固定画面与逐帧运动笔触，在浏览器里用 Three.js/WebGL 静态图元、一张固定低频基底和 Canvas 2D 前景重建动态画面。网页不播放视频，不加载逐帧原画面，也不把参考视频帧作为纹理或贴片发布；运行时只加载一张固定基底 `assets/base.jpg`。

参考视频为 1504×704、24 fps、124 帧，分析统一缩放到 752×352。播放器用 `Math.round(seconds * 24)` 选择 `VTSK0002` 指令帧组，因此 0、1、2.5、4、5 秒分别对应参考帧 0、24、60、96、120。每条轮廓笔触和每个残差笔刷都来自对应帧与固定基底的局部差异，使头发、胸部、左右翅膀和花瓣的可见状态与该时刻的参考帧同步。

## 分层与运动

- 背景、拱门、左右雕像、座椅和座台固定在统一坐标里；运行时位移、缩放、旋转均为零，31 个相机矩阵全是单位矩阵。
- `assets/strokes.bin`（VSTK0004）保存 36,300 条静态 WebGL 绘图指令、区域分类、运动模型及单位相机。旧的动态 WebGL 实例被停用，避免冻结的中值轮廓叠在运动画面上。
- `assets/temporal_runtime.bin`（VTSK0002）保存 124 帧 × 5 区域的短切线笔触和 4 px 残差笔刷。它只有坐标、角度、长度、颜色和不透明度，没有图像块或视频帧。
- `DynamicOverlay` 按当前秒数读取对应帧组，在 Canvas 2D 中绘制发丝曲线变化、胸部呼吸高光、左右翅膀各自的浮动轮廓，以及花瓣的下落、横摆与自旋。
- 固定基底只负责低频背景和静态构图。动态区域按当前帧的局部色差、边缘、时域活动度与软化的区域内部权重绘制，区域边界经过距离变换腐蚀，避免出现多边形硬边。

## 文件

- `index.html` / `styles.css`：播放器页面
- `src/player.js`：静态 WebGL 图元、固定基底、时间同步与 Canvas 前景挂载
- `src/instructions.js`：VSTK0004 解析器、单位相机和静态锁定校验数据
- `src/dynamicOverlay.js`：VTSK0002 帧索引、短笔触、残差笔刷和花瓣图元绘制
- `src/main.js`：播放控制、诊断信息与运行时统计
- `assets/base.jpg`：唯一的固定低频基底
- `assets/strokes.bin` / `assets/strokes.manifest.json`：静态 WebGL 绘图指令及格式说明
- `assets/temporal_runtime.bin` / `assets/temporal_runtime.manifest.json`：逐帧动态绘图指令及格式说明
- `tools/extract_strokes.py`：固定基底、区域分类、静态笔触和 VSTK0004 打包
- `tools/build_temporal_runtime.py`：逐帧局部残差、轮廓采样和 VTSK0002 打包
- `tools/check-instructions.mjs`：VSTK0004 格式、静态锁定、单位相机和数值范围检查
- `tools/check-temporal-runtime.mjs`：VTSK0002 帧表、记录边界和五时刻帧组检查
- `tools/verify-browser.mjs` / `tools/verify-all.mjs`：本地 HTTP 加载、五时刻合成截图、控制台/WebGL/shader 检查、单一基底与帧序列检查
- `tools/verify_roi.py`：静态漂移、动态变化、同帧误差下降与最佳匹配参考帧检查
- `tools/make_comparison.py`：参考帧与浏览器五时刻合成截图对照
- `verification/roi_report_browser.json`、`verification/comparison.png`：浏览器合成图的最终验收数据与参考帧对照

## 本地运行

```powershell
npm install
npm run serve
```

默认访问 `http://127.0.0.1:8765/`。页面支持暂停、拖动时间轴和回到起点；拖动时间轴会立即选择对应的 24 fps 帧组。

## 指令格式

`assets/strokes.bin` 以 `VSTK0004` 开头。头部为 `version u16, keyframes u16, json_length u32, stroke_count u32`，随后是 JSON 元数据、31×6 个相机矩阵浮点数，以及每条笔触：

```text
kind u8, region u8, motion_model u8, flags u8, alpha f32,
params 12 f32, color 3 f32, motion 8 f32, keyframes 31*8 f32
```

静态笔触的关键帧恒为 `[0, 0, 1, 0, 0, 1, time, 0]`；31 个相机矩阵都是单位矩阵。完整字段见 [strokes.manifest.json](assets/strokes.manifest.json)。

`assets/temporal_runtime.bin` 以 `VTSK0002` 开头。42 字节头部保存尺寸、124 帧、24 fps、188×88 的 4 px 网格、记录长度和三张表的偏移。随后是每帧 × 每区域的笔触表及笔刷表；表项为 `count u16, offset u64`。短笔触记录为 14 字节：

```text
x u16 / 16, y u16 / 16, angle u16 / 65535, half_length u8 / 4,
width u8 / 4, rgb u8[3], alpha u8 / 255, is_petal u8, reserved u8
```

残差笔刷记录为 8 字节：`grid_index u32, rgb u8[3], alpha u8 / 255`。这些记录只包含绘图参数，清单见 [temporal_runtime.manifest.json](assets/temporal_runtime.manifest.json)。

## 重新生成

准备 Python 环境并安装 `numpy opencv-python-headless pillow`。系统 PATH 中需有 ffmpeg。原始 BGR24 帧缓存可从只读基线目录复用，本工程不修改该缓存或原有 GitHub 仓库/Pages。

```powershell
powershell -File tools\prepare_frames.ps1 -Video "C:\path\to\reference.mp4" -Work analysis
python tools\analyze_motion.py --cache analysis\frames_bgr24.raw --output analysis\motion_report.json --region-preview analysis\regions.png
python tools\extract_strokes.py --video "C:\path\to\reference.mp4" --cache analysis\frames_bgr24.raw --analysis analysis\motion_report.json --analysis-script tools\analyze_motion.py --output assets\strokes.bin
python tools\build_temporal_runtime.py --frames "C:\path\to\frames_bgr24.raw" --base assets\base.jpg
```

## 验证

```powershell
npm test
python tools\make_comparison.py
python tools\verify_roi.py --screenshots verification\screenshots --reference-frames "C:\path\to\frames_bgr24.raw" --output verification\roi_report_browser.json
```

`npm test` 覆盖 JS 语法、VSTK0004/VTSK0002 格式、五个时间点的本地浏览器加载、精确帧序列、单一固定基底、无逐帧图片请求、无 `<video>`、控制台/WebGL/shader 错误与静态相机。`verify_roi.py` 进一步衡量静态 ROI 漂移、五个动态区域变化、逐帧 MAE 相对固定基底的下降、同帧误差是否低于错帧误差，以及运动残差与参考变化的相关性，避免「只在动但没有对齐」。

验收时使用 0、1、2.5、4、5 秒。浏览器合成截图、WebGL 基底截图、前景层截图和报告输出到 `verification/screenshots/`；[comparison.png](verification/comparison.png) 把五个时刻的参考帧与浏览器合成并排展示。

## 发布

本工程已授权发布到 `vipshao-arch/vector-stroke-motion-v2`，Pages 地址为 `https://vipshao-arch.github.io/vector-stroke-motion-v2/`。`.github/workflows/deploy-pages.yml` 在 `main` 分支推送后上传静态目录并部署 Pages。原有 `vector-stroke-reconstruction` 仓库和 Pages 不属于本工程。

## 许可

Three.js 使用 MIT 许可，文本见 `assets/THREE-LICENSE.txt`。参考视频及由其生成的绘图数据仅作为本项目的输入和演示资产。
