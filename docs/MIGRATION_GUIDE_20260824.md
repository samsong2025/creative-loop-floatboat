# Creative Loop Floatboat 迁移说明

文档版本：2026-08-24  
适用版本：当前工作区代码（API 0.19.0 / Agent API 0.20.0）

## 1. 系统概览

项目是一个由 Docker Compose 启动的本地视频素材处理服务，主要供 Floatboat 前端或其他 HTTP 客户端调用。

运行拓扑：

```text
Floatboat / 浏览器
        |
        | HTTP :8000
        v
creative-loop-api (FastAPI + Uvicorn + FFmpeg + Tesseract + Whisper)
        |
        +-- insight-browser (Selenium Chrome :4444 / noVNC :7900)
        +-- workspace 挂载：原始素材、配置、审核文件、最终输出
        +-- browser-state 挂载：Insight 登录会话
        +-- model-cache 挂载：Whisper 模型缓存
        +-- 可选 ProPainter：动态水印时序修复
```

默认 API 地址：`http://localhost:8000`  
接口文档：`http://localhost:8000/docs`  
健康检查：`GET /health`

## 2. 目录职责

| 路径 | 是否进源码包 | 用途 |
|---|---:|---|
| `api/app` | 是 | FastAPI、采集、语言识别、品牌处理、固定/动态水印、审核 API |
| `api/Dockerfile` | 是 | API 镜像构建，安装 FFmpeg、Tesseract、Python 依赖 |
| `api/requirements.txt` | 是 | Python 依赖锁定版本 |
| `compose.yaml` | 是 | `creative-loop-api` 和 `insight-browser` 服务编排 |
| `creative-loop-control.cmd` | 是 | Windows 安全启动器：`status`、`start`、`rebuild` |
| `workspace/config` | 是 | 产品映射、品牌注册表、品牌资产路径、水印参考图和布局规则 |
| `workspace/assets` | 是 | Floatboat 自有品牌 Logo、Icon、Mid Promo、End Card 素材 |
| `api/app/static` | 是 | 内置审核工作台页面 |
| `tools/*.py` | 是 | 诊断和回归测试工具 |
| `tools/propainter-gpu-env` | 否 | 本机 Python 虚拟环境，不能跨电脑直接复用；目标机需重新安装 |
| `workspace/raw` | 否 | 下载的原始视频，体积大，按需迁移 |
| `workspace/processed` | 否 | 中间视频产物，按需迁移 |
| `workspace/review` | 否 | 诊断报告、预览、审核产物，可按需迁移 |
| `workspace/output` | 否 | 最终视频和上传产物，可按需迁移 |
| `workspace/state` | 否 | 任务状态、上传 ledger、幂等状态；需要续接旧任务时才迁移 |
| `workspace/cache` | 否 | 渲染缓存，可重新生成 |
| `workspace/asset_index.db` | 否 | 本地素材索引，服务启动时会自动创建 |
| `browser-state` | 否 | Selenium/Insight 登录会话，含敏感 Cookie；建议在目标机重新登录 |
| `model-cache` | 否 | Whisper 模型缓存，目标机可重新下载 |
| `logs` | 否 | 运行日志 |

## 3. 当前核心代码链路

### 3.1 采集与语言识别

入口主要在 `api/app/main.py`：

1. 接收详情页链接或搜索条件。
2. 通过 `insight-browser` 执行页面访问、素材下载和详情信息采集。
3. 采集产品名、App 标题、包名、产品 Icon/Logo URL、国家/语言等身份信息。
4. 使用 `faster-whisper` 做源语言识别；识别不完整的素材进入缺失语言/审核路径，不直接进入正式 raw 目录。
5. 素材以 SHA256 和 SQLite 索引去重。

### 3.2 水印识别与路由

主要实现：`api/app/branding_v09.py`、`api/app/routers/watermark.py`、`api/app/services/watermark_engine.py`。

当前逻辑顺序：

1. 先扫描斜向平铺水印；确认后素材被标记为跳过，不进入正式剪辑。
2. 进行产品身份绑定的 Logo/Icon/OCR 水印 census。
3. 区分固定水印、动态水印和未分类水印；同一视频允许同时存在多层水印。
4. 固定水印进入源区域清理，再合成自有品牌 Icon/名称。
5. 动态水印使用产品对应的 Logo/Icon 和动态参考轨迹验证，禁止把动态水印当固定水印覆盖。

### 3.3 动态水印处理

主要实现：`api/app/branding_moving_watermark_insight.py`、`api/app/propainter_adapter.py`。

生产链路为：

```text
源产品身份
 -> 动态视觉模板扫描
 -> 轨迹连续性/运动/持久性验证
 -> 逐帧 bbox mask
 -> ProPainter-Webui inference_propainter.py
 -> 残留相关性 QA
 -> 通过后才交给固定水印/片尾合成
```

ProPainter 失败或残留 QA 不通过时，不会静默降级为模糊、淡化或自有品牌覆盖，而是拒绝把不完整结果当成成功。

### 3.4 片中宣传和片尾

语义结构和片段规划在 `branding_v09.py` 中完成，依据 OCR、Whisper、视觉布局和 CTA 信号生成 `mid_promo_replace` / `end_card_replace`。当 Whisper/VLM 不可用时，当前代码还会对 AppStore、Google Play、Watch Exclusively 等末尾 CTA 做审核级兜底识别，并向前回溯 Logo/宣传起始位置。

### 3.5 渲染与审核

`branding_v09.py` 的生产渲染会生成：

- replacement plan
- dynamic temporal repair report
- render receipt
- residual QC / business QC
- 最终 MP4

前端审核工作台位于 `api/app/static/creative_loop_workbench.html`，API 由 `agent_v019.py` 和 `routers/agent.py` 提供任务、审核和 artifact 访问。

### 3.6 下载与处理并发流水线

采集不再等整批视频下载完成后才开始剪辑。每个视频一旦进入正式 `raw` 目录，就会立即提交到 API 内部线程池进行水印检测、语义分析和渲染；下载线程继续处理后续详情页。默认并发数为 2，可通过 `.env` 中的 `OPERATOR_PROCESS_WORKERS` 设置为 1–4。任务只有在下载结束且线程池全部排空后才进入最终审核状态；单个素材失败会记录在该 item，不会阻塞后续素材。

## 4. 目标电脑要求

### 必需

- Windows 10/11 或 Linux
- Docker Desktop（Windows 使用 WSL2 后端）
- Docker Compose v2
- 至少 8 GB RAM
- 至少 20 GB 可用磁盘空间，不含原始素材和模型缓存
- 能访问 Docker Hub、PyPI 和 Whisper 模型源，或提前准备离线镜像/模型

### 动态水印 ProPainter（可选，但生产环境建议安装）

动态水印修复是否能使用 GPU，不能只看“电脑有独立显卡”。必须同时满足显存、驱动、PyTorch 加速包、模型和 Docker GPU 可见性要求。以下是本项目的实际硬件分级（显存是单张卡的可用显存，不是系统内存）：

| 级别 | 明确硬件门槛（示例） | 后端 | 适用范围 |
| --- | --- | --- | --- |
| CPU/OpenCV | 无独立 GPU，或显存低于 6 GB；建议 8 核 CPU、16 GB RAM | `opencv` / `cpu` | 可运行检测和通用时序修复；速度较慢，残差 QA 不通过时会拒绝出片 |
| CUDA 最低可用 | NVIDIA GTX 1660/1660 Super 6 GB、RTX 2060 6 GB、RTX 3050 8 GB 或更新且显存不低于 6 GB 的同等级型号 | `cuda` + ProPainter | 动态水印 CUDA 推理的最低建议档；适合短片段和 720p，实际长度受 `PROPAINTER_SUBVIDEO_LENGTH` 影响 |
| CUDA 推荐 | NVIDIA RTX 3060 12 GB、RTX 4060/4060 Ti 8/16 GB、RTX 4070 12 GB 或同等级型号 | `cuda` + ProPainter | 720p 日常处理和较长片段；8 GB 是 720p 的推荐起点，12 GB 更稳妥 |
| CUDA 高规格 | NVIDIA RTX 3080 12 GB、RTX 4080 16 GB、RTX 4090 24 GB 或同等级型号 | `cuda` + ProPainter | 1080p、较长视频或多个任务连续处理；仍需按实际模型和片段长度调参 |
| AMD ROCm | Linux 下处于当前 ROCm/PyTorch 支持范围的 AMD 卡，建议 RX 6700 XT 12 GB、RX 6800 16 GB 或更新型号 | `rocm`；不满足时回退 OpenCV | 可使用 ROCm 的动态修复；Windows 对 ROCm 的支持不作为本项目验收条件 |
| DirectML | 支持 DirectML 的独立 GPU，建议显存不低于 6 GB，例如 Intel Arc A750 8 GB | `directml` / OpenCV | 通用轨迹修复路径；不等同于 CUDA ProPainter，速度和效果取决于驱动 |

**明确结论：**

- 6 GB 是 CUDA/ProPainter 的最低建议显存；生产环境建议 8 GB 起步，1080p 或长视频建议 12 GB 起步。
- NVIDIA GeForce MX250 2 GB、MX150 2 GB、GT 1030 2 GB 等低于 6 GB 的显卡，不列入 CUDA/ProPainter 支持配置。即使 `nvidia-smi` 能识别，也可能因显存不足或新版本 PyTorch 不再支持其计算能力而回退到 CPU/OpenCV。
- 低显存卡仍可运行采集、检测、固定水印和 CPU/OpenCV 动态水印路径；它们不能作为“CUDA/ProPainter 无痕修复已验证”的验收机器。
- 上述型号是建议门槛，不是对任意分辨率、片段长度和并发数的性能保证。显存占用还受 `PROPAINTER_SUBVIDEO_LENGTH`、输入分辨率、模型权重和并发任务数影响。

ProPainter-Webui 源码必须包含 `ProPainter/inference_propainter.py` 和模型权重；API 容器必须能访问 ProPainter 目录，Windows 路径要通过 Compose volume 挂载到容器内路径。

## 5. 迁移步骤

### 5.1 解压源码包

将源码包解压到目标电脑，例如：

```powershell
D:\creative-loop-floatboat
```

不要把压缩包解压到带中文或空格的深层路径，尤其是 ProPainter 模型目录。

### 5.2 配置环境变量

```powershell
Copy-Item .env.example .env
```

按需修改 `.env`。默认 CPU Whisper 可以直接运行；不要将真实 Mintegral 凭据写进源码或提交到 Git。

如需发布能力：

```powershell
Copy-Item workspace/config/mintegral.env.example workspace/config/mintegral.env
```

### 显卡与加速后端要求

| 后端 | 运行时与硬件要求 | 配置 | 探测成功标识 |
| --- | --- | --- | --- |
| NVIDIA CUDA | 6 GB 以上 NVIDIA 显卡、匹配驱动、CUDA 版 PyTorch、ProPainter 模型 | `PROPAINTER_BACKEND=cuda` | `propainter_cuda_configured` |
| AMD ROCm | Linux + 当前 ROCm/PyTorch 支持的 AMD 显卡，建议 8 GB 以上 | `PROPAINTER_BACKEND=rocm` | `propainter_rocm_configured` |
| DirectML | 支持 DirectML 的 GPU，建议 6 GB 以上；需 `torch-directml` | `PROPAINTER_BACKEND=directml` | `opencv_temporal_mask_configured` |
| CPU/OpenCV | 无 GPU 或 GPU 不满足上述门槛；需要 Python、OpenCV、FFmpeg | `PROPAINTER_BACKEND=cpu` 或 `opencv` | `opencv_temporal_mask_configured` |

`PROPAINTER_BACKEND=auto` 会按当前环境自动选择后端。使用 ProPainter 时，还需要配置 `PROPAINTER_ROOT`、`PROPAINTER_PYTHON` 以及对应模型文件；通用后端仍会执行动态水印轨迹和残差质检。

填写新申请的凭据。当前旧机器上的凭据已经在迁移排查中暴露，应轮换后再使用。

### 5.3 启动服务

```powershell
docker compose -f compose.yaml up -d --build
docker compose -f compose.yaml ps
Invoke-RestMethod http://localhost:8000/health
```

也可以使用项目启动器：

```powershell
.\creative-loop-control.cmd start
.\creative-loop-control.cmd status
.\creative-loop-control.cmd rebuild
```

### 5.4 首次登录 Insight

浏览器容器启动后访问：`http://localhost:7900/?autoconnect=1&resize=scale`。在目标电脑重新完成 Insight 登录。不要直接复制旧的 `browser-state`，除非明确接受 Cookie 和会话迁移风险。

### 5.5 安装 ProPainter（可选）

后端可通过 `PROPAINTER_BACKEND` 选择 `auto`、`cuda`、`rocm`、`directml`、`cpu` 或 `opencv`。`auto` 会根据当前 Python 环境自动选择；DirectML 和 OpenCV 使用通用逐帧 mask 修复，并继续执行动态水印残差质检。

推荐把 ProPainter 放在目标主机目录，例如 `D:\models\ProPainter-Webui`，并按其项目说明创建独立 Python/CUDA 环境。然后在 `compose.yaml` 中增加只读挂载和环境变量，例如：

```yaml
volumes:
  - D:/models/ProPainter-Webui:/opt/ProPainter-Webui:ro
environment:
  PROPAINTER_ROOT: /opt/ProPainter-Webui
  PROPAINTER_PYTHON: /opt/propainter-env/Scripts/python.exe
```

Windows Docker Desktop 对 GPU/CUDA/ROCm/DirectML 的支持取决于驱动、运行时和镜像配置；API 会根据 `PROPAINTER_BACKEND` 使用可用的修复后端，并对输出执行统一残差质检。

### NVIDIA 显卡并不等于 ProPainter 已可运行

仅安装 NVIDIA 显卡驱动，不能证明动态水印无痕修复链路已经具备。完整运行 CUDA/ProPainter 还必须同时满足：

1. 主机安装与显卡匹配的 NVIDIA 驱动，并且 `nvidia-smi` 能正常列出设备。
2. ProPainter 使用的 Python 环境安装了 CUDA 版 PyTorch，且 `torch.cuda.is_available()` 返回 `True`。
3. `PROPAINTER_ROOT` 指向包含 `ProPainter/inference_propainter.py` 的目录，`PROPAINTER_PYTHON` 指向上述 Python 环境。
4. ProPainter 所需模型权重、FFmpeg 和依赖包均已就绪。
5. Docker 部署时，API 容器能够访问宿主机 GPU；宿主机可见 GPU 不代表容器内自动可见。
6. 显存和系统内存应能容纳当前视频分辨率、帧段长度及模型推理；实际可处理规格取决于模型、分辨率和 `PROPAINTER_SUBVIDEO_LENGTH` 配置。

迁移后建议在 API 容器内执行以下检查：

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
docker compose exec creative-loop-api python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
docker compose exec creative-loop-api python -c "from app.propainter_adapter import probe_propainter_backend; print(probe_propainter_backend())"
```

只有探测结果显示 `propainter_cuda_configured`，并且动态水印残差 QA 通过，才表示 CUDA/ProPainter 路径完整可用；否则应根据探测结果修复驱动、Python、模型或容器挂载配置，不能仅依据“电脑有 NVIDIA 显卡”判断已配置完成。

如果 `nvidia-smi` 能看到显卡，但第二条命令输出 `False`，说明容器内 CUDA 不可用；如果第三条命令输出 `opencv_temporal_mask_configured`，说明当前实际走的是 CPU/OpenCV 通用路径，而不是 ProPainter。显存低于 6 GB 时，即使强制设置 `PROPAINTER_BACKEND=cuda`，也不应作为生产配置使用。

### 没有可用专用 GPU 时的处理方案

当 CUDA、ROCm 或 DirectML 不可用时，系统会在 `PROPAINTER_ALLOW_CPU=1`（默认）下切换到 OpenCV 逐帧 mask + 轨迹修复后端。该后端仍然会：

- 使用已验证的动态水印运动轨迹生成逐帧 mask；
- 对水印区域执行时序/邻帧修复，不添加自有品牌覆盖；
- 执行残差相关性 QA，只有通过 QA 才交给正式合成。

如果 CPU/OpenCV 修复的残差 QA 未通过，系统会保留诊断预览并拒绝生成不可靠成片，这属于安全拦截，不是跳过动态水印处理。需要更强修复能力时，可将 ProPainter 部署到另一台具备可用加速运行时的机器，再通过 `PROPAINTER_ROOT`/`PROPAINTER_PYTHON` 接入；也可以选择 `PROPAINTER_BACKEND=opencv` 明确使用本机通用后端。

## 6. 验证清单

```powershell
docker compose -f compose.yaml ps
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/openapi.json | Out-File openapi-target.json
docker compose -f compose.yaml logs --tail=200 creative-loop-api
```

然后用一条短视频验证：

1. 详情页/采集能生成产品身份信息。
2. 无斜向平铺水印的视频进入后续流程；有斜向平铺水印的视频被过滤。
3. 固定水印生成固定层处理回执。
4. 动态水印生成动态轨迹和时序修复回执；ProPainter 失败时任务明确失败，不生成假成功结果。
5. 片尾/片中宣传段在 replacement plan 中有对应 action。
6. 前端详情页可以播放完整视频。

## 7. 常见问题

### `no configuration file provided: not found`

命令没有在项目根目录执行。先进入含 `compose.yaml` 的目录，或显式指定：

```powershell
docker compose -f D:\creative-loop-floatboat\compose.yaml up -d --build
```

### API 代码修改后没有生效

API 镜像没有热重载，需要重新构建：

```powershell
docker compose -f compose.yaml up -d --build creative-loop-api
```

### Whisper 报模型缓存错误

检查网络和 `model-cache` 挂载；首次启动允许模型下载。也可以先关闭语义 Whisper，让视频进入人工审核级兜底识别。

### 动态水印没有处理

检查 census 的 `verified_dynamic_identity.track_count`、ProPainter backend、temporal report 和 residual QA。只有检测到并验证轨迹、ProPainter 输出存在且残留 QA 通过，才会进入正式输出。

### 视频剪辑耗时过长

先确认 ProPainter、CUDA 驱动、源视频参数和 `PROPAINTER_SUBVIDEO_LENGTH` 配置是否正确。

## 8. 不应迁移的内容

- `.env` 和 `workspace/config/mintegral.env`：包含本机配置/第三方凭据。
- `browser-state`：包含登录 Cookie 和浏览器会话。
- `workspace/raw`、`processed`、`review`、`output`：体积大，按业务需要选择性复制。
- `model-cache`：模型缓存可重新下载。
- `api/app/__pycache__` 和所有 `.bak`/`.backup` 历史源码：不是运行时必需文件，容易造成版本混淆。

## 9. 包内容说明

源码包名称以 `creative-loop-floatboat-migration-YYYYMMDD.zip` 命名，包含：

- 当前有效 API 代码和 Docker 构建文件
- Compose、启动器、前端工作台
- 品牌注册表、产品映射、动态水印参考图
- 自有品牌 Icon/Logo/Mid Promo/End Card
- 测试脚本和本迁移文档
- `.env.example` 与 `mintegral.env.example`

包内不包含真实凭据、浏览器登录态、模型缓存、历史视频和生成报告。
