# ProPainter 动态水印 worker

Floatboat 保留产品专属水印识别与轨迹验证，仅将已验证轨迹转换为逐帧 PNG mask，并调用 `halfzm/ProPainter-Webui/ProPainter/inference_propainter.py` 做时序修复。

配置 API 容器可访问的 ProPainter 源码目录：

```env
PROPAINTER_ROOT=/opt/ProPainter-Webui
PROPAINTER_PYTHON=/opt/conda/envs/propainter/bin/python
PROPAINTER_FP16=1
PROPAINTER_MASK_DILATION=2
PROPAINTER_SUBVIDEO_LENGTH=80
PROPAINTER_TIMEOUT_SECONDS=3600
```

目录必须包含 `ProPainter/inference_propainter.py` 及其模型权重。WebUI 的 SAM/AOT 交互追踪不参与生产检测，避免把人物或字幕误当作水印。
