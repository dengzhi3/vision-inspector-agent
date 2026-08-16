# vision-inspector-agent

一个把 YOLOv8 推理代码按职责拆分的练习项目：模型加载、图片处理、结果结构和输出打印不再挤在同一个脚本里。

## 目录结构

```text
vision-inspector-agent/
│
├── app/
│   ├── __init__.py
│   ├── config.py        # 配置：路径、阈值、设备（支持环境变量覆盖）
│   ├── schemas.py       # 统一返回结构 Detection / DetectionResult
│   ├── model_loader.py  # 模型加载，进程内只加载一次
│   └── predictor.py     # 单图 / 目录批量预测，返回统一结构
│
├── models/
│   └── best.pt          # YOLOv8 权重（单类别 detect: ill）
│
├── sample_data/
│   ├── test.jpg
│   └── test2.jpg
│
├── outputs/             # 批量预测 JSON 结果输出目录
│
├── main.py              # 命令行入口（唯一允许 print 的入口）
├── pyproject.toml
├── uv.lock
└── .gitignore
```

另外项目根目录还有一份本地 `ultralytics/` 源码（定制版 fork），`best.pt` 的网络里用到了其中的自定义模块（`EIEStem`、`CARAFE`、`C2f_DCNv3` 等），运行时请从项目根目录执行，这样 `import ultralytics` 会优先加载这份本地源码。

## 环境准备

```bash
uv sync
```

依赖中除官方 ultralytics 外，还包含了这份定制 fork 需要额外安装的包（timm、einops、efficientnet-pytorch、dill、pywavelets、scipy、pandas、seaborn、py-cpuinfo、shapely、tqdm）。

> 注意：模型中的 DCNv3 算子原本是 CUDA 编译扩展。本仓库对该 fork 做了两处最小补丁（见 `ultralytics/nn/extra_modules/ops_dcnv3/`），使无 CUDA 环境回退到 fork 自带的纯 PyTorch 实现 `dcnv3_core_pytorch`：
> 1. `functions/dcnv3_func.py`：DCNv3 扩展未安装时给 `dcn_version` 一个默认值，避免 NameError；
> 2. `modules/dcnv3.py`：`DCNv3` / `DCNv3_DyHead` 在 `torch.cuda.is_available() == False` 时走纯 PyTorch 前向。
>
> CPU 推理会比 GPU 慢，但功能一致。机器有 GPU 且编译好扩展时，代码会自动走 CUDA 路径。

## 用法

单张图片（JSON 打印到终端）：

```bash
python main.py sample_data/test.jpg
```

批量预测一个图片目录（结果保存到 `outputs/predictions_时间戳.json`）：

```bash
python main.py sample_data
```

常用参数：

```bash
python main.py sample_data --conf 0.5 --device cpu --imgsz 640
python main.py sample_data/test.jpg --model models/best.pt --output result.json
```

## 配置

配置集中在 `app/config.py`，代码里不写死参数，均可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VISION_MODEL_PATH` | `models/best.pt` | 模型权重路径 |
| `VISION_CONFIDENCE` | `0.25` | 置信度阈值 |
| `VISION_IOU` | `0.45` | NMS IoU 阈值 |
| `VISION_IMAGE_SIZE` | `640` | 推理输入尺寸 |
| `VISION_DEVICE` | 空（自动） | 推理设备，如 `cpu`、`0` |
| `VISION_OUTPUT_DIR` | `outputs/` | 批量结果输出目录 |

## 统一返回结构

`app.predictor.predict_image(image_path) -> DetectionResult` 返回：

```json
{
  "image_width": 1280,
  "image_height": 720,
  "inference_time_ms": 45.0,
  "detections": [
    {
      "class_id": 0,
      "class_name": "crack",
      "confidence": 0.91,
      "bbox": [120, 80, 600, 420]
    }
  ]
}
```

`bbox` 为 `[x1, y1, x2, y2]` 整数像素坐标；`confidence` 保留 4 位小数；`inference_time_ms` 保留 1 位小数。

## 验收对照

- 模型只加载一次：`model_loader.load_model` 使用 `lru_cache` 缓存，进程内同一路径只加载一次；
- 推理函数不直接 print：`predictor` 只返回 `DetectionResult`，打印统一由 `main.py` 负责；
- 错误图片抛出明确异常：`predict_image` 对不存在的路径、非图片格式、无法解码的图片抛出 `InvalidImageError`；
- 配置不写死在代码中：路径、阈值、设备集中在 `app/config.py`，可用环境变量覆盖；
- 目录批量预测：`predict_directory` 遍历目录中所有支持的图片，`main.py sample_data` 一键批量预测。
