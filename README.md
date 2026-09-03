# Vision Inspector Agent

一个以 **YOLOv8 视觉检测模型为核心，逐步向完整 Agent 应用演进**的工程化练习项目。

当前阶段主要完成 YOLOv8 推理核心的模块化、配置管理、统一结果结构、批量预测以及自动化测试。模型加载、图片处理、结果结构和命令行输出不再集中在一个脚本中，并使用 Git/GitHub 和 pytest 建立基础的软件工程开发流程。

当前阶段版本：

```text
v0.1.0 - YOLOv8 Inference Core
```

---

## 目录结构

```text
vision-inspector-agent/
│
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── api/                    # 接口层（路由 / 依赖）
│   ├── core/                   # 核心配置与日志
│   ├── database/               # 数据库基础设施（连接 / 会话 / 表结构）
│   ├── repositories/           # 数据访问层
│   ├── services/               # 业务层（流程编排与事务边界）
│   ├── schemas/                # Pydantic 模型
│   ├── vision/                 # AI 视觉能力（模型加载 / 预测）
│   └── utils/                  # 通用工具
│
├── tests/
│   ├── unit/                   # 单元测试
│   ├── integration/            # 集成测试（真实 SQLite）
│   └── api/                    # 接口测试（TestClient）
│
├── scripts/                    # 运维 / 命令行脚本
├── data/                       # SQLite 数据库文件
├── models/                     # 模型权重
├── outputs/                    # 批量预测输出
├── sample_data/                # 示例图片
├── notes/                      # 每周学习记录
├── ultralytics/                # 定制 fork（勿删）
│
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

其中：

* `app/main.py`：FastAPI 应用入口（启动命令 `uvicorn app.main:app`）；
* `app/api/`：路由层，只负责请求/响应与参数校验；
* `app/services/`：业务层，负责流程编排与事务边界；
* `app/repositories/`：数据访问层，只执行 SQL；
* `app/vision/`：模型加载与推理，不依赖数据库；
* `app/core/config.py`：项目配置（环境变量覆盖）；
* `app/schemas/`：统一 Pydantic 结构；
* `tests/`：pytest 自动化测试（unit / integration / api 分层）；
* `notes/`：每周学习和项目开发记录；
* `outputs/`：保存批量预测生成的 JSON 结果；
* `scripts/predict_cli.py`：命令行预测入口（原根目录 `main.py` 迁移至此）。

---

## 定制 Ultralytics

项目根目录保留了一份本地：

```text
ultralytics/
```

源码。

这不是普通的官方 Ultralytics 副本，而是当前 `best.pt` 所依赖的定制 fork。

模型网络中使用了包括：

```text
EIEStem
CARAFE
C2f_DCNv3
```

等自定义模块，因此当前项目运行时必须能够加载这份本地源码。

从项目根目录运行程序时：

```python
from ultralytics import YOLO
```

会优先加载：

```text
vision-inspector-agent/ultralytics/
```

而不是 Python 环境中的普通官方 `ultralytics` 包。

因此当前阶段不要删除该目录。

---

## DCNv3 CPU 回退

模型中的 DCNv3 原本依赖 CUDA 编译扩展。

为了使没有 CUDA 环境的机器也能够完成推理，本项目对定制 Ultralytics fork 做了两处最小修改。

位置：

```text
ultralytics/nn/extra_modules/ops_dcnv3/
```

### 1. `functions/dcnv3_func.py`

当 DCNv3 CUDA 扩展没有安装时，为：

```text
dcn_version
```

提供默认值，避免运行时出现 `NameError`。

### 2. `modules/dcnv3.py`

当：

```python
torch.cuda.is_available() == False
```

时：

```text
DCNv3
DCNv3_DyHead
```

会回退到 fork 自带的纯 PyTorch：

```text
dcnv3_core_pytorch
```

实现。

因此：

```text
有 CUDA + 已编译扩展
        ↓
CUDA DCNv3

没有 CUDA
        ↓
纯 PyTorch DCNv3
```

CPU 推理速度会明显慢于 GPU，但可以保持基本推理功能。

---

# 环境管理

本项目使用 **uv** 管理 Python 环境和依赖。

项目包含：

```text
pyproject.toml
uv.lock
.python-version
```

不使用传统的：

```text
pip install -r requirements.txt
```

作为主要环境管理方式。

---

## 安装依赖

克隆项目后，在项目根目录运行：

```bash
uv sync
```

uv 会根据：

```text
pyproject.toml
+
uv.lock
```

同步项目环境。

默认虚拟环境位于：

```text
.venv/
```

`.venv/` 不提交 Git。

---

## 添加依赖

普通运行依赖：

```bash
uv add package-name
```

开发依赖：

```bash
uv add --dev package-name
```

例如 pytest：

```bash
uv add --dev pytest
```

---

# 运行项目

建议统一通过：

```bash
uv run
```

运行项目命令。

这样 uv 会确保当前项目环境与依赖状态保持同步。

---

## 单张图片预测

```bash
uv run python -m scripts.predict_cli sample_data/test.jpg
```

结果以 JSON 形式打印到终端。

---

## 批量预测

```bash
uv run python -m scripts.predict_cli sample_data
```

预测结果保存到：

```text
outputs/predictions_时间戳.json
```

---

## 常用参数

设置置信度：

```bash
uv run python -m scripts.predict_cli sample_data --conf 0.5
```

指定 CPU：

```bash
uv run python -m scripts.predict_cli sample_data --device cpu
```

设置输入尺寸：

```bash
uv run python -m scripts.predict_cli sample_data --imgsz 640
```

指定模型：

```bash
uv run python -m scripts.predict_cli sample_data/test.jpg --model models/best.pt
```

指定输出：

```bash
uv run python -m scripts.predict_cli sample_data/test.jpg --output result.json
```

完整参数可以查看：

```bash
uv run python -m scripts.predict_cli -h
```

---

# 配置

配置集中在：

```text
app/core/config.py
```

代码中不直接写死模型路径、阈值和设备等参数，可以通过环境变量覆盖。

| 环境变量                | 默认值              | 说明               |
| ------------------- | ---------------- | ---------------- |
| `VISION_MODEL_PATH` | `models/best.pt` | 模型权重路径           |
| `VISION_CONFIDENCE` | `0.25`           | 置信度阈值            |
| `VISION_IOU`        | `0.45`           | NMS IoU 阈值       |
| `VISION_IMAGE_SIZE` | `640`            | 推理输入尺寸           |
| `VISION_DEVICE`     | 空（自动）            | 推理设备，如 `cpu`、`0` |
| `VISION_OUTPUT_DIR` | `outputs/`       | 批量结果输出目录         |

---

# 统一返回结构

核心推理接口：

```python
app.vision.predictor.predict_image(image_path) -> DetectionResult
```

返回统一结构：

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

其中：

```text
bbox
```

采用：

```text
[x1, y1, x2, y2]
```

整数像素坐标。

`confidence` 保留 4 位小数。

`inference_time_ms` 保留 1 位小数。

---

# 自动化测试

第二阶段开始使用 **pytest** 为推理核心建立自动化测试。

pytest 作为开发依赖由 uv 管理：

```bash
uv add --dev pytest
```

运行测试：

```bash
uv run pytest -v
```

当前测试重点覆盖：

* 正常图片能够完成推理；
* 不存在的图片能够返回明确异常；
* 损坏图片能够被识别；
* 推理结果包含正确的图片尺寸；
* 推理结果能够返回统一的 detections 结构；
* 空检测结果能够被正常处理；
* 配置和结果结构符合预期；
* 异常输入不会导致不可控程序崩溃。

---

## pytest 收集范围

项目根目录存在完整的定制：

```text
ultralytics/
```

源码。

其中包含大量第三方：

```text
test_*.py
```

文件。

如果直接让 pytest 递归扫描整个仓库，会错误收集：

```text
ultralytics/nn/extra_modules/...
```

中的 DCNv4、Triton、Cutlass、Selective Scan 等第三方测试。

因此在 `pyproject.toml` 中限制 pytest 只收集项目自己的：

```text
tests/
```

测试。

例如：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = [
    ".venv",
    "ultralytics",
    "outputs",
    "models",
]
```

---

## Windows 临时目录问题

部分 Windows 环境中，pytest 默认临时目录：

```text
C:\Users\<user>\AppData\Local\Temp\pytest-of-<user>
```

可能因为 ACL 权限异常出现：

```text
PermissionError: [WinError 5] 拒绝访问
```

这不是 YOLO 推理测试本身失败，而是 pytest 无法创建自己的测试临时目录。

如果遇到该问题，可以临时指定项目内部目录：

```bash
uv run pytest -v --basetemp=".pytest_tmp"
```

此时 pytest 会使用：

```text
vision-inspector-agent/.pytest_tmp/
```

作为临时目录。

`.pytest_tmp/` 应加入 `.gitignore`：

```gitignore
.pytest_tmp/
```

---

# Git 开发流程

本项目从第二阶段开始使用 Git/GitHub 管理开发过程。

当前基础工作流：

```text
Issue
  ↓
创建功能分支
  ↓
开发 / 测试
  ↓
Commit
  ↓
Push
  ↓
Pull Request
  ↓
Code Review
  ↓
Merge into main
```

例如测试开发使用：

```text
feature/add-tests
```

分支。

---

## 创建开发分支

```bash
git branch feature/add-tests
git checkout feature/add-tests
```

完成开发后：

```bash
git add .
git commit -m "test: add YOLOv8 inference tests"
git push -u origin feature/add-tests
```

然后在 GitHub 创建 Pull Request：

```text
feature/add-tests
        ↓
       main
```

---

## Issue 与 Pull Request

GitHub Issue 用于记录项目待完成任务，例如：

```text
Add pytest coverage for YOLOv8 inference
```

Pull Request 用于将功能分支中的修改合并到：

```text
main
```

如果 PR 完成的内容对应某个 Issue，例如：

```text
#1
```

可以在 PR 描述中写：

```text
Closes #1
```

当 PR 成功合并到默认分支后，GitHub 会自动关闭对应 Issue。

---

# 当前阶段验收

## 推理核心

* [x] 模型加载与推理解耦；
* [x] 模型进程内只加载一次；
* [x] 推理函数不直接 `print`；
* [x] 使用统一 `DetectionResult` 返回结构；
* [x] 不存在或损坏图片能够抛出明确异常；
* [x] 模型路径和推理参数支持配置；
* [x] 支持目录批量预测；
* [x] 支持 CPU 环境运行定制 DCNv3 模型。

## 工程化

* [x] 使用 uv 管理项目环境；
* [x] 使用 `pyproject.toml` 管理依赖；
* [x] 使用 `uv.lock` 锁定依赖版本；
* [x] 使用 Git 管理代码版本；
* [x] 使用功能分支开发；
* [x] 建立 pytest 自动化测试；
* [x] pytest 只收集项目自己的测试；
* [x] 建立 GitHub Issue；
* [x] 建立 Pull Request 工作流。

---

# 当前版本

```text
v0.1.0 - YOLOv8 Inference Core
```

当前阶段目标是建立一个：

```text
可运行
+
可配置
+
可测试
+
可版本管理
```

的 YOLOv8 推理核心。

后续将在此基础上继续增加：

```text
FastAPI
    ↓
模型服务化
    ↓
数据库
    ↓
LLM
    ↓
RAG
    ↓
Agent
```

最终逐步形成完整的：

```text
VisionInspector Agent
```

## FastAPI API

启动开发服务器：

```bash
uv run uvicorn app.main:app --reload


## API Schema Validation

VisionInspector uses Pydantic models to define and validate
the data contract between the YOLO inference layer and FastAPI.

Current schemas include:

- Detection
- DetectionResult
- PredictionResponse
- ModelInfoResponse
- ErrorResponse

Detection confidence and bounding box values are validated
before being exposed through the API.

## Architecture


API Layer

↓

Service Layer

↓

Repository Layer

↓

SQLite Database


## Features

- YOLO inference
- Prediction persistence
- Prediction history
- Transaction management
- Automated testing