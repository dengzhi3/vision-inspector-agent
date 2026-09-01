# Week 4 - Pydantic and API Schema Design

## 1. 本周目标
把 YOLO 推理结果和 FastAPI 响应从普通 dict 改造成由 Pydantic 管理的数据结构，让接口的输入输出变得规范、安全且自带文档。

## 2. 学习内容

### BaseModel
我的理解：**数据结构的“模具”**。
以前用字典（dict）全靠手写键名，容易拼写错误且没有类型提示。用 BaseModel 子类定义后，就能把原始 JSON/字典**转换成带属性的 Python 对象**（如 `result.confidence`），写代码时有智能补全，非常安全。

### Field
我的理解：**字段的“增强器”**。
用来给字段添加元数据或约束，比如默认值（`default=0.0`）、取值范围（`ge=0, le=1`）、正则匹配，还能加 `description` 让 Swagger 文档变得更清晰。

### Nested Model
我的理解：**“俄罗斯套娃”**。
一个模型里面套另一个模型（比如 `PredictionResponse` 里包含一个 `DetectionResult`，`DetectionResult` 里又包含多个 `Detection`）。这样就能完美映射复杂的嵌套 JSON 结构，层层都有类型保护。

### field_validator
我的理解：**字段的“安检门”**。
在数据赋值给字段之前，运行自定义逻辑去检查或修改它（例如校验置信度必须在 0~1 之间 mode="after"，或者把图片后缀统一转小写mode = "before"）。如果不合法就抛出 `ValueError`，阻止脏数据进入系统。

### response_model
我的理解：**外卖的“打包盒”**。
专门用在 FastAPI 的路由装饰器里。它**只负责成功（2xx）时**把业务函数返回的数据，按照模型规则过滤、转换后交给客户端。没在模型里定义的字段会被自动剔除，防止敏感数据泄露。

### ConfigDict(extra="forbid")
我的理解：**模型的“守门员”**。
默认情况下，Pydantic 能接受多余的字段。加上 `extra="forbid"` 后，只要用户传了模型中没定义的字段，立刻报错。强制前后端严格遵守接口契约，防止前端乱传参数。

## 3. 项目变化

### schemas.py
这是本周的核心，定义了一整套“契约”：
- `HealthResponse`：健康检查返回值（`{"status": "ok"}`）。
- `Detection`：单个目标（框坐标、置信度、类别名称）。
- `DetectionResult`：一张图的检测结果（包含图片尺寸、所有目标列表、目标总数）。
- `ModelInfoResponse`：模型元信息（路径、输入尺寸、类别映射）。
- `PredictionResponse`：最外层统一成功响应（包装状态码、数据等）。
- `ErrorResponse`：最外层统一错误响应（错误码、错误信息）。

### predictor.py
- **原来**：`predict_image` 返回普通 dict，键名全靠硬编码，容易写错。
- **现在**：返回 `DetectionResult` 对象。用 `**result.to_dict()` 或直接赋值构建，类型更安全，IDE 能自动补全字段。

### api.py
- **原来**：路由函数直接返回 `dict`，没有约束。
- **现在**：在装饰器里指定 `response_model=PredictionResponse`，确保成功响应结构永远统一；同时 `responses` 里加上 `ErrorResponse`（虽然需要配合异常处理器才会自动生效，但文档里先画好“蓝图”）。

## 4. 我遇到的问题
1. **`responses` 里定义的 `ErrorResponse` 为什么不自动生效？**  
   因为 FastAPI 只把它当成“广告牌”展示在文档里。如果直接抛出 `HTTPException`，返回的是 `{"detail": "..."}`，根本不符合 `ErrorResponse` 的结构。**解决方式**：必须写一个全局异常处理器，手动把异常转为 `ErrorResponse` 格式返回。

2. **Pydantic v2 序列化报错 `TypeError`？**  
   之前习惯用 `.dict()`，但在 Pydantic v2 中已弃用，必须统一改成 `.model_dump()`。同样，`.parse_obj()` 要改成 `.model_validate()`。

3. **嵌套模型验证失败时，错误信息很懵**  
   如果嵌套层级太深（比如 3 层），Pydantic 报错信息会很长。需要在 `field_validator` 中加清晰的 `raise ValueError("具体的错误提示")`，方便调试。

4. **上传的图片文件名带中文导致后缀提取异常**  
   虽然用了 `Path(file.filename or "").suffix.lower()`，但如果前端传的是纯中文无后缀，`suffix` 为空，后面写临时文件会变成无后缀文件，可能影响 `predict_image` 的读取逻辑。**解决**：在 `api.py` 开头加一个校验，没后缀直接抛 400。

## 5. 本周测试
- **单元测试（未集成 TestClient 前）**：手动在 Python 交互环境里实例化 Pydantic 模型，故意传错类型（如把 `confidence` 传字符串），观察是否触发 `ValidationError`。
- **接口测试（Postman/cURL）**：
  - 正常上传图片 -> 返回结构是否完全匹配 `PredictionResponse`（多字段、少字段、字段类型）。
  - 上传非图片文件（如 .txt） -> 是否触发 400，且返回的 JSON 是否匹配 `ErrorResponse` 格式。
- **边界测试**：上传一张纯黑图或超大尺寸图，看 `predictor` 内部的字段（如框坐标）是否有负数或 `inf`，Pydantic 能否拦截。

## 6. 我的总结
**Pydantic 的本质是“移动作业”**。它把原本散落在各个函数里的数据校验、类型转换、字段过滤工作，**统一移动到了数据进入或离开系统的边界上**（请求参数、响应输出）。这样做的好处是：业务逻辑（`predictor.py`）只需要关心“怎么推理”，再也不用操心“数据长什么样”。

**关于 `response_model` 和 `responses`**：前者是强制执行的“交通警察”（管成功数据），后者是挂在墙上的“指示牌”（管文档）。想只写一套异常代码就覆盖所有错误响应，必须补上 `@app.exception_handler`，否则文档和实际返回永远是两张皮。