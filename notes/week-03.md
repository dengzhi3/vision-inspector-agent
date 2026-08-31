# Week 03 - HTTP and FastAPI

## 1. 本周学习内容

- HTTP Request / Response
- GET / POST / DELETE
- HTTP Status Code
- JSON
- REST API
- FastAPI
- Route
- UploadFile
- HTTPException
- Swagger / OpenAPI

## 2. 本周项目增量

新增 FastAPI 服务：

- GET /health
- GET /model/info
- POST /predictions

POST /predictions 可以上传图片并复用已有的
app.predictor.predict_image() 完成 YOLOv8 推理。

## 3. 当前调用链

客户端
→ HTTP Request
→ FastAPI
→ predictor.py
→ model_loader.py
→ YOLOv8
→ DetectionResult
→ JSON Response

## 4. 本周理解

### GET

主要用于获取资源或信息。

### POST

用于向服务提交数据。本项目通过 POST /predictions 上传图片。

### HTTP 状态码

- 200：请求成功
- 400：客户端请求错误
- 404：资源不存在
- 500：服务器内部错误

### FastAPI

FastAPI 负责 HTTP 接口和路由，不直接实现 YOLO 推理逻辑。

## 5. 本周遇到的问题

记录实际遇到的问题及解决方式。

## 6. 下一周

Pydantic 与接口设计：

- BaseModel
- 字段约束
- 嵌套模型
- Validator
- 严格校验
- 统一错误格式