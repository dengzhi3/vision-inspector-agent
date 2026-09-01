# Week 5 - FastAPI Testing and Async Basics

## 学习内容

- **TestClient**：FastAPI 提供的 `httpx` 测试客户端，用于模拟发送 HTTP 请求，无需真实启动 Uvicorn 服务即可测试路由逻辑。
- **pytest fixture**：用于管理测试的前置依赖（如数据库连接、TestClient 实例），利用依赖注入实现测试环境的复用和隔离。
- **Mock**：用“假对象”替换真实对象，目的是隔离被测代码，避免依赖外部系统（如文件系统、网络、数据库）。
- **patch**：`unittest.mock` 中的强大工具，用于在测试上下文中临时替换某个属性或方法的实现（常见用法：`@patch("module.function")`）。
- **async/await**：Python 异步编程的语法糖，允许在单线程内并发执行 I/O 等待型任务，解决高并发下的阻塞问题。
- **concurrency vs parallelism**：并发是逻辑上的“同时”（任务交替执行），并行是物理上的“同时”（多核真正一起算）。
- **BackgroundTasks**：FastAPI 提供的轻量级后台任务工具，常用于在返回响应后执行发送邮件、写日志等非关键 I/O 操作（注意：是依赖于请求生命周期的短暂任务）。
- **file size limit**：文件大小限制，通常在 Nginx 层或 FastAPI 的 `max_size` 参数中控制，防止恶意超大文件耗尽服务器内存或磁盘。
- **timeout**：超时机制，防止某个耗时操作（如调用外部 API、GPU 推理）无限期挂起，保证系统的鲁棒性。

## 项目新增功能

- **POST /predictions/batch**：批量预测接口。接收多张图片文件（`list[UploadFile]`），将其推入异步任务队列（或使用 `BackgroundTasks`），并立即返回一个 `task_id` 给前端，避免长时间阻塞。
- **GET /predictions/{task_id}/status**：任务状态查询接口。前端拿着 `task_id` 轮询此接口，获取当前任务的 `PENDING` / `RUNNING` / `COMPLETED` / `FAILED` 状态及最终结果。

## Mock YOLO

**为什么 API 测试不真正运行 YOLO：**

1. **执行速度**：真正的 YOLO 模型推理涉及大量矩阵运算，单张图片可能耗时数百毫秒甚至数秒。如果单元测试中真实运行，几十个用例会导致测试套件（Test Suite）运行极慢，违背“快速反馈”原则。
2. **环境依赖**：YOLO 通常依赖 GPU、CUDA 驱动以及巨大的预训练权重文件（几百 MB）。在 CI/CD（持续集成）环境中，往往不具备 GPU 资源，且下载权重会极不稳定。
3. **确定性与可复现性**：真正的模型输出存在微小浮点数误差，且推理速度不稳定。而单元测试需要 **确定性的结果**（比如固定返回 `[{"label": "cat", "conf": 0.9}]`），以便精确断言 API 的响应结构是否完整。
4. **测试分层设计**：这是典型的**单元测试/集成测试分层**思想。API 测试只关心“路由是否正确”、“参数解析是否成功”、“状态码是否符合预期”，而模型的精度验证应属于离线评估阶段，不应耦合在 API 自动化测试中。

## async/await 我的理解

- **`async def`**：定义一个可被暂停和恢复的协程函数。可以把它理解为“具备异步能力的函数蓝图”。
- **`await`**：挂起当前协程，让出 CPU 控制权，等待背后的 I/O 操作（如网络请求、文件读取）完成后，再恢复执行后面的代码。
- **核心原则**：`async/await` 仅对 **I/O-bound（I/O 密集型）** 任务有效。如果 `await` 后面跟着的是一个耗时的 CPU 计算（如 `time.sleep` 或大量循环），它会堵塞整个事件循环，导致“假异步”。
- **我的代码实践**：在 FastAPI 接口中，对于读取文件、请求外部 API 等操作使用 `await`；对于 CPU 密集型任务（如 YOLO 推理），则必须将任务交给 **Celery** 或线程池执行，绝不能直接用 `await` 去等待，否则接口会彻底卡死。

## 并发和并行区别

结合 Python 的特性，我这样区分：

- **并发**：单核单线程内，利用事件循环（Event Loop）在多个任务之间快速切换。虽然微观上某一时刻只有一个任务在执行，但宏观上看起来像是“同时进行”。就像只有一个服务员，在等菜的间隙给另一桌点单。
- **并行**：需要多核 CPU 支持，在同一时刻真正有多个任务同时在运行。Python 由于 **GIL（全局解释器锁）** 的存在，多线程无法实现真正的并行；要利用并行，通常使用 `multiprocessing`（多进程）或 `concurrent.futures.ProcessPoolExecutor`。
- **总结**：我的 `asyncio.gather` 创造了并发，但绝对不是并行。它能提速是因为把网络等待时间重叠利用了，而不是因为多核同时在算。

## 遇到的问题

1. **事件循环冲突**：在使用 `pytest` 测试 `async` 异步接口时，直接调用 `client.get` 会报错（`RuntimeError: Task attached to a different loop`）。解决方法：安装 `pytest-asyncio` 并添加 `@pytest.mark.asyncio` 装饰器，或使用 `httpx.AsyncClient` 进行测试。
2. **超时后任务并未销毁**：使用 `asyncio.timeout` 包裹 GPU 推理代码，超时抛出 `TimeoutError` 后，通过监控发现 GPU 显存并未释放。原因是底层任务只是被标记为 `CancelledError`，但没有真正停止底层 C++ 扩展（PyTorch）的执行。最终我放弃了用 `asyncio` 控制 GPU，转向 Celery 进行超时控制。
3. **Mock 与 Patch 的作用域**：使用 `@patch("module.yolo_model.detect")` 时，注意路径必须是**目标模块引用处**（`where it is looked up`），而非定义处。搞混路径会导致 Mock 不起效。
4. **文件读取位置偏移**：批量读取 `UploadFile` 时，由于文件指针 `tell()` 会随读取移动，在后台任务中直接使用该文件对象会导致空读。需要在传递给后台线程前 `.read()` 提取并重置指针，或直接传递文件字节流。

## Week 6

**SQL and SQLite**

进入持久化存储阶段。之前我们通过内存字典 `{task_id: status}` 存储任务状态，一旦服务重启，所有任务状态都会丢失。

- **学习目标**：
  - 熟悉 **SQLite** 轻量级文件数据库，掌握其基本 DDL（`CREATE TABLE`）和 DML（`INSERT`、`UPDATE`、`SELECT`）语法。
  - 引入 **SQLAlchemy**（或 `aiosqlite`）实现异步数据库操作，将 `TaskStatus` 枚举持久化到数据库表中。
  - 设计 `tasks` 表：包含 `id`（主键）、`status`（状态字符串）、`result`（预测结果 JSON）、`created_at` 和 `updated_at`（时间戳）。
  - 重构 `GET /predictions/{task_id}/status`：从查询内存改为查询 SQLite 数据库。
  - 了解事务（Transaction）概念，确保“更新任务状态”与“存储结果”的原子性。