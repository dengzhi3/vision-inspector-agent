# Week 02 - Git, GitHub and pytest

## 1. 本周学习内容

- Git 工作区、暂存区、本地仓库
- git add
- git commit
- git branch
- git checkout
- git merge
- git remote
- git push / git pull
- GitHub Issue
- GitHub Milestone
- Pull Request
- pytest
- fixture
- parametrize
- exception testing

## 2. 项目增量

本周为 VisionInspector Agent 的 YOLOv8 推理核心增加了 pytest 测试。

完成：

- 正常图片测试
- 不存在图片测试
- 损坏图片测试
- DetectionResult 测试
- fixture
- 参数化测试

## 3. 本周遇到的问题

### pytest 扫描 ultralytics 源码

原因：

pytest 默认递归收集 test_*.py，导致扫描到 ultralytics 目录中的第三方测试。

解决：

在 pyproject.toml 中设置 testpaths，只测试 tests/。

### Windows pytest 临时目录权限错误

错误：

PermissionError: [WinError 5]

原因：

Windows 默认 pytest 临时目录权限异常。

临时解决：

uv run pytest -v --basetemp=".pytest_tmp"

## 4. 本周理解

Git commit 是本地代码快照。

GitHub Issue 用于记录需要解决的任务。

Pull Request 用于把一个分支中的修改提交给另一个分支进行检查和合并。

pytest 用于自动验证代码行为是否符合预期。

## 5. 下一周

学习：

HTTP
FastAPI
REST API
GET / POST
Swagger