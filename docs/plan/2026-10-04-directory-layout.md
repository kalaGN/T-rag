# 目录整理计划与记录

范围：用户已批准的目录结构整理与独立业务 API 移除，不包含通用知识库功能实现。

## 任务

- [x] 将 LLM 和本地 Embedding 服务归入 models，Embedding 客户端构建从存储模块移到 models/embedding.py；同步导入和测试补丁路径。
- [x] 将 store 更名为 storage，保留 Qdrant 和 DocStore 实现，不改变存储数据。
- [x] 将单元测试按对应模块归入 tests/unit；移除已删除 API 的测试。
- [x] 删除独立 API 模块、CLI 和脚本入口，开发脚本拒绝无效模式后再启动依赖。
- [x] 增加根 README，更新当前使用文档。历史设计和架构图仍作为旧版资料保存。
- [x] 验证：29 项单元测试通过（移除 3 项独立 API 测试），模块 CLI 帮助正常，所有源码模块可导入，脚本语法与 diff 空白检查通过。未执行真实问答或浏览器测试，本次不改变页面交互。

knowledge、integration、e2e 目录在对应功能或用例落地时建立，不提前添加空目录。项目发行名称、环境变量和 Qdrant 集合暂保留，避免与目录整理混合迁移。

FastAPI、uvicorn 仍保留在依赖中；Gradio 本身依赖这些组件，移除独立业务 API 不代表移除页面运行所需的 HTTP 框架。
