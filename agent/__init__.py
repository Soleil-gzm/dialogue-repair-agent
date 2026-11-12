"""Agent 子系统（可独立开源部分）

与主工程解耦，仅 A1 对照臂需要主工程环境（它本身就是"现有能力基线"）。

目录结构：
    agent/docs/      设计文档
    agent/benchmark/ Benchmark harness（指纹、lock、对照臂）
    agent/fixtures/  公开轨数据（程序化构造）
    agent/tests/     本子系统测试

开源边界见 agent/oss_manifest.txt 与 agent/README.md。
"""
