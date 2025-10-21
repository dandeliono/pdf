# PDF 水印去除网站

这是一个通过 Web 界面上传 PDF 并去除重复背景水印的示例项目，包含 FastAPI 后端和纯前端页面，并支持 Docker 一键部署。实现思路参考了
[「一个去掉PDF背景水印的思路」](https://blog.csdn.net/waitdeng/article/details/140003003)。

## 功能特性

- 支持上传 PDF 文件并按照阈值识别重复背景图片
- 自定义单页覆盖比例阈值与页面重复比例阈值，按需调节检测灵敏度
- FastAPI 提供 REST API，前端通过 Fetch 接口交互
- 单个 Docker 容器即可部署，启动后访问根路径即可使用

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -r backend/requirements.txt
uvicorn app.main:app --reload
```

开发模式下，默认在 `http://127.0.0.1:8000` 打开网页界面。

## Docker 部署

```bash
docker build -t pdf-watermark-remover .
docker run --rm -p 8000:8000 pdf-watermark-remover
```

部署完成后访问 `http://localhost:8000` 即可进入前端页面。

## API 说明

- `POST /api/remove-watermark`
  - 表单字段 `file`: PDF 文件（必填）
  - 表单字段 `coverage_ratio`: 图片覆盖比例阈值，默认 `0.5`
  - 表单字段 `page_ratio`: 背景图片出现页数占比阈值，默认 `0.8`
  - 返回值: 去除背景水印后的 PDF 文件（二进制流）

- `GET /api/health`: 健康检查接口，返回 `{"status": "ok"}`

## 限制说明

该示例通过检测在多页重复出现且覆盖大部分页面的图片来移除背景水印。对于嵌入文本或矢量对象中的水印，可能无法完全移除。
