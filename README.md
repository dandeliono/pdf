# PDF 水印去除网站

这是一个通过 Web 界面上传 PDF 并去除常见文字水印的示例项目，包含 FastAPI 后端和纯前端页面，并支持 Docker 一键部署。

## 功能特性

- 支持上传 PDF 文件并自动检测常见关键字（默认为 `watermark`、`confidential`、`draft`）
- 自定义水印关键字，针对文本水印与批注进行清理
- 可选“矢量模式”和“图像覆盖模式”：图像模式会将页面渲染成图片并覆盖 OCR/文本匹配到的区域，适合处理无法透过 PDF 对象直接删除的水印
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
  - 表单字段 `file`: PDF 文件（与 `upload_id` 二选一）
  - 表单字段 `upload_id`: 通过分片上传接口返回的会话 ID
  - 表单字段 `keywords`: 逗号分隔的关键字列表（可选）
  - 表单字段 `mode`: 处理模式，可选值 `auto`（默认）、`vector`、`raster`
  - 返回值: 去水印后的 PDF 文件（二进制流）

- `GET /api/health`: 健康检查接口，返回 `{"status": "ok"}`

## 限制说明

该示例通过去除文本和批注中包含关键字的内容来实现水印清理。对于图像或复杂矢量水印，可能无法完全移除。
