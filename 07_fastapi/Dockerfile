# ===== 阶段8学习：FastAPI 服务的 Dockerfile =====
# 基础镜像：官方 Python 3.11 精简版（~45MB）
FROM python:3.11-slim

# 容器内工作目录（不存在会自动创建）
WORKDIR /app

# 先只复制依赖文件并安装（利用 Docker 缓存：
# 依赖没变时，改代码重新打包只需几秒，不用重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再复制项目代码
COPY main.py .

# 声明容器对外端口（8000 = uvicorn 端口）
EXPOSE 8000

# 容器启动命令
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
