#!/bin/bash
# 设置女书知识图谱项目的Python环境
# 使用uv作为包管理工具

# 设置项目目录和环境名称
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="nushu_kg"
ENV_DIR="${PROJECT_DIR}/.venv"

echo "===== 女书知识图谱项目环境设置 ====="
echo "项目目录: ${PROJECT_DIR}"
echo "环境名称: ${ENV_NAME}"
echo "环境位置: ${ENV_DIR}"

# 检查是否已安装uv
if ! command -v uv &> /dev/null; then
    echo "正在安装uv工具..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 添加uv到当前shell路径
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 创建虚拟环境
echo "正在创建虚拟环境..."
uv venv "${ENV_DIR}"

# 激活环境
source "${ENV_DIR}/bin/activate"

# 安装依赖
echo "正在安装项目依赖..."
uv pip install -r "${PROJECT_DIR}/env/requirements.txt"

# 安装可选的ipykernel以支持Jupyter笔记本
echo "正在安装Jupyter支持..."
uv pip install ipykernel
python -m ipykernel install --user --name="${ENV_NAME}" --display-name="Python (${ENV_NAME})"

echo "===== 环境设置完成 ====="
echo "使用以下命令激活环境:"
echo "    source ${ENV_DIR}/bin/activate"
echo "女书知识图谱项目环境已准备就绪!"