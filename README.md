```markdown
# Optimize

本项目为**优化算法核心仓库**，实现了基于 **SolidWorks 2023** 几何模型与 **.NET 8.0** 运行时的联合优化流程。  
算法已封装为 **REST API**，支持 Python 端一键调用。

---

## 📌 前置依赖

| 软件 / 运行时 | 版本  |
|---------------|-------|
| SolidWorks    | 2023  | 
| .NET          |       |

---

## 🚀 运行步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/xmustu/optimize.git
   cd optimize
   ```

2. **安装 Python 依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动服务**
   ```bash
   python main.py
   ```
   默认将在本地启动 FastAPI 服务（`http://127.0.0.1:9100/docs` 可查看交互式文档）。



## 🔍 示例调用

```bash
curl -X POST http://127.0.0.1:9100/optimize \
  -H "Content-Type: application/json" \
  -d '{"model_path": ""}'
```

