from fastapi import FastAPI, HTTPException
import subprocess
import json
import os
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from typing import Optional
import uvicorn
import asyncio
import io
import threading
import time
from typing import Optional, Dict, Any, List, Union
from sldwks import start_main
import sys
app = FastAPI(title="几何优化服务",)

# 存储当前运行的任务
active_tasks = {}  # task_id -> conversation_id

class optimize_parameters(BaseModel):
    parameters:Optional[Dict[str, Any]] = Field(
        default=None,
    )
# 任务状态模型
class TaskStatus(BaseModel):
    task_id: str = Field(..., description="任务ID，用于跟踪请求")
    status: str = Field(..., description="任务状态，可能的值包括 'pending', 'running', 'done', 'failed'")
    message: Optional[str] = None  # 可选的状态消息，提供额外信息

# 请求模型
class AlgorithmRequest(BaseModel):
    """几何优化请求模型，包含格式验证"""
    task_id: str = Field(..., description="任务ID，用于跟踪请求")
    conversation_id: str = Field(..., description="对话ID，用于关联会话")
    geometry_description: str = Field(..., description="几何描述文本，要求简洁明了")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选参数字典，用于调整算法行为"
    )

    # @field_validator('geometry_description')
    # def validate_geometry_description(cls, v):
    #     if not v.strip():
    #         raise ValueError('几何描述不能为空')
    #     return v



def check_environment():
    """检查SolidWorks和.NET环境是否就绪"""
    return True
    pass
# 线程标准输出重定向工具
class ThreadOutputRedirector:
    """用于重定向线程中标准输出的上下文管理器"""
    def __init__(self, log_file: str):

        self.log_file = log_file
        self.original_stdout = None
        self.buffer = io.StringIO()  # 临时缓冲区

    def write(self, message: str):
        """重写write方法，添加任务ID前缀并写入文件"""
        # # 在每条输出前添加任务ID和时间戳
        # timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # formatted_msg = f"[{timestamp}][{self.task_id}] {message}"
        # self.buffer.write(formatted_msg)
        # 同步写入文件（确保实时性）
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message)

    def flush(self):
        """刷新缓冲区"""
        self.buffer.flush()

    def __enter__(self):
        """进入上下文，保存原stdout并替换"""
        self.original_stdout = sys.stdout
        sys.stdout = self  # 重定向标准输出到当前实例
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原stdout"""
        sys.stdout = self.original_stdout
        # 处理异常信息（如果有）
        if exc_type:
            error_msg = f"任务执行出错: {exc_val}\n"
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][{self.task_id}] {error_msg}")

def run_algorithm_in_thread(model_path: str):
    backend_log_file = os.path.join(os.path.dirname(model_path), "backend_log.txt")
    try:
        with ThreadOutputRedirector(backend_log_file):
            start_main(model_path)

    except Exception as e:
        pass
    finally:
        pass
@app.get("/health")
async def health_check():
    """健康检查接口，供主后端监控"""
    env_ready = check_environment()
    return {
        "status": "healthy" if env_ready else "unhealthy",
        "dependencies": {
            "solidworks": "available" if env_ready else "missing",
            ".net8.0": "available"  
        }
    }

@app.post("/run-algorithm", response_model=TaskStatus)
async def run_algorithm(request: AlgorithmRequest):
    """调用算法程序处理请求"""
    if not check_environment():
        raise HTTPException(status_code=503, detail="算法依赖环境未就绪（SolidWorks/.NET 8.0）")
    
    """启动算法并返回任务ID，日志通过SSE推送"""
    task_id = request.task_id
    active_tasks[task_id] = "running"


    thread = threading.Thread(target=run_algorithm_in_thread,
                              args=(request.parameters["model_path"],),
                              daemon=True)
    thread.start()

    return TaskStatus(
        task_id=task_id,
        status="running",
        message="算法已启动"
    )

@app.post("/sent_parameter")
async def send_parameter(modelpath: str, request: optimize_parameters):
    with open(f"{modelpath}/parametes.txt", "w", encoding="utf-8") as f:
        f.write(request.parameters)

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=9100,  reload=True)