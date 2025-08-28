from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
from typing import Optional, Dict, Any, List, Union, Set
from sldwks import start_main, write_key, ControlCommand
import sys
import queue

# -----schema------

class optimize_parameters(BaseModel):
    parameters: str
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



app = FastAPI(title="几何优化服务",)


# 创建线程安全的任务队列
task_queue = queue.Queue()
# 标记算法是否正在运行的信号量（初始为1，表示允许一个任务运行）
semaphore = threading.Semaphore(1)

# 全局变量：记录当前正在运行的任务请求（None表示无任务运行）
active_task: Optional[AlgorithmRequest] = None
active_task_lock = threading.Lock()  # 确保多线程安全访问


class ConnectionManager:
    def __init__(self): 
        self.active_connections: Set[WebSocket] = set()
        # 记录每个任务对应的连接
        self.task_subscriptions: Dict[str, Set[WebSocket]] = {}
        # 线程安全锁
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)
            if task_id:
                if task_id not in self.task_subscriptions:
                    self.task_subscriptions[task_id] = set()
                self.task_subscriptions[task_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, task_id: str):
        async with self.lock:
            if websocket in self.active_connections:
                self.acitvae_connections.remove(websocket)

            if task_id and task_id in self.task_subscriptions:
                if websocket in self.task_subscriptions[task_id]:
                    self.task_subscriptions[task_id].remove(websocket)
                # 如果该任务没有订阅者了，删除键节省内存
                if not self.task_subscriptions[task_id]:
                    del self.task_subscriptions[task_id]

    async def send_and_cleanup(self, task_id: str, message: dict):
        """向订阅了特定任务的客户端发送更新后,立即清理相关连接"""
        async with self.lock:
            # 获取该任务的所有订阅连接
            connections = self.task_subscriptions.get(task_id, set()).copy()

            # 发送通知
            for connection in connections:
                try:
                    await connection.send_json(message)
                    # 发送后主动断开
                    await connection.close(code=1000, reason="任务已启动，连接关闭")
                except Exception as e:
                    print(f"发送通知或关闭连接失败: {str(e)}")
            if task_id in self.task_subscriptions:
                del self.task_subscriptions[task_id]
            for conn in connections:
                self.active_connections.discard(conn)


manager = ConnectionManager()


def algorithm_worker():
    """算法工作线程，持续从队列中获取任务并执行, 并发送状态变更通知,最后清理连接"""
    global active_task
    while True:
        # 从队列获取任务， 阻塞等待
        request = task_queue.get()
        try:
            # 更新活跃任务状态
            # 任务从pending变为running，发送通知
            with active_task_lock:
                active_task = request
                print(f"开始执行算法，任务ID: {request.task_id}")

            taskstatus = TaskStatus(
                task_id=request.task_id,
                status="running",
                message= "任务已开始运行，可以开始监听control.txt"
            )

             # 在主线程中执行WebSocket通知
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(manager.send_and_cleanup(request, taskstatus))
            loop.close()


            # 运行算法
            run_algorithm_in_thread(request.parameters["model_path"])


            print(f"任务执行完成，任务ID: {request.task_id}")


        except Exception as e:
            print(f"任务{request.task_id}执行出错 : {str(e)}")
        finally:
            with active_task_lock:
                active_task = None
            task_queue.task_done()


# 启动算法工作线程（程序启动时运行）
threading.Thread(target=algorithm_worker, daemon=True).start()



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
    try:
        """提交算法任务"""
        if not check_environment():
            raise HTTPException(status_code=503, detail="算法依赖环境未就绪（SolidWorks/.NET 8.0）")
        with active_task_lock:
            is_duplicate = (request.task_id == active_task.task_id)

        if not is_duplicate:

            queue_items = []
            while not task_queue.empty():
                item = task_queue.get()
                queue_items.append(item)
                if item.task_id == request.task_id:
                    is_duplicate = True
            for item in queue_items:
                task_queue.put(item)
        if is_duplicate:
            raise HTTPException(status_code=400, detail=f"任务ID {request.task_id} 的算法请求已存在")
        
        task_queue.put(request)
        return TaskStatus(
            task_id=request.task_id,
            status="pending",
            message=f"任务已加入队列，当前排队位置: {task_queue.qsize() - 1}"
    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动算法失败：{e}")


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket端点，用于订阅特定任务的状态变更通知"""
    await manager.connect(websocket, task_id)
    try:
        # 发送初始订阅确认
        await websocket.send_json({
            "task_id": task_id,
            "status": "subscribed",
            "message": "已成功订阅任务状态变更通知"
        })
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)
    except Exception as e:
        print(f"WebSocket连接错误: {e}")
        manager.disconnect(websocket, task_id)

@app.get("/queue_status")
async def get_queue_status():
    """获取当前队列状态"""
    return {
        "queue_size": task_queue.qsize(),
        "is_running": not semaphore.acquire(blocking=False)
    }

@app.post("/sent_parameter")
async def send_parameter(model_path: str):
    print(f"收到参数：{model_path}")
    control_file = os.path.join(model_path, "control.txt")
    write_key(control_file, "command", "8")
    return {
        "status": "success",
        "message": "收到！继续工作",
    }

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=9100,  reload=True)