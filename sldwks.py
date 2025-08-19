import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import time
import os
import subprocess
import sys
from matplotlib.font_manager import FontProperties
from ha import HA
import builtins
from contextlib import redirect_stdout
import sys
# 保存原始的print函数
original_print = builtins.print

# 定义新的print函数，默认设置flush=True
def custom_print(*args, **kwargs):
    # 如果用户没有指定flush参数，则默认设置为True
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    # 调用原始的print函数
    original_print(*args, **kwargs)

# 替换内置的print函数
builtins.print = custom_print

# 设置环境变量，确保输出不被缓冲
os.environ["PYTHONUNBUFFERED"] = "1"

# 仅使用Windows 10/11系统默认自带的中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# 手动指定黑体字体文件（Windows默认路径）
font = FontProperties(
    fname=r"C:\Windows\Fonts\simhei.ttf",  # 黑体的绝对路径
    size=10
)


# 命令常量
class ControlCommand:
    INIT = 0
    PARAMS_READY = 1
    RESULT_READY = 2
    EXIT = 3
    CSERROR = -1
    PYERROR = -2


# 工具函数：读写文件
def read_key(file_path, key):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1]
    except Exception as e:
        print(f"读取键值失败（{key}）：{e}")
    return None


def write_key(file_path, key, value):
    lines = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]

    found = False
    for i in range(len(lines)):
        if lines[i].startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def wait_for_command(target_command, control_file_path, timeout=120):
    start_time = time.time()
    elapsed = 0
    while elapsed < timeout:
        cmd = read_key(control_file_path, "command")
        if cmd in [str(ControlCommand.CSERROR), str(ControlCommand.PYERROR)]:
            error_code = read_key(control_file_path, "error_code")
            raise RuntimeError(f"sub初始化错误（代码：{error_code}）")

        if cmd == str(target_command):
            print(f"已收到目标命令：{target_command}（耗时{elapsed:.1f}秒）")
            return True

        if int(elapsed) % 5 == 0:
            print(f"等待命令[{target_command}]中...已等待{elapsed:.1f}秒（当前命令：{cmd or '无'}）")
        time.sleep(1)
        elapsed = time.time() - start_time

    raise TimeoutError(f"等待命令[{target_command}]超时（{timeout}秒）")


def wait_for_valid_param_count(control_file_path, min_count=1, timeout=120):
    start_time = time.time()
    elapsed = 0
    while elapsed < timeout:
        param_count_str = read_key(control_file_path, "param_count")
        if param_count_str and param_count_str.isdigit():
            param_count = int(param_count_str)
            if param_count >= min_count:
                print(f"获取到有效参数数量：{param_count}（耗时{elapsed:.1f}秒）")
                return param_count

        if int(elapsed) % 3 == 0:
            print(f"等待有效参数数量中...已等待{elapsed:.1f}秒（当前值：{param_count_str or '未设置'}）")
        time.sleep(1)
        elapsed = time.time() - start_time

    raise TimeoutError(f"等待有效参数数量超时（{timeout}秒），最后获取值：{param_count_str}")


class GeometrySimulation:
    def __init__(self, exe_path, model_path):
        self.exe_path = exe_path
        self.model_path = model_path
        self.param_count = 0
        self.param_names = []
        self.initial_values = []
        self.param_bounds = None
        self.process = None
        self.is_running = False
        self.simulation_count = 0  # 总仿真计数（包含重试）
        self.original_simulation_count = 0  # 原始参数组计数（不包含重试）

        self.model_dir = os.path.dirname(model_path)
        self.control_file = os.path.join(self.model_dir, "control.txt")
        self.data_file = os.path.join(self.model_dir, "data.txt")
        self.log_file = os.path.join(self.model_dir, "logdebug.txt")

        self._init_files()
        self._initialize_communication()

    def _init_files(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with open(self.control_file, "w", encoding="utf-8") as f:
            f.write("command=\n")
            f.write("param_count=0\n")
            f.write("error_code=0\n")

        with open(self.data_file, "w", encoding="utf-8") as f:
            f.write("volume=0\n")
            f.write("stress=0\n")

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"=== 日志开始于 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        print(f"{self.control_file}")
        print(f"{self.data_file}")
        print(f"{self.log_file}")

    def _initialize_communication(self):
        try:
            print(f"启动sub：{self.exe_path}")
            print(f"模型路径：{self.model_path}")

            self.log_handle = open(self.log_file, "a", encoding="utf-8")
            self.process = subprocess.Popen(
                [self.exe_path, self.model_path],
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            import threading
            threading.Thread(target=self._monitor_process, daemon=True).start()

            print("等待sub启动SolidWorks并完成初始化...")
            wait_for_command(ControlCommand.INIT, self.control_file, timeout=180)
            self.param_count = wait_for_valid_param_count(self.control_file, timeout=180)
            self._read_param_details()

            print(f"初始化完成，共获取到{self.param_count}个参数")
            self._calculate_param_bounds()
            self.is_running = True

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"初始化失败：{str(e)}")

    def _read_param_details(self):
        self.param_names = []
        self.initial_values = []
        max_retry = 3
        for i in range(self.param_count):
            for retry in range(max_retry):
                name = read_key(self.control_file, f"param_name_{i}")
                value = read_key(self.control_file, f"initial_value_{i}")
                if name and value:
                    self.param_names.append(name)
                    self.initial_values.append(float(value))
                    print(f"获取参数{i}：{name} = {value}")
                    break
                else:
                    print(f"重试获取参数{i}（第{retry + 1}/{max_retry}次）...")
                    time.sleep(2)
            else:
                raise ValueError(f"参数{i}信息不完整（名称：{name or '缺失'}，值：{value or '缺失'}）")

    def _monitor_process(self):
        while True:
            if self.process.poll() is not None:
                exit_code = self.process.returncode
                self.is_running = False
                print(f"\n[警告] sub进程已退出，退出码: {exit_code}")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n=== sub进程已退出，退出码: {exit_code}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                error_code = read_key(self.control_file, "error_code")
                if error_code:
                    print(f"[错误码] {error_code}")
                break
            time.sleep(1)

    def run_simulation(self, params):
        if not self.is_running or self.process.poll() is not None:
            raise RuntimeError("sub进程未运行，无法执行仿真")

        # 原始参数组计数（不随重试增加）
        self.original_simulation_count += 1
        current_original_count = self.original_simulation_count

        params = np.array(params, dtype=np.float64)
        if len(params) != self.param_count:
            raise ValueError(f"参数数量不匹配（需{self.param_count}个，实际{len(params)}个）")

        # 重试逻辑：如果应力为0则重新发送
        max_retries = 3  # 最大重试次数
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # 每次重试都更新总仿真计数
                self.simulation_count += 1
                current_total_count = self.simulation_count

                # 写入参数到数据文件
                for i in range(self.param_count):
                    write_key(self.data_file, f"param_{i}", f"{params[i]:.8f}")

                # 发送参数就绪信号
                write_key(self.control_file, "command", str(ControlCommand.PARAMS_READY))
                print(f"发送参数（原始组#{current_original_count}，第{retry_count + 1}次尝试）：{params}，等待sub处理...")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n=== 收到参数（原始组#{current_original_count}，总计数#{current_total_count}）：{params}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

                start_time = time.time()
                while True:
                    if time.time() - start_time > 300:  # 单次仿真超时时间
                        raise TimeoutError(f"仿真超时（原始组#{current_original_count}）")
                    if self.process.poll() is not None:
                        raise RuntimeError("sub进程已崩溃")

                    current_cmd = read_key(self.control_file, "command")
                    if current_cmd == str(ControlCommand.RESULT_READY):
                        # 读取结果
                        volume_str = read_key(self.data_file, "volume")
                        stress_str = read_key(self.data_file, "stress")

                        if not volume_str or not stress_str:
                            raise ValueError("未获取到完整的仿真结果")

                        volume = float(volume_str)
                        stress = float(stress_str)
                        write_key(self.control_file, "command", "")
                        break
                    if current_cmd == str(ControlCommand.CSERROR):
                        error_code = read_key(self.control_file, "error_code")
                        raise RuntimeError(f"sub仿真错误（代码: {error_code}）")
                    if current_cmd == str(ControlCommand.PYERROR):
                        error_code = read_key(self.control_file, "error_code")
                        raise RuntimeError(f"Python交互错误（代码: {error_code}）")

                    time.sleep(0.5)

                # 检查应力是否为0，若为0则重试
                if stress == 0:
                    print(f"警告：原始组#{current_original_count}第{retry_count + 1}次尝试返回应力为0，将重试...")
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"=== 警告：原始组#{current_original_count}第{retry_count + 1}次尝试返回应力为0，将重试 ===\n")
                    retry_count += 1
                    continue  # 继续重试

                # 应力不为0，正常返回结果
                print(
                    f"仿真完成（原始组#{current_original_count}，总计数#{current_total_count}）：体积={volume:.6f}, 应力={stress:.6f}")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"=== 仿真完成（原始组#{current_original_count}，总计数#{current_total_count}）：体积={volume:.6f}, 应力={stress:.6f}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                return volume, stress

            except Exception as e:
                print(f"仿真执行失败（原始组#{current_original_count}）: {e}")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"=== 仿真失败（原始组#{current_original_count}，总计数#{current_total_count}）：{str(e)}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                try:
                    write_key(self.control_file, "command", str(ControlCommand.PYERROR))
                    write_key(self.control_file, "error_code", "1001")
                except:
                    pass
                raise

        # 超过最大重试次数仍返回应力0，抛出异常
        raise RuntimeError(f"原始组#{current_original_count}经{max_retries + 1}次尝试后仍返回应力0，无法继续")

    def _calculate_param_bounds(self):
        self.param_bounds = np.zeros((self.param_count, 2))
        for i in range(self.param_count):
            initial = self.initial_values[i]
            self.param_bounds[i, 0] = initial * 0.8
            self.param_bounds[i, 1] = initial * 1.2

        print("计算参数范围：")
        for name, (min_val, max_val) in zip(self.param_names, self.param_bounds):
            print(f"  {name}：[{min_val:.6f}, {max_val:.6f}]")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n=== 参数范围 ===\n")
            for name, (min_val, max_val) in zip(self.param_names, self.param_bounds):
                f.write(f"  {name}：[{min_val:.6f}, {max_val:.6f}]\n")

    def _cleanup(self):
        self.is_running = False
        try:
            write_key(self.control_file, "command", str(ControlCommand.EXIT))
            time.sleep(1)
        except:
            pass

        if hasattr(self, 'log_handle') and not self.log_handle.closed:
            self.log_handle.write(f"\n=== 程序结束于 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            self.log_handle.close()

        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                if self.process.wait(timeout=5) is None:
                    self.process.kill()
            except:
                pass
            print("sub进程已终止")

    def __del__(self):
        self._cleanup()


class OptimizationProblem(Problem):
    def __init__(self, simulation, max_stress):
        self.fes = 0
        xl = simulation.param_bounds[:, 0]
        xu = simulation.param_bounds[:, 1]
        super().__init__(
            n_var=simulation.param_count,
            n_obj=1,
            n_constr=1,
            xl=xl,
            xu=xu
        )
        self.simulation = simulation
        self.max_stress = max_stress
        self.simulation_results = []  # 存储每次仿真的（参数, 体积, 应力）

    def _evaluate(self, x, out, *args, **kwargs):
        self.fes = x.shape[0]
        volumes = []
        stresses = []
        for params in x:
            try:
                volume, stress = self.simulation.run_simulation(params)
                volumes.append(volume)
                stresses.append(stress)
                self.simulation_results.append((params, volume, stress))  # 保存结果
            except Exception as e:
                print(f"仿真评估错误：{e}")
                volumes.append(1e18)
                stresses.append(1e18)
        out["F"] = np.array(volumes).reshape(-1, 1)
        out["G"] = np.array(stresses).reshape(-1, 1) - self.max_stress


def run_optimization(exe_path, model_path, max_stress, generations=3, population_size=5):
    try:
        simulation = GeometrySimulation(exe_path, model_path)
    except Exception as e:
        print(f"初始化仿真失败：{e}")
        log_file = os.path.join(os.path.dirname(model_path), "logdebug.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"=== 初始化仿真失败：{str(e)}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        return None, None

    problem = OptimizationProblem(simulation, max_stress)
    algorithm = GA(
        pop_size=population_size,
        eliminate_duplicates=True,
        verbose=False
    )
    # algorithm = HA(
    #     pop_size=population_size,
    #     method="Adam"
    # )

    history = []

    def callback(algorithm):
        gen = algorithm.n_gen
        for ind in algorithm.pop:
            history.append([gen] + ind.X.tolist() + [ind.F[0]])

    print(f"\n=== 开始优化 ===")
    with open(simulation.log_file, "a", encoding="utf-8") as f:
        f.write(
            f"\n=== 开始优化：代数={generations}, 种群大小={population_size}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    start_time = time.time()
    try:
        res = minimize(
            problem,
            algorithm,
            ('n_gen', generations),
            seed=42,
            verbose=True,
            callback=callback
        )
    except Exception as e:
        print(f"优化过程出错：{e}")
        with open(simulation.log_file, "a", encoding="utf-8") as f:
            f.write(f"=== 优化过程出错：{str(e)}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        return None, simulation

    end_time = time.time()
    print(f"优化完成，耗时：{end_time - start_time:.2f}秒")
    with open(simulation.log_file, "a", encoding="utf-8") as f:
        f.write(f"=== 优化完成，耗时：{end_time - start_time:.2f}秒，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # 保存优化历史
    columns = ["Generation"] + simulation.param_names + ["Volume"]
    df = pd.DataFrame(history, columns=columns)
    history_path = os.path.join(os.path.dirname(model_path), "optimization_history.csv")
    df.to_csv(history_path, index=False)

    print("\n=== 优化结果详细信息 ===")
    if problem.simulation_results:
        # 1. 筛选可行解（应力 ≤ 最大允许应力且应力≠0）
        feasible_solutions = [
            (params, vol, stress)
            for params, vol, stress in problem.simulation_results
            if stress <= max_stress and stress != 0
        ]

        # 2. 选择最优解（优先从可行解中选体积最小的）
        if feasible_solutions:
            # 可行解中按体积升序排序
            feasible_solutions.sort(key=lambda x: x[1])
            best_params, best_volume, best_stress = feasible_solutions[0]
            is_feasible = True
        else:
            # 无可行解时，从所有非0应力解中选体积最小的
            valid_solutions = [s for s in problem.simulation_results if s[2] != 0]
            if valid_solutions:
                valid_solutions.sort(key=lambda x: x[1])
                best_params, best_volume, best_stress = valid_solutions[0]
                is_feasible = False
            else:
                print("⚠️ 所有仿真结果均为应力0，无法筛选有效解")
                return None, simulation

        # 3. 打印结果
        print("1. 最优参数：")
        for name, val in zip(simulation.param_names, best_params):
            print(f"   {name}：{val:.6f}")

        print(f"\n2. 最优体积：{best_volume:.6f}")
        print(f"3. 最优应力：{best_stress:.6f}")

        # 4. 约束判断
        print(f"\n4. 约束条件：最大允许应力 = {max_stress:.6f}")
        if is_feasible:
            print(f"   符合约束要求")  
        else:
            print(f"   无可行解，将呈现体积最小的不可行解") 

        # 5. 写入日志
        with open(simulation.log_file, "a", encoding="utf-8") as f:
            f.write("\n=== 优化结果详细信息 ===\n")
            f.write("1. 最优参数：\n")
            for name, val in zip(simulation.param_names, best_params):
                f.write(f"   {name}：{val:.6f}\n")
            f.write(f"\n2. 最优体积：{best_volume:.6f}\n")
            f.write(f"3. 最优应力：{best_stress:.6f}\n")
            f.write(f"\n4. 约束条件：最大允许应力 = {max_stress:.6f}\n")
            f.write(f"   {'符合约束要求' if is_feasible else '无可行解，选中体积最小的不可行解'}\n")
    else:
        print("未获取到有效的仿真结果")

    # 发送退出命令
    print("\n所有仿真完成，发送退出命令...")
    write_key(simulation.control_file, "command", str(ControlCommand.EXIT))

    return res, simulation


def visualize_results(simulation, model_path):
    history_path = os.path.join(os.path.dirname(model_path), "optimization_history.csv")
    if not os.path.exists(history_path):
        print("无优化历史数据，跳过可视化")
        return

    df = pd.read_csv(history_path)
    last_gen = df["Generation"].max()
    last_gen_data = df[df["Generation"] == last_gen]

    # 收敛曲线
    plt.figure(figsize=(10, 6))
    gen_min = df.groupby("Generation")["Volume"].min()
    plt.plot(gen_min.index, gen_min.values, "b-o", linewidth=2)
    plt.xlabel("迭代代数")
    plt.ylabel("最小体积")
    plt.title("优化收敛曲线")
    plt.grid(True)
    curve_path = os.path.join(os.path.dirname(model_path), "convergence_curve.png")
    plt.savefig(curve_path, dpi=300)
    #plt.show()

    # 参数分布
    if simulation and simulation.param_count >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            last_gen_data[simulation.param_names[0]],
            last_gen_data[simulation.param_names[1]],
            c=last_gen_data["Volume"],
            cmap="viridis",
            s=100,
            alpha=0.7
        )
        plt.colorbar(scatter, label="体积")
        plt.xlabel(simulation.param_names[0])
        plt.ylabel(simulation.param_names[1])
        plt.title(f"第{last_gen}代种群参数分布")
        plt.grid(True)
        dist_path = os.path.join(os.path.dirname(model_path), "parameter_distribution.png")
        plt.savefig(dist_path, dpi=300)
        #plt.show()


def start_main(model_path: str = None):
    EXE_PATH = r".\net8.0\sldxunhuan.exe"
    #"C:\Users\asus\source\repos\sldxunhuan\bin\Debug\net8.0\sldxunhuan.exe"
    # model_path = os.environ.get("MODEL_PATH")

    # # 调试：打印Python解释器路径（确认是否是目标环境）
    # print(f"子进程Python路径:{sys.executable}")

    # # 调试：打印所有可用的环境变量键名
    # print("子进程收到的环境变量键名列表:")
    # for key in sorted(os.environ.keys()):
    #     print(f"- {key}")

    # # 尝试接收MODEL_PATH
    # model_path = os.environ.get("MODEL_PATH")
    # if model_path:
    #     print(f"成功收到MODEL_PATH: {model_path}")
    # else:
    #     print("未收到MODEL_PATH环境变量")

    # # 检查是否收到PYTHONUNBUFFERED（用于验证环境变量传递是否完全失效）
    # print(f"PYTHONUNBUFFERED值: {os.environ.get('PYTHONUNBUFFERED')}")

    MODEL_PATH = model_path
    print("MODEL_PATH:", MODEL_PATH)
    #"C:\Users\asus\Desktop\wenjian\AutoFrame.SLDPRT"
    MAX_STRESS = 1667155999
    GENERATIONS = 3
    POPULATION_SIZE = 4

    try:
        if not os.path.exists(EXE_PATH):
            raise FileNotFoundError(f"sub不存在: {EXE_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")

        res, sim = run_optimization(
            exe_path=EXE_PATH,
            model_path=MODEL_PATH,
            max_stress=MAX_STRESS,
            generations=GENERATIONS,
            population_size=POPULATION_SIZE
        )
        visualize_results(sim, MODEL_PATH)
    except Exception as e:
        print(f"程序出错：{e}")
        log_file = os.path.join(os.path.dirname(MODEL_PATH), "logdebug.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"=== 程序出错：{str(e)}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        input("按回车键退出...")

if __name__ == '__main__':
    start_main(r".\AutoFrame.SLDPRT")