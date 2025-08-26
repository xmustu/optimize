import os
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.ga import GA
from sldwks import GeometrySimulation, OptimizationProblem

from ha import HA
import time

# 假设已经定义了problem
class HACallback(Callback):
    """自定义回调函数记录优化历史"""

    def __init__(self):
        super().__init__()
        self.history = []

    def notify(self, algorithm):

        # 获取当前代的信息
        gen = algorithm.n_gen
        fes = algorithm.problem.fes

        # 获取当前种群
        pop = algorithm.pop

        #打印信息
        def constraint_sort_key(fitness, cv):
            # 返回：(是否违反约束, 违反程度, 适应度)
            return (cv > 0, cv, fitness)
            # 组合所有个体信息

        fitnesses = pop.get("F").flatten()
        cvs = algorithm.pop_cv
        xs = pop.get("X")

        individuals = [(fitnesses[i], cvs[i], xs[i]) for i in range(len(pop))]

        # 根据你定义的排序方式排序
        sorted_individuals = sorted(individuals, key=lambda ind: constraint_sort_key(ind[0], ind[1]))

        print(f"\nGeneration {algorithm.n_gen} Top 4:")
        for i, (fit, cv, x) in enumerate(sorted_individuals[:4]):
            print(f"  Rank {i + 1}: fitness={fit.item():.6f}, cv={cv.item():.6f}, x={x}")

        # 计算当前代的最佳可行解
        feasible_indices = np.where(algorithm.pop_cv <= 0)[0]

        if len(feasible_indices) > 0:
            # 从可行解中找到 f 最小的下标
            best_index = feasible_indices[np.argmin(pop.get("F")[feasible_indices])]
        else:
            # 没有可行解，从所有个体中找到 cv 最小的下标
            best_index = np.argmin(algorithm.pop_cv)

        x = pop.get("X")[best_index]
        x = np.atleast_2d(x)
        out = {}
        algorithm.problem._evaluate(x, out)
        fitness = out["F"]
        if hasattr(algorithm.problem, "evaluate") and algorithm.problem.has_constraints():
            result = out["G"]
            G = np.atleast_1d(result)
            cv = float(np.sum(np.maximum(0, G)))
            print("有约束")
        else:
            cv = 0.0
            print("没有约束")
            # 组合所有个体信息


        # 记录当前代的信息
        self.history.append({
            'Generation': gen,
            'FEs': fes,
            'F': fitness,
            'CV': cv,
            "Feasibility_ratio": np.sum(cvs <= 0) / pop.size
        })
class GACallback(Callback):
    """自定义回调函数记录优化历史"""

    def __init__(self):
        super().__init__()
        self.history = []

    def notify(self, algorithm):
        # 获取当前代的信息
        gen = algorithm.n_gen
        fes = algorithm.problem.fes

        # 获取当前种群
        pop = algorithm.pop

        def constraint_sort_key(fitness, cv):
            # 返回：(是否违反约束, 违反程度, 适应度)
            return (cv > 0, cv, fitness)
            # 组合所有个体信息

        fitnesses = pop.get("F").flatten()
        cvs = pop.get("CV").flatten()
        xs = pop.get("X")

        # 组合所有个体信息
        individuals = [(fitnesses[i], cvs[i], xs[i]) for i in range(len(pop))]

        # 根据你定义的排序方式排序
        sorted_individuals = sorted(individuals, key=lambda ind: constraint_sort_key(ind[0], ind[1]))

        print(f"\nGeneration {algorithm.n_gen} Top 4:")
        for i, (fit, cv, x) in enumerate(sorted_individuals[:4]):
            print(f"  Rank {i + 1}: fitness={fit:.6f}, cv={cv:.6f}, x={x}")



        # 计算当前代的最佳可行解
        feasibles = [ind for ind in pop if np.all(ind.CV <= 0)]

        if feasibles:
            # 有可行解：选择目标函数最小的
            best = min(feasibles, key=lambda ind: ind.F[0])
        else:
            # 没有可行解：选择违反约束最小的
            best = min(pop, key=lambda ind: ind.CV)



        best_index = None
        for i, ind in enumerate(pop):
            if ind == best:
                best_index = i
                break

        x = pop.get("X")[best_index]
        x = np.atleast_2d(x)
        out = {}
        algorithm.problem._evaluate(x, out)
        fitness = out["F"]
        if hasattr(algorithm.problem, "evaluate") and algorithm.problem.has_constraints():
            result = out["G"]
            G = np.atleast_1d(result)
            cv = float(np.sum(np.maximum(0, G)))
            print("有约束")
        else:
            cv = 0.0
            print("没有约束")

        # 记录当前代的信息
        self.history.append({
            'Generation': gen,
            'FEs': fes,
            'F': fitness,
            'CV': cv,
            "Feasibility_ratio":np.sum(cvs <= 0)/pop.size
        })


# 实验参数
methods = [ "HA","GA"]
# methods = ["L-BFGS-B","NoMethod","GA"]
n_trials = 15
max_gen = 20
pop_size = 30


# 存储所有历史数据
all_history = {method: [] for method in methods}

# 进行30次实验
for trial in range(n_trials):
    print(f"\n=== 实验 {trial + 1}/{n_trials} ===")

    # 固定种子确保可重复性
    seed = 42 + trial
    np.random.seed(seed)


    for method in methods:
        print(f"运行方法: {method}")
        if method == "NoMethod":
            ha = HA(method=method, pop_size=pop_size, seed=seed,activate_method= False)
            callback = HACallback()
        elif method == "GA":
            ha = GA(pop_size=pop_size)
            callback = GACallback()
        else :# 初始化HA算法
            ha = HA(method="L-BFGS-B", pop_size=pop_size, seed=seed)
            callback = HACallback()

        # # 初始化回调
        EXE_PATH = r"C:\Users\dell\Projects\CAutoD\wenjian\net8.0\sldxunhuan.exe"
        MODEL_PATH = r"C:\Users\dell\Projects\CAutoD\wenjian\AutoFrame.SLDPRT"
        MAX_STRESS = 1667155999
        try:
            simulation = GeometrySimulation(EXE_PATH, MODEL_PATH)
        except Exception as e:
            print(f"初始化仿真失败：{e}")
            log_file = os.path.join(os.path.dirname(MODEL_PATH), "logdebug.txt")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"=== 初始化仿真失败：{str(e)}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")


        #problem = OptimizationProblem(simulation, MAX_STRESS)
        problem = OptimizationProblem(simulation, simulation.max_stress)
     
        # 运行优化
        res = minimize(
            problem,
            ha,
            ('n_gen', max_gen),
            seed=seed,
            verbose=False,
            callback=callback
        )

        # 保存历史记录
        all_history[method].append(callback.history)
        simulation._cleanup()

        print(f"方法 {method} 完成")

# 计算每个方法在相同代数下的平均值
avg_results = []

# 最大代数
max_generations = max_gen

for method in methods:
    # 为每个代数创建存储列表
    gen_data = {gen: {'F': [], 'CV': [], 'FEs': [],'Feasibility_ratio':[]} for gen in range(1, max_generations + 1)}

    # 收集每个代数的数据
    for trial_history in all_history[method]:
        for record in trial_history:
            gen = record['Generation']
            if gen <= max_generations:
                gen_data[gen]['F'].append(record['F'])
                gen_data[gen]['CV'].append(record['CV'])
                gen_data[gen]['FEs'].append(record['FEs'])
                gen_data[gen]['Feasibility_ratio'].append(record['Feasibility_ratio'])

    # 计算每个代数的平均值
    for gen in range(1, max_generations + 1):
        if gen_data[gen]['F']:  # 确保有数据
            avg_F = np.mean(gen_data[gen]['F'])
            avg_CV = np.mean(gen_data[gen]['CV'])
            avg_FEs = np.mean(gen_data[gen]['FEs'])
            avg_Feasibility_ratio = np.mean(gen_data[gen]['Feasibility_ratio'])

            avg_results.append({
                'Method': method,
                'Generation': gen,
                'Avg_F': avg_F,
                'Avg_CV': avg_CV,
                'Avg_Feasibility_ratio': avg_Feasibility_ratio,
                'Avg_FEs': avg_FEs
            })

# 保存平均结果
results_df = pd.DataFrame(avg_results)
os.makedirs("csv", exist_ok=True)
results_df.to_csv('csv/SLD_average_per_generation.csv', index=False)

# # 按方法保存单独文件
# for method in methods:
#     method_df = results_df[results_df['Method'] == method]
#     method_df.to_csv(f'{method}_average_per_generation.csv', index=False)
#
# print("\n平均结果已保存:")
# print(results_df.head())