import json
from math import pow, cos, pi

from mealpy import PSO, FloatVar
import numpy as np

# Hyperparameters
# Variables Interval: [-100, 100]
# Repetitions: 30
# 10 variables
# C1 = 2.0
# C2 = 2.0
# W = 0.9 -l_dec-> 0.4

# Config 1:
# Population Size: 30
# Iterations: 500

# Config 2:
# Population Size: 50
# Iterations: 1000

# Config 3:
# Population Size: 100
# Iterations: 2000


def rotated_high_conditional_elliptic_function(d: int, x_array: list) -> float:
    result = 0
    for i in range(1, d):
        result += pow(10, (i-1) / (d-1)) * pow(x_array[i], 2)
    return result


def wierstrass_function(d: int, x_array: list) -> float:
    result = 0
    a = 0.5
    b = 3
    kmax = 20
    for i in range(1, d):
        op_1 = 0
        for k in range(0, kmax+1):
            op_1 += pow(a, k) * cos(2*pi*pow(b, k)*(x_array[i] + 0.5))

        op_2 = 0
        for k in range(0, kmax+1):
            op_2 += pow(a, k) * cos(2*pi*pow(b, k)*0.5)

        result = op_1 + d * op_2

    return result


def run_pso(
    objective_function: str = "rotated_high_conditional_elliptic_function",
    c1: float = 2.0,
    c2: float = 2.0,
    w: float = 0.9,
    l_upper: float = 100,
    l_lower: float = -100,
    d: int = 10,
    config: dict = {},
):
    if config == {}:
        raise ValueError("Config is empty")

    if objective_function == "rotated_high_conditional_elliptic_function":
        obj_func = rotated_high_conditional_elliptic_function
    elif objective_function == "wierstrass_function":
        obj_func = wierstrass_function
    else:
        raise ValueError("Invalid objective function")
    problem_dict = {
        "bounds": FloatVar(lb=(l_lower,) * d, ub=(l_upper,) * d, name="delta"),
        "obj_func": lambda x: obj_func(d, x),
        "minmax": "min",
        "log_to": None
    }
    model = PSO.OriginalPSO(
        epoch=config["max_iterations"],
        pop_size=config["population_size"],
        c1=c1,
        c2=c2,
        w=w,
    )
    g_best = model.solve(problem_dict)

    return g_best.target.fitness


def init_results():
    return {
        "median_fitness": 0.,
        "std_fitness": 0.,
        "mean_fitness": 0.,
        "best_fitness": 0.,
    }


def main(
    objective_function: str = "rotated_high_conditional_elliptic_function",
):
    print(
        f"Running PSO for {objective_function}"
    )
    c1 = 2.0
    c2 = 2.0
    w = 0.9
    l_upper = 100
    l_lower = -100
    d = 10
    repetitions = 30
    configs = [
        {
            "population_size": 30,
            "max_iterations": 500,
        },
        {
            "population_size": 50,
            "max_iterations": 1000,
        },
        {
            "population_size": 100,
            "max_iterations": 2000,
        }
    ]

    for i, config in enumerate(configs):
        print(f"Running config {i+1}: {config}")
        results = init_results()
        fitnesses = []
        for j in range(repetitions):
            print(f"{j+1}/{repetitions} repetitions")
            best_fitness = results["best_fitness"]
            result = run_pso(
                objective_function=objective_function,
                c1=c1,
                c2=c2,
                w=w,
                l_upper=l_upper,
                l_lower=l_lower,
                d=d,
                config=config
            )
            if result < best_fitness or best_fitness == 0:
                best_fitness = result
            fitnesses.append(result)

        results["mean_fitness"] = np.mean(fitnesses)
        results["std_fitness"] = np.std(fitnesses)
        results["median_fitness"] = np.median(fitnesses)
        results["best_fitness"] = best_fitness
        configs[i]["results"] = results

    return configs


def run():
    objective_function = "rotated_high_conditional_elliptic_function"
    results = main(objective_function=objective_function)
    for i, config in enumerate(results):
        print(f"Config {i+1}: {config}")
    with open("rhce_results.json", "w") as f:
        json.dump(results, f, indent=4)

    objective_function = "wierstrass_function"
    results = main(objective_function=objective_function)
    for i, config in enumerate(results):
        print(f"Config {i+1}: {config}")
    with open("wierstrass_results.json", "w") as f:
        json.dump(results, f, indent=4)
