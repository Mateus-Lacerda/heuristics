from argparse import ArgumentParser

import numpy as np
from mealpy import ACOR, FloatVar

from cost_functions import (
    rotated_high_conditional_elliptic_function,
    wierstrass_function
)
from plots import run_with_plot

# Hyperparameters
# Variables Interval: [-100, 100]
# Repetitions: 30
# 10 variables
# p = 0.5
# a = 1
# b = 2

# Config 1:
# Population Size: 30
# Iterations: 500

# Config 2:
# Population Size: 50
# Iterations: 1000

# Config 3:
# Population Size: 100
# Iterations: 2000


def run_aco(
    objective_function: str = "rotated_high_conditional_elliptic_function",
    p: float = 0.5,
    a: float = 1.0,
    b: float = 2.0,
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
    model = ACOR.OriginalACOR(
        epoch=config["max_iterations"],
        pop_size=config["population_size"],
        intent_factor=p,
        zeta=a,
        sample_count=b

    )
    g_best = model.solve(problem_dict)

    return g_best.target.fitness


def init_results():
    return {
        "median_fitness": 0.,
        "std_fitness": 0.,
        "mean_fitness": 0.,
        "best_fitness": 0.,
        "history": []
    }


def main(
    objective_function: str = "rotated_high_conditional_elliptic_function",
):
    print(
        f"Running ACO for {objective_function}"
    )
    p = 0.5
    a = 1.0
    b = 2.0
    l_upper = 100
    l_lower = -100
    d = 10
    repetitions = 30
    configs = [
        {
            "population_size": 20,
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
            result = run_aco(
                objective_function=objective_function,
                p=p,
                a=a,
                b=b,
                l_upper=l_upper,
                l_lower=l_lower,
                d=d,
                config=config
            )
            if result < best_fitness or best_fitness == 0:
                best_fitness = result
            fitnesses.append(result)
            results["history"].append(best_fitness)


        results["mean_fitness"] = np.mean(fitnesses)
        results["std_fitness"] = np.std(fitnesses)
        results["median_fitness"] = np.median(fitnesses)
        results["best_fitness"] = best_fitness
        configs[i]["results"] = results

    return configs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--weistrass", "-w", action="store_true", help="Run Weistrass function")
    parser.add_argument("--rotated", "-r", action="store_true", help="Run Rotated High Conditional Elliptic function")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()

    if args.weistrass:
        objective_function = "wierstrass_function"
        run_with_plot(objective_function, main, "aco")

    if args.rotated:
        objective_function = "rotated_high_conditional_elliptic_function"
        run_with_plot(objective_function, main, "aco")
