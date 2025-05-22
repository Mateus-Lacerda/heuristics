import json

import matplotlib.pyplot as plt


def run_with_plot(
    objective_function: str, main: callable, algo: str
):
    results = main(objective_function=objective_function)
    for i, config in enumerate(results):
        print(f"Config {i+1}: {config}")

    plt.figure(figsize=(12, 6))

    for config in results:
        plt.plot(config["results"]["history"],
                 label=f"Config {results.index(config)+1}")

    plt.xlabel("Iterações")
    plt.ylabel("Melhor valor da função (Best Fitness)")
    plt.title(f"Convergência do PSO para {objective_function}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{algo}_convergence_{objective_function}.png")

    with open(f"{algo}_{objective_function}_results.json", "w") as f:
        json.dump(results, f, indent=4)
