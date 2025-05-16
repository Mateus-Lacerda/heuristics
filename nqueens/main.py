import math
import random
import time


def cost_function(state):
    """
    Essa funcao verifica quantas rainhas estao ameacadas.
    """
    cost = 0
    # Coluna => Rainha que estamos vendo se esta ameacada
    for col in range(len(state)):
        # Linha => Rainha que pode estar ameacando
        for row in range(len(state)):
            if col != row:
                if state[col] == state[row]:
                    cost += 1
                    continue
                if abs(col - row) == abs(state[col] - state[row]):
                    cost += 1
    return cost


def get_neighbors(state):
    neighbours = []
    for col in range(len(state)):
        for row in range(len(state)):
            if state[col] != row:
                neighbour = state.copy()
                neighbour[col] = row
                neighbours.append(neighbour)
    return neighbours


def simple_hill_climbing(initial_state, max_iterations=1000, stop_on_max_local=False):
    """
    Implements Simple Hill Climbing algorithm

    Args:
        initial_state (list): Starting point for the algorithm
        max_iterations (int): Maximum number of iterations to prevent infinite loops

    Returns:
        tuple: (best_state, best_value) found by the algorithm
    """
    current_state = initial_state
    current_value = cost_function(current_state)
    costs = [current_value]

    for _ in range(max_iterations):
        # Get neighboring states
        neighbors = get_neighbors(current_state)

        # Flag to check if we found a better neighbor
        found_better = False

        # Check neighbors one by one (Simple Hill Climbing)
        for neighbor in neighbors:
            neighbor_value = cost_function(neighbor)
            # If t we find a better neighbor, move to it immediately
            if neighbor_value < current_value:
                costs.append(neighbor_value)
                current_state = neighbor
                current_value = neighbor_value
                found_better = True
                break

        if not found_better:
            break

    return current_state, costs


def simulated_annealing(
    state, t_zero=10000,
    cool_down=0.95, max_iterations=1000, step=1
):
    best_solution = state
    best_cost = cost_function(state)
    costs = [best_cost]
    stop_on_plateu = 0

    while t_zero > 0.1:
        neighbors = get_neighbors(state)

        for neighbor in neighbors:
            neighbor_cost = cost_function(neighbor)

            # Probabilidade de aceitação de um estado pior
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost
                state = neighbor
                stop_on_plateu = 0
            else:
                # Aceita com uma probabilidade baseada na temperatura
                prob = math.exp((best_cost - neighbor_cost) / t_zero)
                if random.random() < prob:
                    best_solution = neighbor
                    best_cost = neighbor_cost
                    state = neighbor
                    stop_on_plateu = 0

        costs.append(best_cost)
        t_zero *= cool_down

        if best_cost == 0:
            break

        # Se não houver melhoria em um número grande de iterações, pare.
        if stop_on_plateu > 20:
            break

    return best_solution, costs


def run_once() -> tuple[dict, dict]:
    """
    Run the hill climbing and simulated annealing algorithms once
    and return the results.
    """
    initial_state = [random.randint(0, 7) for _ in range(8)]
    hc_it = time.time()
    hill_climbing_result = simple_hill_climbing(initial_state)
    hc_time = time.time() - hc_it

    sa_it = time.time()
    simulated_annealing_result = simulated_annealing(initial_state)
    sa_time = time.time() - sa_it

    return {
        "hill_climbing": {
            "state": hill_climbing_result[0],
            "time": hc_time,
            "scores": hill_climbing_result[1]
        },
        "simulated_annealing": {
            "state": simulated_annealing_result[0],
            "time": sa_time,
            "scores": simulated_annealing_result[1]
        }
    }


def main():
    """
    Main function to run the algorithms and print the results.
    """
    hill_climbing_results = []
    simulated_annealing_results = []

    for _ in range(500):
        results = run_once()
        hill_climbing_results.append(results["hill_climbing"])
        simulated_annealing_results.append(results["simulated_annealing"])

    # Print the results
    print("Resultados do Hill Climbing:")
    avg_time_hc = sum(result["time"] for result in hill_climbing_results) / len(hill_climbing_results)
    avg_score_hc = sum(result["scores"][-1] for result in hill_climbing_results) / len(hill_climbing_results)
    print(f"Tempo médio: {avg_time_hc:.4f} segundos")
    print(f"Pontuação média: {avg_score_hc:.4f}")
    melhor_hc = min(hill_climbing_results, key=lambda x: x["scores"][-1])
    pior_hc = max(hill_climbing_results, key=lambda x: x["scores"][-1])
    porcentagem_acerto_hc = (len([result for result in hill_climbing_results if result["scores"][-1] == 0]) / len(hill_climbing_results)) * 100

    print("Melhor estado:", melhor_hc["state"])
    print("Melhor pontuação:", melhor_hc["scores"][-1])
    print("Melhor tempo:", melhor_hc["time"])
    print("Pior pontuação:", pior_hc["scores"][-1])
    print("Pior tempo:", pior_hc["time"])
    print(f"Porcentagem de acertos: {porcentagem_acerto_hc:.2f}%")
    print()
    print("Resultados do Simulated Annealing:")
    avg_time_sa = sum(result["time"] for result in simulated_annealing_results) / len(simulated_annealing_results)
    avg_score_sa = sum(result["scores"][-1] for result in simulated_annealing_results) / len(simulated_annealing_results)
    print(f"Tempo médio: {avg_time_sa:.4f} segundos")
    print(f"Pontuação média: {avg_score_sa:.4f}")
    melhor_sa = min(simulated_annealing_results, key=lambda x: x["scores"][-1])
    pior_sa = max(simulated_annealing_results, key=lambda x: x["scores"][-1])
    porcentagem_acerto_sa = (len([result for result in simulated_annealing_results if result["scores"][-1] == 0]) / len(simulated_annealing_results)) * 100

    print("Melhor estado:", melhor_sa["state"])
    print("Melhor pontuação:", melhor_sa["scores"][-1])
    print("Melhor tempo:", melhor_sa["time"])
    print("Pior pontuação:", pior_sa["scores"][-1])
    print("Pior tempo:", pior_sa["time"])
    print(f"Porcentagem de acertos: {porcentagem_acerto_sa:.2f}%")
    print()


if __name__ == "__main__":
    main()

# initial_state = [random.randint(0, 7) for _ in range(8)]
# print(initial_state)
# print(simple_hill_climbing(initial_state))
# print(simulated_annealing(initial_state))
# # neighbours = hill_cimb(initial_state)
# # print("Initial State:", initial_state)
# # print("Generated Neighbours:")
# # for neighbour in neighbours:
# #     print(neighbour)
