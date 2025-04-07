import random
import math

import matplotlib.pyplot as plt

def objective_function(state):
    """
    Essa funcao verifica quantas rainhas estao ameacadas.
    """
    score = len(state) * (len(state) - 1)
    # Coluna => Rainha que estamos vendo se esta ameacada
    for col in range(len(state)):
        # Linha => Rainha que pode estar ameacando
        for row in range(len(state)):
            if col != row:
                if state[col] == state[row]:
                    score -= 1
                    continue
                if abs(col - row) == abs(state[col] - state[row]):
                    score -= 1
    return score


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
    current_value = objective_function(current_state)
    scores = [current_value]


    for _ in range(max_iterations):
        # Get neighboring states
        neighbors = get_neighbors(current_state)

        # Flag to check if we found a better neighbor
        found_better = False

        # Check neighbors one by one (Simple Hill Climbing)
        for neighbor in neighbors:
            neighbor_value = objective_function(neighbor)
            # If t we find a better neighbor, move to it immediately
            if neighbor_value > current_value:
                scores.append(neighbor_value)
                current_state = neighbor
                current_value = neighbor_value
                found_better = True
                break 

        if neighbor_value == 56:
            break
        # if not found_better:
            # break

    return current_state, scores


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

        current_cost = cost_function(state)

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


def make_comparison_plot(costs1, costs2):

    plt.plot(costs1, label='Simple Hill Climbing')
    plt.plot(costs2, label='Simulated Annealing')
    # Align the x-axis and y-axis to 0,0
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Comparison of Optimization Algorithms')
    plt.legend()
    plt.savefig('comparison_plot.png')


def main():
    initial_state = [random.randint(0, 7) for _ in range(8)]
    hill_climb_result = simple_hill_climbing(initial_state)
    sa_result = simulated_annealing(initial_state)

    make_comparison_plot(hill_climb_result[1], sa_result[1])

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
