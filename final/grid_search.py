import random
from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from meta_heuristics_fs import AntColonyOptimizationFS

def plot_search_results(results: list):
    """
    Gera e exibe gráficos para analisar os resultados da busca de hiperparâmetros.
    """
    if not results:
        print("Nenhum resultado válido para plotar.")
        return

    # 1. Converter a lista de resultados em um DataFrame do Pandas para fácil manipulação
    df = pd.DataFrame(results)
    params_df = df['params'].apply(pd.Series)
    df = pd.concat([df.drop('params', axis=1), params_df], axis=1)

    print("\n--- Análise Gráfica dos Resultados ---")

    # --- Gráfico 1: MSE vs. Número de Features ---
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df,
        x='num_features',
        y='mse',
        hue='evaporation_rate',
        size='Q',
        palette='viridis',
        sizes=(50, 250),
        alpha=0.8
    )
    best_run = df.loc[df['mse'].idxmin()]
    plt.scatter(best_run['num_features'], best_run['mse'], color='red',
                s=200, edgecolor='black', zorder=5, label='Melhor Resultado')
    plt.title('MSE Final vs. Número de Features Selecionadas', fontsize=16)
    plt.xlabel('Número de Features', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.legend(title="Legenda")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    plt.savefig('plot1.png')

    # --- Gráfico 2: Impacto de cada Hiperparâmetro no MSE ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impacto dos Hiperparâmetros no MSE Final', fontsize=18)
    hyperparameters = ['n_ants', 'iterations', 'evaporation_rate', 'Q']
    plot_titles = ['Nº de Formigas', 'Nº de Iterações',
                   'Taxa de Evaporação', 'Intensidade de Feromônio (Q)']
    for i, (param, title) in enumerate(zip(hyperparameters, plot_titles)):
        ax = axes.flatten()[i]
        sns.boxplot(ax=ax, data=df, x=param, y='mse', palette='coolwarm')
        ax.set_title(f'MSE por {title}', fontsize=14)
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.savefig('plot2.png')

    # --- NOVO GRÁFICO 3: Impacto da Penalização de Features ---
    # Este gráfico compara diretamente o MSE final quando a penalização de features está ativa ou não.
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x='penalize_parameters_size',
        y='mse',
        palette='pastel'
    )
    plt.title('Impacto da Penalização no MSE Final', fontsize=16)
    plt.xlabel('Penalizar por Número de Features', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    # Mapeia os labels do eixo x para nomes mais descritivos
    plt.xticks(ticks=[False, True], labels=['Não (False)', 'Sim (True)'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    plt.savefig('plot3.png')


def grid_search(
    x_cal, y_cal, x_test, y_test, cost_function: Callable
):
    param_grid = {
        'n_ants': [50, 100, 200],
        'iterations': [50, 100, 150],
        'evaporation_rate': [0.1, 0.3, 0.5, 0.7],
        'Q': [0.1, 0.3, 0.5],
        'penalize_parameters_size': [True, False]
    }

    results = []
    num_searches = 20

    for i in range(num_searches):
        params = {
            'n_ants': random.choice(param_grid['n_ants']),
            'iterations': random.choice(param_grid['iterations']),
            'evaporation_rate': random.choice(param_grid['evaporation_rate']),
            'Q': random.choice(param_grid['Q']),
            'penalize_parameters_size': random.choice(param_grid['penalize_parameters_size']),
        }
        print(f"Buscando com os parâmetros: {params}")

        aco = AntColonyOptimizationFS(
            columns_list=[f'feature_{i}' for i in range(x_cal.shape[1])],
            data_dict={
                0: {'x_train': x_cal, 'y_train': y_cal, 'x_test': x_test, 'y_test': y_test}
            },
            use_validation_data=False,
            penalize_parameters_size=params['penalize_parameters_size'],
            model=LinearRegression(),
            cost_function_improvement='decrease',
            cost_function=cost_function,
            average=None,
            n_ants=params['n_ants'],
            iterations=params['iterations'],
            evaporation_rate=params['evaporation_rate'],
            Q=params['Q'],
            run_time=5,
        )

        best_features = aco.GetBestFeatures()

        if not best_features or not best_features[0]:
            print("Nenhuma feature selecionada, pulando.")
            results.append(
                {'params': params, 'mse': float('inf'), 'num_features': 0})
            continue

        feature_names = best_features

        final_model = LinearRegression()
        final_model.fit(x_cal[feature_names], y_cal)
        predictions = final_model.predict(x_test[feature_names])

        final_mse = mean_squared_error(y_test, predictions)

        results.append({
            'params': params,
            'mse': final_mse,
            'num_features': len(best_features)
        })
        print(f"MSE final: {final_mse:.4f} com {
              len(best_features)} features.\n")

    # Filtra execuções que não retornaram features
    valid_results = [r for r in results if r['mse'] != float('inf')]

    if not valid_results:
        print("Nenhuma execução produziu um resultado válido.")
        return

    # Encontra a melhor combinação
    best_result = min(valid_results, key=lambda x: x['mse'])

    print("\n--- Melhor Resultado Encontrado ---")
    print(f"Melhores Parâmetros: {best_result['params']}")
    print(f"Menor MSE no Teste: {best_result['mse']:.4f}")
    print(f"Número de Features: {best_result['num_features']}")

    plot_search_results(valid_results)


if __name__ == '__main__':
    from pipeline import prepare_data, cost_function
    x_cal, y_cal, x_test, y_test, _, _ = prepare_data()
    grid_search(
        x_cal, y_cal, x_test, y_test, cost_function
    )
