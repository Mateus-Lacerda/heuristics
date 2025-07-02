import pandas as pd
import scipy.io
from sklearn.linear_model import LinearRegression

from meta_heuristics_fs import AntColonyOptimizationFS


def load_data(filepath: str):
    """Carrega os dados do arquivo .mat e os retorna."""
    raw_data = scipy.io.loadmat(filepath)

    data = {}

    data.update({
        'X_cal': raw_data['inputCalibration'].copy(),
        'y_cal': raw_data['targetCalibration'].ravel(),
        'X_test': raw_data['inputTest'].copy(),
        'y_test': raw_data['targetTest'].ravel(),
        'X_val': raw_data['inputValidation'].copy(),
    })

    wavenumbers = raw_data['wl'].ravel()

    data['wavenumbers'] = wavenumbers

    print("Dados carregados com sucesso.")
    print(f"  Calibração (X, y): {data['X_cal'].shape}, {data['y_cal'].shape}")
    print(f"  Teste (X, y):      {data['X_test'].shape}, {data['y_test'].shape}")
    print(f"  Validação (X):   {data['X_val'].shape}")
    print(f"  Números de onda (cm-1): {data['wavenumbers'].shape}, de {data['wavenumbers'].min():.2f} a {data['wavenumbers'].max():.2f}")
    return data


def prepare_data(
    path='2012/ShootOut2012MATLAB/ShootOut2012MATLAB.mat'
) -> tuple:
    data = load_data(path)
    x_cal = pd.DataFrame(data['X_cal'], columns=[f'feature_{i}' for i in range(data['X_cal'].shape[1])])
    y_cal = pd.Series(data['y_cal'], name='target')
    x_test = pd.DataFrame(data['X_test'], columns=[f'feature_{i}' for i in range(data['X_test'].shape[1])])
    y_test = pd.Series(data['y_test'], name='target')
    x_val = pd.DataFrame(data['X_val'], columns=[f'feature_{i}' for i in range(data['X_val'].shape[1])])
    wavenumbers = data['wavenumbers']
    return x_cal, y_cal, x_test, y_test, x_val, wavenumbers


def cost_function(actual: pd.DataFrame, predicted: pd.DataFrame):
    """Cost function to calculate mean squared error."""
    return ((actual - predicted) ** 2).mean()


def find_best_features(config: dict = {"foo": "bar"}) -> list:
    x_cal, y_cal, x_test, y_test, _x_val, _wavenumbers = prepare_data()
    aco = AntColonyOptimizationFS(
        columns_list=[f'feature_{i}' for i in range(x_cal.shape[1])],
        data_dict={
            0: {
                'x_train': x_cal, 'y_train': y_cal, 'x_test': x_test, 'y_test': y_test
            }
        },
        use_validation_data=False,
        model=LinearRegression(),
        cost_function_improvement='decrease',
        cost_function=cost_function,
        average=None,
        iterations=100,
        n_ants=1000,
        run_time=1,  # in minutes
        evaporation_rate=0.5,
        Q=0.2
    )
    return aco.GetBestFeatures()

if __name__ == '__main__':
    best_features = find_best_features()
    print("Melhores features encontradas:", best_features)
