import scipy.io
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from typing import Dict, Any, Callable
import matplotlib.pyplot as plt


# --- Tipos e Configurações ---
NDArray = npt.NDArray[np.float64]
plt.style.use('seaborn-v0_8-whitegrid')  # Estilo profissional para os gráficos


# --- Classe da Pipeline ---

class Pipeline:
    """
    Uma classe para encapsular os dados e o estado de uma pipeline de machine learning,
    permitindo o encadeamento de operações de forma fluida com o operador '>>'.
    """
    def __init__(self, **kwargs: Any):
        self.data: Dict[str, Any] = kwargs
        self.model: Any = None
        self.metrics: Dict[str, float] = {}
        self.predictions: Dict[str, NDArray] = {}

    def __rshift__(self, other_func: Callable[['Pipeline'], 'Pipeline']) -> 'Pipeline':
        """Sobrecarga do operador >> para encadear funções na pipeline."""
        import functools
        func_name = other_func.func.__name__ if isinstance(other_func, functools.partial) else other_func.__name__
        print(f"--- Executando Etapa: {func_name} ---")
        return other_func(self)

    def __repr__(self) -> str:
        """Representação do objeto Pipeline."""
        keys = ", ".join(self.data.keys())
        return f"<Pipeline com dados: [{keys}]>"


# --- Funções de Pré-processamento (Heurísticas) ---

def multiplicative_scatter_correction(input_data: NDArray) -> NDArray:
    """Realiza a Correção Multiplicativa de Dispersão (MSC)."""
    mean_spectrum = np.mean(input_data, axis=0)
    corrected_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        spectrum = input_data[i, :]
        coeffs = np.polyfit(mean_spectrum, spectrum, 1)
        corrected_data[i, :] = (spectrum - coeffs[1]) / coeffs[0]
    return corrected_data


# --- Etapas da Pipeline ---

def carregar_dados(pipeline: Pipeline, filepath: str) -> Pipeline:
    """Carrega os dados do arquivo .mat e os adiciona à pipeline."""
    try:
        data = scipy.io.loadmat(filepath)
        pipeline.data.update({
            'X_cal_raw': data['inputCalibration'],
            'X_cal': data['inputCalibration'].copy(),
            'y_cal': data['targetCalibration'].ravel(),
            'X_test': data['inputTest'].copy(),
            'y_test': data['targetTest'].ravel(),
            'X_val': data['inputValidation'].copy(),
        })

        # Robustamente encontra a chave de número de onda/comprimento de onda
        wavenumbers_key = None
        possible_keys = ['wave', 'wl', 'wavenumbers', 'wavelengths']
        for key in possible_keys:
            if key in data:
                wavenumbers_key = key
                break

        if not wavenumbers_key:
            raise KeyError(
                "Nenhuma chave para número de onda/comprimento de onda encontrada. "
                f"Chaves tentadas: {possible_keys}"
            )

        wavenumbers = data[wavenumbers_key].ravel()

        # Heurística: Detecta se os dados estão em nanômetros (nm) e converte para cm-1.
        # Números de onda (cm-1) são tipicamente > 4000. Comprimentos de onda (nm) são < 3000.
        if np.max(wavenumbers) < 3000:
            print(f"  Aviso: Dados espectrais parecem estar em nanômetros (nm) (max: {np.max(wavenumbers):.2f}). Convertendo para números de onda (cm-1).")
            wavenumbers = 10**7 / wavenumbers
            # A conversão inverte a ordem, então revertemos o array e os dados
            wavenumbers = wavenumbers[::-1]
            for key in ['X_cal_raw', 'X_cal', 'X_test', 'X_val']:
                if key in pipeline.data:
                    pipeline.data[key] = np.ascontiguousarray(np.fliplr(pipeline.data[key]))

        pipeline.data['wavenumbers'] = wavenumbers

        print("Dados carregados com sucesso.")
        print(f"  Calibração (X, y): {pipeline.data['X_cal'].shape}, {pipeline.data['y_cal'].shape}")
        print(f"  Teste (X, y):      {pipeline.data['X_test'].shape}, {pipeline.data['y_test'].shape}")
        print(f"  Validação (X):   {pipeline.data['X_val'].shape}")
        print(f"  Números de onda (cm-1): {pipeline.data['wavenumbers'].shape}, de {pipeline.data['wavenumbers'].min():.2f} a {pipeline.data['wavenumbers'].max():.2f}")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filepath}' não foi encontrado.")
        exit()  # Termina a execução se os dados não forem encontrados
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        exit()
    return pipeline

def aplicar_msc(pipeline: Pipeline) -> Pipeline:
    """Aplica a Correção Multiplicativa de Dispersão (MSC) aos dados."""
    pipeline.data['X_cal'] = multiplicative_scatter_correction(pipeline.data['X_cal'])
    pipeline.data['X_test'] = multiplicative_scatter_correction(pipeline.data['X_test'])
    pipeline.data['X_val'] = multiplicative_scatter_correction(pipeline.data['X_val'])
    print("  - MSC aplicado a todos os conjuntos de dados.")
    return pipeline

def aplicar_derivada(pipeline: Pipeline) -> Pipeline:
    """Aplica a 1ª derivada (Savitzky-Golay) para realçar picos."""
    # Ensure window_length is odd and polyorder < window_length
    # Default values are 5 and 2, which are fine.
    window_length = 5
    polyorder = 2
    if pipeline.data['X_cal'].shape[1] < window_length:
        # Adjust if the number of features is too small after selection
        window_length = max(3, pipeline.data['X_cal'].shape[1] - 2 if pipeline.data['X_cal'].shape[1] > 2 else 1)
        polyorder = min(polyorder, window_length - 1)
        if window_length % 2 == 0: # Ensure odd
            window_length -= 1
        print(f"  Ajustando Savitzky-Golay: window_length={window_length}, polyorder={polyorder} devido ao número reduzido de features.")

    # Aplica a todos os datasets de forma consistente
    for key in ['X_cal', 'X_test', 'X_val']:
        pipeline.data[key] = savgol_filter(pipeline.data[key], window_length=window_length, polyorder=polyorder, deriv=1, axis=1)

    print("  - 1ª Derivada (Savitzky-Golay) aplicada.")
    return pipeline

def selecionar_features(pipeline: Pipeline, min_wn: float, max_wn: float) -> Pipeline:
    """Seleciona as features espectrais dentro de um intervalo de número de onda."""
    wavenumbers = pipeline.data['wavenumbers']
    indices = np.where((wavenumbers >= min_wn) & (wavenumbers <= max_wn))[0]

    if len(indices) == 0:
        raise ValueError(f"Nenhuma feature encontrada na faixa de número de onda {min_wn}-{max_wn} cm-1. Verifique os números de onda ou a faixa selecionada.")

    pipeline.data['X_cal'] = pipeline.data['X_cal'][:, indices]
    pipeline.data['X_test'] = pipeline.data['X_test'][:, indices]
    pipeline.data['X_val'] = pipeline.data['X_val'][:, indices]
    pipeline.data['wavenumbers'] = wavenumbers[indices]

    print(f"  - Features selecionadas. Usando {len(indices)} features na faixa de {min_wn}-{max_wn} cm-1.")
    return pipeline

def treinar_modelo(pipeline: Pipeline) -> Pipeline:
    """Treina um modelo de Regressão Linear."""
    model = LinearRegression()
    model.fit(pipeline.data['X_cal'], pipeline.data['y_cal'])
    pipeline.model = model
    print("  - Modelo de Regressão Linear treinado.")
    return pipeline

def avaliar_modelo(pipeline: Pipeline) -> Pipeline:
    """Avalia o modelo no conjunto de teste e calcula as métricas."""
    y_pred = pipeline.model.predict(pipeline.data['X_test'])
    pipeline.predictions['test'] = y_pred

    error = pipeline.data['y_test'] - y_pred
    r2 = np.corrcoef(pipeline.data['y_test'], y_pred)[0, 1] ** 2 if len(y_pred) > 1 else 1.0

    pipeline.metrics = {
        "rmsep": np.sqrt(mean_squared_error(pipeline.data['y_test'], y_pred)),
        "sep": np.std(error),
        "bias": np.mean(error),
        "r2": r2,
    }
    print("  Performance no Conjunto de Teste:")
    print(f"    RMSEP: {pipeline.metrics['rmsep']:.4f}")
    print(f"    R²:    {pipeline.metrics['r2']:.4f}")
    return pipeline


# --- Etapas de Visualização ---

def plotar_espectros(pipeline: Pipeline, titulo: str, chave_x: str) -> Pipeline:
    """Plota os espectros de um determinado conjunto de dados na pipeline."""
    # Check if wavenumbers are available and if the data exists
    if 'wavenumbers' not in pipeline.data or chave_x not in pipeline.data:
        print(f"Aviso: Não foi possível plotar '{titulo}'. Dados ou números de onda ausentes.")
        return pipeline

    # Ensure the data has at least one sample to plot
    if pipeline.data[chave_x].shape[0] == 0:
        print(f"Aviso: Não há amostras para plotar em '{titulo}'.")
        return pipeline
    plt.figure(figsize=(10, 6))
    num_samples_to_plot = min(10, pipeline.data[chave_x].shape[0])
    plt.plot(pipeline.data['wavenumbers'], pipeline.data[chave_x][:num_samples_to_plot].T)
    plt.title(titulo, fontsize=16)
    plt.xlabel("Número de Onda (cm⁻¹)")
    plt.ylabel("Absorbância (unidade arbitrária)")
    plt.gca().invert_xaxis()  # Eixo X invertido, comum em espectroscopia
    plt.show()
    return pipeline

def plotar_predicoes(pipeline: Pipeline) -> Pipeline:
    """Plota os valores reais vs. preditos para o conjunto de teste."""
    if 'y_test' not in pipeline.data or 'test' not in pipeline.predictions:
        print("Aviso: Não foi possível plotar predições. Dados de teste ou predições ausentes.")
        return pipeline
    y_true = pipeline.data['y_test']
    y_pred = pipeline.predictions['test']

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    # Linha de referência y=x
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)

    plt.title("Valores Reais vs. Preditos (Conjunto de Teste)", fontsize=16)
    plt.xlabel("Valores Reais (% w/w)")
    plt.ylabel("Valores Preditos (% w/w)")
    plt.gca().set_aspect('equal', adjustable='box')

    # Adiciona métricas ao gráfico
    metrics_text = (
        f"RMSEP = {pipeline.metrics['rmsep']:.3f}\n"
        f"R² = {pipeline.metrics['r2']:.3f}\n"
        f"Bias = {pipeline.metrics['bias']:.3f}"
    )
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.show()
    return pipeline


# --- Função Principal ---

def main() -> None:
    """
    Função principal que executa a pipeline de análise de dados espectrais
    inspirada no desafio IDRC 2012.
    """
    import functools
    mat_file_path = '2012/ShootOut2012MATLAB/ShootOut2012MATLAB.mat'
    # A pipeline é definida como uma sequência de etapas encadeadas
    pipeline_final = (
        Pipeline()
        >> functools.partial(carregar_dados, filepath=mat_file_path)
        >> functools.partial(plotar_espectros, titulo="Espectros Originais (Calibração)", chave_x='X_cal_raw')
        >> aplicar_msc
        >> functools.partial(plotar_espectros, titulo="Espectros Após MSC (Calibração)", chave_x='X_cal')
        >> aplicar_derivada
        >> functools.partial(plotar_espectros, titulo="Espectros Após 1ª Derivada (Calibração)", chave_x='X_cal')
        >> functools.partial(selecionar_features, min_wn=8700, max_wn=8950)
        >> treinar_modelo
        >> avaliar_modelo
        >> plotar_predicoes
    )
    # Predição final no conjunto de validação (não temos os valores reais para comparar)
    # Check if model and X_val are available before predicting
    if pipeline_final.model and 'X_val' in pipeline_final.data:
        y_val_pred = pipeline_final.model.predict(pipeline_final.data['X_val'])
        print("\n--- Predição Final ---")
        print(f"  - Predições geradas para as {len(y_val_pred)} amostras de validação.")
        # np.savetxt("validation_predictions.csv", y_val_pred, delimiter=",") # Uncomment to save predictions
    else:
        print("\n--- Predição Final ---")
        print("  - Não foi possível gerar predições para o conjunto de validação (modelo ou dados ausentes).")
    print("\nDesafio finalizado com uma pipeline profissional!")

# if __name__ == "__main__":
#     main()
